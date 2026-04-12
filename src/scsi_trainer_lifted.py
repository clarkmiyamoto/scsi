"""
Self-Consistent Stochastic Interpolant (SCSI) Trainer.

Implements Algorithm 2 from the paper:
    Modi, Han, Vanden-Eijnden, Bruna (2025).
    "Generative Modeling from Black-box Corruptions
     via Self-Consistent Stochastic Interpolants."

The key idea: we never access clean data.  Instead we iteratively
    1.  backward-transport corrupted observations  y ~ mu  to get
        pseudo-clean samples  x = Phi_Theta(y),
    2.  re-corrupt them through the forward model  y_tilde = F(x),
    3.  build a stochastic interpolant between x and y_tilde,
    4.  update the velocity (and optionally denoiser) networks via SI losses.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain
import math
from typing import Any

import torch
from torch import Tensor

from distribution import Distribution
from forward import ForwardModel
from interpolant import Interpolant
from inference import ODE, SDE, EulerSimulator, EulerMaruyamaSimulator, select_schedule


def _cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SCSIConfig:
    """Hyperparameters for the SCSI bi-level training loop.

    Attributes:
        outer_iterations:      K   – number of outer-loop iterations.
                               Paper uses 20,000 for 2-D and 50,000 for imaging.
        inner_steps:           T_tr – SGD steps per outer iteration.
                               Paper finds T_tr=1 sufficient.
        batch_size:            Number of corrupted samples per SGD step.
        lr:                    Learning rate (paper: 5e-4).
        warmup_steps:          Linear warmup steps for cosine LR schedule.
        p_mixture:             Probability of using re-corrupted F(Phi(y))
                               vs. the original observation y (Section C.2).
        schedule_type:         Time discretization schedule type.
                               Choose from: {AVAILABLE_SCHEDULES}.
        num_transport_steps:   Number of ODE integration steps for the
                               backward transport  Phi_Theta(y).  Paper: 64.
        num_resamples:         For each backward-transported x, how many
                               times to re-sample the forward model
                               (amortises the ODE cost; paper uses 2).
        log_every:             Print loss every this many *inner* steps.
    """
    outer_iterations: int = 20000
    inner_steps: int = 1
    batch_size: int = 4096
    lr: float = 5e-4
    warmup_steps: int = 500
    p_mixture: float = 0.9
    schedule_type: str = "uniform"
    num_transport_steps: int = 64
    num_resamples: int = 2
    log_every: int = 1000


# ---------------------------------------------------------------------------
# SCSI Trainer (ODE mode)
# ---------------------------------------------------------------------------

class SCSITrainer:
    """
    Bi-level trainer for the Self-Consistent Stochastic Interpolant.

    This implements Algorithm 2 (Appendix B.1) for the ODE setting
    (epsilon = 0, gamma_t = 0, only a drift network b is trained).

    The interpolant object is used only for its schedule coefficients
    (alpha, beta, gamma) and the `interpolant` / `interpolant_dot`
    evaluation helpers.  Its `sample()` method is NOT called –
    training pairs (x0, x1) are built internally.
    """

    def __init__(
        self,
        observation_dist: Distribution,
        auxiliary_dist: Distribution,
        forward_model: ForwardModel,
        interpolant: Interpolant,
        drift_model: torch.nn.Module,
        config: SCSIConfig,
        log_fn: Callable[[dict[str, Any]], None] | None = None,
    ):
        """
        Args:
            observation_dist:  Distribution mu of corrupted observations.
            auxiliary_dist:    Distribution pi for lifted interpolant.
            forward_model:     Black-box forward map F.
            interpolant:       Interpolant providing the schedule (alpha, beta, gamma).
            drift_model:       Neural network b_theta(x, t, cond) predicting dI/dt.
            config:            Training hyperparameters.
            log_fn:            Optional callback for external logging (e.g. wandb.log).
                               Called with a dict of metrics at each log interval.
        """
        self.obs_dist = observation_dist
        self.auxiliary_dist = auxiliary_dist
        self.F = forward_model
        self.interpolant = interpolant
        self.drift_model = drift_model
        self.config = config
        self.log_fn = log_fn
        self.loss_fn = torch.nn.MSELoss()

        # Cache ODE / simulator so we don't re-allocate every step
        self._ode = ODE(self.drift_model)
        self._simulator = EulerSimulator(self._ode)

        self._validate_initalization()

    # ------------------------------------------------------------------
    # Backward transport:  Phi_Theta(y)  –  ODE from t=1 to t=0
    # ------------------------------------------------------------------
    
    def _validate_initalization(self):
        pass

        
    @torch.no_grad()
    def _backward_transport(self, y: Tensor, w: Tensor) -> Tensor:
        """Map corrupted observations y back to pseudo-clean x.

        Uses the current drift_model to integrate the probability-flow
        ODE backward from t=1 (observations) to t=0 (data).

        Args:
            y: (B, dim) tensor of corrupted observations.
            w: (B, dim) tensor of auxiliary observations.
        Returns:
            (B, dim) tensor of pseudo-clean samples.
        """
        schedule = select_schedule(self.config.schedule_type)(self.config.num_transport_steps)
        x = self._simulator.solve_backwards(w, schedule, conditional=y)
        return x

    # ------------------------------------------------------------------
    # Single inner SGD step
    # ------------------------------------------------------------------

    def _inner_step(self, optimizer: torch.optim.Optimizer) -> float:
        """One SGD update on the SI velocity loss (Eq. 3 in the paper).

        Returns:
            Scalar loss value.
        """
        cfg = self.config
        B = cfg.batch_size

        # 1. Sample corrupted observations  y ~ mu
        y = self.obs_dist.sample(B)   # (B, dim_Y)

        # 1.1 Sample auxiliary observations  z ~ pi
        w = self.auxiliary_dist.sample(B)   # (B, dim_X)

        # 2. Backward transport to get pseudo-clean  x = Phi_Theta(y)
        x = self._backward_transport(y, w)  # (B, dim_X)  -- stop-gradient

        # 3. Re-corrupt through the forward model  y_tilde = F(x)
        #    Re-sample num_resamples times to amortise ODE cost.
        x_all = x.repeat(cfg.num_resamples, 1)        # (B*R, dim_X)
        y_all = y.repeat(cfg.num_resamples, 1)         # (B*R, dim_Y)

        with torch.no_grad():
            y_tilde = self.F(x_all)                    # (B*R, dim_Y)

        # 4. Mixture trick (Section C.2):
        #    with probability p, use y_tilde; with probability 1-p, use original y
        mask = (torch.rand(x_all.shape[0], 1, device=x_all.device) < cfg.p_mixture).float()
        y_tilde_masked = mask * y_tilde + (1 - mask) * y_all      # (B*R, dim_Y)

        # 5. Build the interpolant and its time-derivative
        n = x_all.shape[0]
        ts = torch.rand(n, device=x_all.device)        # (n,)
        zs = torch.zeros_like(x_all)                  # (n, dim_X)

        w_prime = self.auxiliary_dist.sample(n) # (n, dim)

        #    I_t   = alpha_t * x_all +  beta_t * w_prime    +  gamma_t * z
        I = self.interpolant.interpolant(ts, x_all, w_prime, zs)

        #    dI/dt = alpha_dot * x_all +  beta_dot * w_prime  +  gamma_dot * z
        dIdt = self.interpolant.interpolant_dot(ts, x_all, w_prime, zs)

        # 6. Compute loss and SGD update
        optimizer.zero_grad()
        pred = self.drift_model(I, ts, conditional=y_tilde_masked)
        loss = self.loss_fn(pred, dIdt)
        loss.backward()
        optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Full bi-level training loop
    # ------------------------------------------------------------------

    def train(self) -> list[float]:
        """Run the SCSI bi-level training loop (Algorithm 2).

        Returns:
            List of per-step loss values.
        """
        cfg = self.config
        total_steps = cfg.outer_iterations * cfg.inner_steps
        self.drift_model.train()

        optimizer = torch.optim.Adam(
            self.drift_model.parameters(),
            lr=cfg.lr,
        )
        # Cosine schedule with linear warmup (Section C.2 of the paper)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_warmup_lambda(
                step, cfg.warmup_steps, total_steps,
            ),
        )

        losses: list[float] = []
        global_step = 0

        for k in range(cfg.outer_iterations):
            # Inner loop: T_tr SGD steps with the *current* transport map
            for i in range(cfg.inner_steps):
                loss = self._inner_step(optimizer)
                scheduler.step()
                losses.append(loss)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(
                        f"[outer {k+1}/{cfg.outer_iterations}  "
                        f"inner {i+1}/{cfg.inner_steps}]  "
                        f"step {global_step}  loss {loss:.6f}  "
                        f"lr {lr_now:.2e}"
                    )
                    if self.log_fn is not None:
                        self.log_fn({
                            "loss": loss,
                            "lr": lr_now,
                            "global_step": global_step,
                        })

            # (In the paper, Theta^(k) <- Theta here.  Since we use T_tr=1
            #  and a single model, this is implicit – the next backward
            #  transport already uses the updated parameters.)

        print(f"Training complete.  {total_steps} steps, final loss {losses[-1]:.6f}")
        return losses