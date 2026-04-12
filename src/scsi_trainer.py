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
        forward_model: ForwardModel,
        interpolant: Interpolant,
        drift_model: torch.nn.Module,
        config: SCSIConfig,
        log_fn: Callable[[dict[str, Any]], None] | None = None,
    ):
        """
        Args:
            observation_dist:  Distribution mu of corrupted observations.
            forward_model:     Black-box forward map F.
            interpolant:       Interpolant providing the schedule (alpha, beta, gamma).
            drift_model:       Neural network b_theta(x, t) predicting dI/dt.
            config:            Training hyperparameters.
            log_fn:            Optional callback for external logging (e.g. wandb.log).
                               Called with a dict of metrics at each log interval.
        """
        self.obs_dist = observation_dist
        self.F = forward_model
        self.interpolant = interpolant
        self.drift_model = drift_model
        self.config = config
        self.log_fn = log_fn
        self.loss_fn = torch.nn.MSELoss()

        # Cache ODE / simulator so we don't re-allocate every step
        self._ode = ODE(self.drift_model)
        self._simulator = EulerSimulator(self._ode)

    # ------------------------------------------------------------------
    # Backward transport:  Phi_Theta(y)  –  ODE from t=1 to t=0
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _backward_transport(self, y: Tensor) -> Tensor:
        """Map corrupted observations y back to pseudo-clean x.

        Uses the current drift_model to integrate the probability-flow
        ODE backward from t=1 (observations) to t=0 (data).
        """
        schedule = select_schedule(self.config.schedule_type)(self.config.num_transport_steps)
        x = self._simulator.solve_backwards(y, schedule)
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
        y = self.obs_dist.sample(B)   # (B, dim)

        # 2. Backward transport to get pseudo-clean  x = Phi_Theta(y)
        x = self._backward_transport(y)  # (B, dim)  -- stop-gradient

        # 3. Re-corrupt through the forward model  y_tilde = F(x)
        #    Re-sample num_resamples times to amortise ODE cost.
        x_all = x.repeat(cfg.num_resamples, 1)        # (B*R, dim)
        y_all = y.repeat(cfg.num_resamples, 1)         # (B*R, dim)

        with torch.no_grad():
            y_tilde = self.F(x_all)                    # (B*R, dim)

        # 4. Mixture trick (Section C.2):
        #    with probability p, use y_tilde; with probability 1-p, use original y
        mask = (torch.rand(x_all.shape[0], 1, device=x_all.device) < cfg.p_mixture).float()
        x1 = mask * y_tilde + (1 - mask) * y_all      # (B*R, dim)

        # 5. Build the interpolant and its time-derivative
        n = x_all.shape[0]
        ts = torch.rand(n, device=x_all.device)        # (n,)
        zs = torch.randn_like(x_all)                   # (n, dim)

        #    I_t  = alpha_t * x  +  beta_t * x1  +  gamma_t * z
        I = self.interpolant.interpolant(ts, x_all, x1, zs)

        #    dI/dt = alpha_dot * x  +  beta_dot * x1  +  gamma_dot * z
        dIdt = self.interpolant.interpolant_dot(ts, x_all, x1, zs)

        # 6. Compute loss and SGD update
        optimizer.zero_grad()
        pred = self.drift_model(I, ts)
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


# ---------------------------------------------------------------------------
# SCSI Trainer (SDE mode)
# ---------------------------------------------------------------------------

class SCSITrainerSDE:
    """
    Bi-level SCSI trainer for the SDE setting (epsilon > 0).

    Learns both a drift b(x,t) predicting dI/dt and a denoiser g(x,t)
    predicting the noise z, via the losses in Eq. (3) and (4).
    """

    def __init__(
        self,
        observation_dist: Distribution,
        forward_model: ForwardModel,
        interpolant: Interpolant,
        drift_model: torch.nn.Module,
        denoiser_model: torch.nn.Module,
        config: SCSIConfig,
        denoiser_weight: float = 1.0,
        log_fn: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.obs_dist = observation_dist
        self.F = forward_model
        self.interpolant = interpolant
        self.drift_model = drift_model
        self.denoiser_model = denoiser_model
        self.config = config
        self.denoiser_weight = denoiser_weight
        self.log_fn = log_fn
        self.loss_fn = torch.nn.MSELoss()

        # Cache ODE / simulator so we don't re-allocate every step
        self._sde = SDE(self.drift_model, self.denoiser_model, self.interpolant, self.interpolant.gamma)
        self._simulator = EulerMaruyamaSimulator(self._sde)

    @torch.no_grad()
    def _backward_transport(self, y: Tensor) -> Tensor:
        """Map corrupted observations y back to pseudo-clean x via backward ODE.

        Note: even in SDE training mode, we use the ODE (drift-only) for
        the backward transport inside the training loop, which is standard
        practice in the paper.
        """
        schedule = select_schedule(self.config.schedule_type)(self.config.num_transport_steps)
        x = self._simulator.solve_backwards(y, schedule)
        return x

    def _inner_step(self, optimizer: torch.optim.Optimizer) -> float:
        cfg = self.config
        B = cfg.batch_size

        y = self.obs_dist.sample(B)
        x = self._backward_transport(y)

        x_all = x.repeat(cfg.num_resamples, 1)
        y_all = y.repeat(cfg.num_resamples, 1)

        with torch.no_grad():
            y_tilde = self.F(x_all)

        mask = (torch.rand(x_all.shape[0], 1, device=x_all.device) < cfg.p_mixture).float()
        x1 = mask * y_tilde + (1 - mask) * y_all

        n = x_all.shape[0]
        ts = torch.rand(n, device=x_all.device)
        zs = torch.randn_like(x_all)

        I = self.interpolant.interpolant(ts, x_all, x1, zs).detach()
        dIdt = self.interpolant.interpolant_dot(ts, x_all, x1, zs).detach()

        optimizer.zero_grad()

        # Drift loss  (Eq. 3):  ||b(I_t, t) - dI/dt||^2
        drift_loss = self.loss_fn(self.drift_model(I, ts), dIdt)

        # Denoiser loss (Eq. 4):  ||g(I_t, t) - z||^2
        denoiser_loss = self.loss_fn(self.denoiser_model(I, ts), zs)

        loss = drift_loss + denoiser_loss
        loss.backward()
        optimizer.step()

        return drift_loss.item(), denoiser_loss.item()

    def train(self) -> list[float]:
        cfg = self.config
        total_steps = cfg.outer_iterations * cfg.inner_steps
        self.drift_model.train()
        self.denoiser_model.train()

        optimizer = torch.optim.Adam(
            chain(
                self.drift_model.parameters(),
                self.denoiser_model.parameters(),
            ),
            lr=cfg.lr,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_warmup_lambda(
                step, cfg.warmup_steps, total_steps,
            ),
        )

        drift_losses: list[float] = []
        denoiser_losses: list[float] = []
        global_step = 0

        for k in range(cfg.outer_iterations):
            for i in range(cfg.inner_steps):
                drift_loss, denoiser_loss = self._inner_step(optimizer)
                scheduler.step()
                drift_losses.append(drift_loss)
                denoiser_losses.append(denoiser_loss)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(
                        f"[outer {k+1}/{cfg.outer_iterations}  "
                        f"inner {i+1}/{cfg.inner_steps}]  "
                        f"step {global_step}  drift_loss {drift_loss:.6f}  denoiser_loss {denoiser_loss:.6f}  "
                        f"lr {lr_now:.2e}"
                    )
                    if self.log_fn is not None:
                        self.log_fn({
                            "total_loss": drift_loss + denoiser_loss,
                            "drift_loss": drift_loss,
                            "denoiser_loss": denoiser_loss,
                            "lr": lr_now,
                            "global_step": global_step,
                        })

        final_loss = drift_losses[-1] + denoiser_losses[-1]
        print(f"Training complete.  {total_steps} steps, final loss {final_loss:.6f}")
        return final_loss