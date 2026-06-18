"""Self-consistency training loop (the SCSI bootstrap).

Stripped port of ``src/trainer_si.py::Trainer`` keeping only the heart of the
algorithm: the two-phase schedule and the frozen-transport refresh. Dropped:
DDP, AMP, ``torch.compile``, the score/SDE path, loss-spike resets, SLURM
plumbing, and the clean-data "cheat" warmup.

Phases
------
* **warmup** (``step < warmup_steps``): the pseudo-clean target ``x_0`` is the
  honest FBP reconstruction of the observation (``warmup_target_fn``), optionally
  spread over the rotation orbit (``warmup_orbit_random_fn``).
* **bootstrap** (``step >= warmup_steps``): ``x_0`` is produced by transporting
  the observation through a *frozen* copy of the drift model, refreshed every
  ``transport_steps`` steps. This is the self-consistency loop.
"""
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


class EMA:
    """Minimal exponential-moving-average of model weights (replaces the
    ``ema_pytorch`` dependency)."""

    def __init__(self, model, beta: float = 0.999):
        self.beta = beta
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.ema_model.parameters(), model.parameters()):
            ep.mul_(self.beta).add_(p.detach(), alpha=1.0 - self.beta)
        for eb, b in zip(self.ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def _infinite(dl):
    while True:
        for batch in dl:
            yield batch


class Trainer:

    def __init__(self, model, interpolant, dataset, *, lr: float = 2e-4,
                 batch_size: int = 32, train_steps: int = 2000,
                 warmup_steps: int = 800, transport_steps: int = 200,
                 ema_decay: float = 0.999, results_folder: str = "./results_clean",
                 warmup_target_fn=None, warmup_orbit_random_fn=None,
                 save_every: int = 500, log_every: int = 50,
                 num_workers: int = 0, device=None, logger=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.interpolant = interpolant
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.transport_steps = transport_steps
        self.warmup_target_fn = warmup_target_fn
        self.warmup_orbit_random_fn = warmup_orbit_random_fn
        self.save_every = save_every
        self.log_every = log_every
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.ema = EMA(self.model, beta=ema_decay)
        self.dl = _infinite(DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, drop_last=True))
        self.logger = logger
        self.step = 0

    def train(self):
        device = self.device
        transport_map = None
        best = float("inf")
        losses = []
        print(f"Training {self.train_steps} steps on {device} "
              f"(warmup {self.warmup_steps}, refresh every {self.transport_steps})",
              flush=True)

        _clean_viz, obs_viz, cond_viz = next(self.dl)
        obs_viz = obs_viz.to(device)
        cond_viz = cond_viz.to(device)
        _clean_viz = _clean_viz.to(device)

        while self.step < self.train_steps:
            self.model.train()
            _clean, obs, cond = next(self.dl)
            obs, cond = obs.to(device), cond.to(device)

            if self.step < self.warmup_steps:
                with torch.no_grad():
                    x0 = self.warmup_target_fn(cond) if self.warmup_target_fn is not None else obs
                    if self.warmup_orbit_random_fn is not None:
                        x0 = self.warmup_orbit_random_fn(x0)
                loss = self.interpolant.loss_fn(self.model, obs, cond, x0=x0)
            else:
                # Refresh the frozen transport model at the warmup->bootstrap
                # transition and every transport_steps thereafter.
                if transport_map is None or self.step % self.transport_steps == 0:
                    transport_map = copy.deepcopy(self.model).eval()
                    for p in transport_map.parameters():
                        p.requires_grad_(False)
                    print(f"[step {self.step}] refreshed frozen transport model", flush=True)
                    with torch.no_grad():
                        fbp_viz = self.warmup_target_fn(cond_viz) if self.warmup_target_fn else obs_viz
                        recon_viz = self.interpolant.transport(self.ema.ema_model, obs_viz, cond_viz)
                    self.logger.log_recon(self.step, _clean_viz, fbp_viz, recon_viz)
                loss = self.interpolant.loss_fn(self.model, obs, cond, b_fixed=transport_map)

            self.opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.ema.update(self.model)

            self.step += 1
            lval = loss.item()
            losses.append(lval)
            phase = "warmup" if self.step <= self.warmup_steps else "bootstrap"
            self.logger.log_step(self.step, lval, grad_norm.item(), phase)
            if self.step % self.log_every == 0:
                print(f"step {self.step}/{self.train_steps} [{phase}]  loss {lval:.4f}", flush=True)
            if lval < best:
                best = lval
                self.save("model-best.pt")
            if self.step % self.save_every == 0:
                self.save("model-latest.pt")

        self.save("model-latest.pt")
        np.save(self.results_folder / "losses.npy", np.asarray(losses))
        print(f"Done. Best loss {best:.4f}. Artifacts in {self.results_folder}", flush=True)

    def save(self, name: str):
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.ema_model.state_dict(),
        }, self.results_folder / name)
