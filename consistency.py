"""
Flow Map Matching (FMM) for the two-moons distribution.

A minimal implementation of the consistency-model-style framework from
Boffi, Albergo & Vanden-Eijnden, "Flow map matching with stochastic interpolants" (2024).

We learn a TWO-TIME flow map X_{s,t}(x) that transports samples of an underlying
probability flow ODE from time s to time t. Once trained, generation from the
prior rho_0 = N(0, I) to the target rho_1 = two-moons is a single call X_{0,1}(x_0).
Multi-step generation also works: x_{t_k} = X_{t_{k-1}, t_k}(x_{t_{k-1}}).

Stochastic interpolant (eq. 3.1, with alpha_t = 1-t, beta_t = t, gamma_t = 0):
    I_t = (1 - t) x_0 + t x_1,        x_0 ~ N(0, I),  x_1 ~ rho_1
    dI_t/dt = x_1 - x_0

Network parameterization (eq. 4.1):
    X_{s,t}(x) = x + (t - s) * v_theta(x, s, t)
which automatically enforces the boundary X_{s,s}(x) = x.

Training objective: direct Flow Map Matching loss (Proposition 3.11, eq. 3.17):
    L_FMM = E[ |  d/dt X_{s,t}( X_{t,s}(I_t) ) - dI_t/dt |^2
             + |        X_{s,t}( X_{t,s}(I_t) )  - I_t   |^2 ]
The first term is the Lagrangian / consistency term (it pins the network to the
true probability flow ODE, since E[dI_t | I_t] = b_t(I_t)). The second enforces
two-time invertibility, X_{s,t} o X_{t,s} = id.

We use the "strip" weighting w_{s,t} = 1{|t-s| <= 1/K} from Section 3.7, which
makes direct training much easier; samples are then generated with K Euler-like
jumps along the learned map.

Usage on a Mac:
    conda activate torch
    python fmm_two_moons.py
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons


# --------------------------------------------------------------------------- #
# Device selection (CUDA > MPS > CPU)
# --------------------------------------------------------------------------- #
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def sample_two_moons(n: int, noise: float = 0.05, scale: float = 2.0) -> torch.Tensor:
    """Sample n points from a (rescaled, centered) two-moons distribution."""
    x, _ = make_moons(n_samples=n, noise=noise)
    x = x.astype(np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) * scale  # center + rescale
    return torch.from_numpy(x)


# --------------------------------------------------------------------------- #
# Model: a two-time MLP that returns a velocity-like quantity v_theta(x, s, t).
# The flow map is X_{s,t}(x) = x + (t - s) * v_theta(x, s, t).
# --------------------------------------------------------------------------- #
class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal time embedding (a la transformers / diffusion models)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        # Pre-compute log-spaced frequencies; register as buffer so they move with .to(device).
        half = dim // 2
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = t[:, None] * self.freqs[None, :] * 2 * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowMapMLP(nn.Module):
    """
    MLP that consumes (x, s, t) and outputs a vector v in R^d.
    The flow map is X_{s,t}(x) = x + (t - s) * v.
    """

    def __init__(self, data_dim: int = 2, hidden: int = 256, n_layers: int = 4,
                 t_embed_dim: int = 64):
        super().__init__()
        self.data_dim = data_dim
        self.s_embed = SinusoidalEmbedding(t_embed_dim)
        self.t_embed = SinusoidalEmbedding(t_embed_dim)

        in_dim = data_dim + 2 * t_embed_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def velocity(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s_e = self.s_embed(s)
        t_e = self.t_embed(t)
        h = torch.cat([x, s_e, t_e], dim=-1)
        return self.net(h)

    def flow_map(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """X_{s,t}(x) = x + (t - s) * v_theta(x, s, t).  Boundary X_{s,s}(x) = x is exact."""
        v = self.velocity(x, s, t)
        return x + (t - s)[:, None] * v


# --------------------------------------------------------------------------- #
# FMM loss (eq. 3.17 of Boffi et al.)
#
# We use forward-mode AD (torch.func.jvp) to get d/dt X_{s,t}(y) at fixed s, y
# in a single extra forward pass. This is what the paper recommends in Sec. 3.3
# ("can be computed efficiently using forward-mode automatic differentiation").
# It also avoids the per-output-dim grad loop and works cleanly on MPS.
# --------------------------------------------------------------------------- #
def fmm_loss(model: FlowMapMLP, x0: torch.Tensor, x1: torch.Tensor,
             strip_width: float = 0.25, min_gap: float = 0.02) -> torch.Tensor:
    """
    Direct flow-map-matching loss.

    For each example we sample (s, t) with min_gap <= |t - s| <= strip_width
    (a gapped variant of the strip weighting from Sec. 3.7) and form the
    interpolant I_t = (1-t) x0 + t x1 (so dI_t/dt = x1 - x0 in this linear
    case).  We then compute the round-trip

        y = X_{t, s}( I_t )                # push back from t to s
        z = X_{s, t}( y )                  # push forward from s to t  (~= I_t)

    and ask that:
        d/dt z  (at fixed s, y) ==  dI_t/dt = x1 - x0    [consistency / Lagrangian]
        z       ==  I_t                                  [invertibility]

    NOTE on min_gap: with the parameterization X_{s,t}(x) = x + (t-s) v_theta,
    the map collapses to the identity at t == s for ANY v_theta, so the
    consistency target dI_t/dt = x1 - x0 (a random vector independent of y)
    cannot be matched there. Sampling pairs that approach t == s makes the
    loss landscape ill-conditioned and is a common cause of NaNs partway
    through training.
    """
    B = x0.shape[0]
    device = x0.device

    # Sample s ~ U[0, 1].
    s = torch.rand(B, device=device)
    # Sample |delta| ~ U[min_gap, strip_width] with random sign, then clip into [0, 1].
    mag = min_gap + (strip_width - min_gap) * torch.rand(B, device=device)
    sign = torch.where(torch.rand(B, device=device) < 0.5,
                       -torch.ones_like(mag), torch.ones_like(mag))
    t = (s + sign * mag).clamp(0.0, 1.0)

    # Interpolant and its (analytic) time derivative.
    I_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    dI_dt = x1 - x0

    # ---- Push back from t to s.  Gradients flow through y for the invertibility term.
    y = model.flow_map(I_t, t, s)

    # ---- Forward leg: compute z = X_{s,t}(y) AND dz/dt at fixed s, y in one shot.
    # We use torch.func.jvp with a tangent of 1 on t and 0 on s, y.
    def fwd(t_in: torch.Tensor) -> torch.Tensor:
        return model.flow_map(y, s, t_in)

    z, dz_dt = torch.func.jvp(fwd, (t,), (torch.ones_like(t),))

    consistency = ((dz_dt - dI_dt) ** 2).sum(dim=-1).mean()
    invertibility = ((z - I_t) ** 2).sum(dim=-1).mean()
    return consistency + invertibility


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model: FlowMapMLP, n: int, n_steps: int = 1, device: torch.device | None = None
           ) -> torch.Tensor:
    """
    Generate n samples by jumping from t=0 to t=1 in n_steps equal-sized jumps
    of the learned flow map.  n_steps=1 is the one-step (consistency) regime.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = torch.randn(n, model.data_dim, device=device)
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for k in range(n_steps):
        s = times[k].expand(n)
        t = times[k + 1].expand(n)
        x = model.flow_map(x, s, t)
    return x


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(
    n_iters: int = 8_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    strip_width: float = 0.25,
    min_gap: float = 0.02,
    device: torch.device | None = None,
    seed: int = 0,
    log_every: int = 500,
):
    if device is None:
        device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FlowMapMLP(data_dim=2, hidden=256, n_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)

    losses = []
    n_skipped = 0
    t0 = time.time()
    for it in range(1, n_iters + 1):
        # Generate fresh data on CPU and move to device. For the MPS backend, this
        # is faster than letting sklearn run on the device buffer, and it avoids
        # any chance of float64 sneaking in (MPS doesn't support float64).
        x1 = sample_two_moons(batch_size).to(device, non_blocking=True)
        x0 = torch.randn(batch_size, 2, device=device)

        loss = fmm_loss(model, x0, x1, strip_width=strip_width, min_gap=min_gap)

        # NaN / Inf guard: skip this step rather than poisoning the optimizer state.
        if not torch.isfinite(loss):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            if n_skipped <= 5 or n_skipped % 100 == 0:
                print(f"  iter {it:>5d} | non-finite loss, skipping "
                      f"(total skipped: {n_skipped})")
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if it % log_every == 0 or it == 1:
            print(f"  iter {it:>5d} | loss = {loss.item():.4f} | "
                  f"elapsed = {time.time() - t0:.1f}s")

        # Keep MPS memory tidy on smaller Macs.
        if device.type == "mps" and it % 1000 == 0:
            torch.mps.empty_cache()

    if n_skipped:
        print(f"\n  total non-finite steps skipped: {n_skipped} / {n_iters}")
    return model, losses


def _loss_series_for_plot(losses: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Full iteration axis with bad points set to NaN for log-log plots.

    Masks non-finite values, non-positive values (invalid on log scale), and
    extreme upper outliers (rare huge finite losses from unstable steps).
    NaNs break the line so spikes are not interpolated over.
    """
    y = np.asarray(losses, dtype=float)
    it = np.arange(1, len(y) + 1)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() >= 10:
        hi = np.percentile(y[ok], 99.99)
        ok &= y <= hi
    y_plot = y.astype(float)
    y_plot[~ok] = np.nan
    return it, y_plot


# --------------------------------------------------------------------------- #
# Plotting / main
# --------------------------------------------------------------------------- #
def main(out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # Mac-friendly defaults: keep everything in fp32 (MPS doesn't do fp64).
    torch.set_default_dtype(torch.float32)

    device = pick_device()
    print(f"Using device: {device}")

    print("\nTraining flow map matching model ...")
    model, losses = train(n_iters=8_000, batch_size=1024, lr=1e-3,
                          strip_width=0.25, min_gap=0.02, device=device)

    # ---- samples plot: target | prior | N ∈ {1,4,20,50,100} ----
    fig, axes = plt.subplots(1, 7, figsize=(28, 4))

    target = sample_two_moons(4_000).numpy()
    axes[0].scatter(target[:, 0], target[:, 1], s=2, alpha=0.5, c="C0")
    axes[0].set_title("Target: two moons")

    prior = torch.randn(4_000, 2).numpy()
    axes[1].scatter(prior[:, 0], prior[:, 1], s=2, alpha=0.5, c="C7")
    axes[1].set_title(r"Prior $\rho_0 = \mathcal{N}(0, I)$")

    for ax, n_steps in zip(axes[2:], [1, 4, 20, 50, 100]):
        gen = sample(model, 4_000, n_steps=n_steps, device=device).cpu().numpy()
        ax.scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.5, c="C3")
        ax.set_title(f"FMM samples, N = {n_steps} step{'s' if n_steps > 1 else ''}")

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    samples_path = os.path.join(out_dir, "fmm_two_moons_samples.png")
    fig.savefig(samples_path, dpi=130)
    plt.close(fig)
    print(f"Saved samples plot to {samples_path}")

    # ---- loss curve ----
    fig, ax = plt.subplots(figsize=(6, 4))
    iters, loss_plot = _loss_series_for_plot(losses)
    ax.loglog(iters, loss_plot, lw=0.6)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("FMM loss")
    ax.set_title("Training loss")
    fig.tight_layout()
    loss_path = os.path.join(out_dir, "fmm_two_moons_loss.png")
    fig.savefig(loss_path, dpi=130)
    plt.close(fig)
    print(f"Saved loss plot to {loss_path}")

    return samples_path, loss_path


if __name__ == "__main__":
    main()