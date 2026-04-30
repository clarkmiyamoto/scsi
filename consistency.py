"""
Flow Map Matching (FMM) — two-moons (MLP) and MNIST (U-Net).

A minimal implementation of the consistency-model-style framework from
Boffi, Albergo & Vanden-Eijnden, "Flow map matching with stochastic interpolants" (2024).

We learn a TWO-TIME flow map X_{s,t}(x) that transports samples of an underlying
probability flow ODE from time s to time t. Once trained, generation from the
prior rho_0 = N(0, I) to the target is a single call X_{0,1}(x_0).
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

Usage:
    conda activate torch
    python consistency.py                           # two-moons with MLP (default)
    python consistency.py --dataset mnist           # MNIST with U-Net
    python consistency.py --dataset mnist --n_iters 30000 --batch_size 256
"""

import argparse
import math
import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
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


_MNIST_CACHE: torch.Tensor | None = None
_MNIST_MEAN: float = 0.0
_MNIST_STD: float = 1.0


def sample_mnist(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Return n randomly sampled, standardized MNIST images of shape (n, 1, 28, 28).

    The full training set is loaded and cached in memory on first call so
    subsequent calls are fast (just an index into a pre-loaded tensor).
    """
    global _MNIST_CACHE, _MNIST_MEAN, _MNIST_STD
    if _MNIST_CACHE is None:
        ds = torchvision.datasets.MNIST(
            root=os.path.expanduser("~/.cache/mnist"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        imgs, _ = next(iter(loader))          # (60000, 1, 28, 28), values in [0, 1]
        _MNIST_MEAN = imgs.mean().item()
        _MNIST_STD = imgs.std().item()
        _MNIST_CACHE = (imgs - _MNIST_MEAN) / _MNIST_STD
    idx = torch.randint(0, len(_MNIST_CACHE), (n,))
    x = _MNIST_CACHE[idx]
    if device is not None:
        x = x.to(device, non_blocking=True)
    return x


# --------------------------------------------------------------------------- #
# Model: MLP for 2-D toy data
# --------------------------------------------------------------------------- #
class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal time embedding (a la transformers / diffusion models)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
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
        self.sample_shape = (data_dim,)
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
# Model: U-Net for MNIST (28×28 images)
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional encoding + MLP projection for UNet time conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class _ResBlock(nn.Module):
    """Conv ResBlock with FiLM conditioning on a time embedding vector."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = nn.Linear(t_dim, 2 * out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = F.gelu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class _Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class FlowMapUNet(nn.Module):
    """
    U-Net flow map for MNIST 28×28 images.

    Adapts the conditional CleanUNet (from mra_mnist_unet.py) to unconditional
    generation: single input channel (no observation y concatenated), and two-time
    conditioning — both s and t are embedded with SinusoidalTimeEmb and fused via
    a linear projection into one time vector fed to all ResBlocks via FiLM.

    Interface matches FlowMapMLP: .velocity(x, s, t) and .flow_map(x, s, t).

    Architecture:
        Encoder:     28×28 → 14×14 → 7×7
        Bottleneck:  two ResBlocks at 7×7
        Decoder:     7×7  → 14×14 → 28×28  (with skip connections)
    """

    def __init__(self, in_ch: int = 1, c1: int = 64, c2: int = 128, c3: int = 256,
                 time_dim: int = 128):
        super().__init__()
        self.sample_shape = (in_ch, 28, 28)

        self.s_emb = SinusoidalTimeEmb(time_dim)
        self.t_emb = SinusoidalTimeEmb(time_dim)
        self.time_proj = nn.Linear(2 * time_dim, time_dim)

        # Encoder
        self.enc1a = _ResBlock(in_ch, c1, time_dim)
        self.enc1b = _ResBlock(c1, c1, time_dim)
        self.down1 = _Downsample(c1)                     # 28 → 14

        self.enc2a = _ResBlock(c1, c2, time_dim)
        self.enc2b = _ResBlock(c2, c2, time_dim)
        self.down2 = _Downsample(c2)                     # 14 → 7

        # Bottleneck
        self.mid1 = _ResBlock(c2, c3, time_dim)
        self.mid2 = _ResBlock(c3, c3, time_dim)

        # Decoder
        self.up2 = _Upsample(c3)                         # 7 → 14
        self.dec2a = _ResBlock(c3 + c2, c2, time_dim)    # skip from enc2
        self.dec2b = _ResBlock(c2, c2, time_dim)

        self.up1 = _Upsample(c2)                         # 14 → 28
        self.dec1a = _ResBlock(c2 + c1, c1, time_dim)    # skip from enc1
        self.dec1b = _ResBlock(c1, c1, time_dim)

        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, c1), nn.GELU(),
            nn.Conv2d(c1, in_ch, 3, padding=1),
        )

    def _fuse_time(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.time_proj(torch.cat([self.s_emb(s), self.t_emb(t)], dim=-1))

    def velocity(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self._fuse_time(s, t)

        # Encoder
        h1 = self.enc1b(self.enc1a(x, te), te)                               # (B,c1,28,28)
        h2 = self.enc2b(self.enc2a(self.down1(h1), te), te)                  # (B,c2,14,14)

        # Bottleneck
        h = self.mid2(self.mid1(self.down2(h2), te), te)                     # (B,c3,7,7)

        # Decoder with skip connections
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), h2], dim=1), te), te)  # (B,c2,14,14)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), h1], dim=1), te), te)  # (B,c1,28,28)

        return self.out_proj(h)                                               # (B,in_ch,28,28)

    def flow_map(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """X_{s,t}(x) = x + (t - s) * v_θ(x, s, t).  Boundary X_{s,s}(x) = x is exact."""
        v = self.velocity(x, s, t)
        dt = (t - s).view(-1, *([1] * (x.dim() - 1)))   # broadcast over spatial dims
        return x + dt * v


# --------------------------------------------------------------------------- #
# FMM loss (eq. 3.17 of Boffi et al.)
#
# We use forward-mode AD (torch.func.jvp) to get d/dt X_{s,t}(y) at fixed s, y
# in a single extra forward pass. This is what the paper recommends in Sec. 3.3
# ("can be computed efficiently using forward-mode automatic differentiation").
# It also avoids the per-output-dim grad loop and works cleanly on MPS.
#
# Works for any data shape: (B, d) for MLP or (B, C, H, W) for U-Net.
# --------------------------------------------------------------------------- #
def fmm_loss(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor,
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
    # t_v broadcasts over all data dimensions (works for both MLP and U-Net shapes).
    t_v = t.view(B, *([1] * (x0.dim() - 1)))
    I_t = (1.0 - t_v) * x0 + t_v * x1
    dI_dt = x1 - x0

    # ---- Push back from t to s.  Gradients flow through y for the invertibility term.
    y = model.flow_map(I_t, t, s)

    # ---- Forward leg: compute z = X_{s,t}(y) AND dz/dt at fixed s, y in one shot.
    # We use torch.func.jvp with a tangent of 1 on t and 0 on s, y.
    def fwd(t_in: torch.Tensor) -> torch.Tensor:
        return model.flow_map(y, s, t_in)

    z, dz_dt = torch.func.jvp(fwd, (t,), (torch.ones_like(t),))

    # Sum over all non-batch dims, then average over batch.
    consistency = ((dz_dt - dI_dt) ** 2).flatten(1).sum(dim=-1).mean()
    invertibility = ((z - I_t) ** 2).flatten(1).sum(dim=-1).mean()
    return consistency + invertibility


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model: nn.Module, n: int | None = None, n_steps: int = 1,
           device: torch.device | None = None,
           x0: torch.Tensor | None = None) -> torch.Tensor:
    """
    Generate samples by jumping from t=0 to t=1 in n_steps equal-sized jumps.
    n_steps=1 is the one-step (consistency) regime.

    Pass x0 to start from a fixed noise tensor (useful for apples-to-apples
    comparisons across different n_steps values). If x0 is None, n fresh
    samples are drawn from N(0, I).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    if x0 is not None:
        x = x0.to(device)
        n = x.shape[0]
    else:
        x = torch.randn(n, *model.sample_shape, device=device)
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
    model: nn.Module,
    sample_data: Callable[[int], torch.Tensor],
    n_iters: int = 8_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    strip_width: float = 0.25,
    min_gap: float = 0.02,
    device: torch.device | None = None,
    seed: int = 0,
    log_every: int = 500,
    on_log: Callable[[int, float, nn.Module], None] | None = None,
) -> tuple[nn.Module, list[float]]:
    """
    Train *model* to match the distribution returned by *sample_data*.

    sample_data(n) must return a tensor of shape (n, *data_shape) already on
    the correct device.  x0 (the prior) is drawn via torch.randn_like(x1).
    """
    if device is None:
        device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)

    losses = []
    n_skipped = 0
    t0 = time.time()
    for it in range(1, n_iters + 1):
        x1 = sample_data(batch_size)
        x0 = torch.randn_like(x1)

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
            if on_log is not None:
                on_log(it, loss.item(), model)

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
# W&B logging helpers
# --------------------------------------------------------------------------- #
_LOG_STEPS = (1, 2, 4, 8, 16, 32)


def _wandb_figure_two_moons(model: nn.Module, x_fixed: torch.Tensor) -> plt.Figure:
    """6-panel scatter plot — same fixed noise, varying number of flow-map steps."""
    fig, axes = plt.subplots(1, len(_LOG_STEPS), figsize=(4 * len(_LOG_STEPS), 4))
    for ax, n_steps in zip(axes, _LOG_STEPS):
        gen = sample(model, n_steps=n_steps, x0=x_fixed).cpu().numpy()
        ax.scatter(gen[:, 0], gen[:, 1], s=3, alpha=0.4, c="C3")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"N = {n_steps}")
    fig.suptitle("FMM samples (same noise, varying steps)", y=1.01)
    fig.tight_layout()
    return fig


def _wandb_figure_mnist(model: nn.Module, x_fixed: torch.Tensor) -> plt.Figure:
    """Grid with one row per step count — same fixed noise, varying steps."""
    rows = []
    for n_steps in _LOG_STEPS:
        gen = sample(model, n_steps=n_steps, x0=x_fixed).cpu()   # (n, 1, 28, 28)
        gen_vis = (gen * _MNIST_STD + _MNIST_MEAN).clamp(0.0, 1.0)
        rows.append(torchvision.utils.make_grid(gen_vis, nrow=gen_vis.shape[0], padding=2))

    fig, axes = plt.subplots(len(_LOG_STEPS), 1,
                             figsize=(rows[0].shape[2] / 40, len(_LOG_STEPS) * 1.2))
    for ax, row_img, n_steps in zip(axes, rows, _LOG_STEPS):
        ax.imshow(row_img.permute(1, 2, 0).squeeze(-1).numpy(), cmap="gray")
        ax.set_ylabel(f"N={n_steps}", rotation=0, labelpad=28, va="center")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("FMM samples (same noise, varying steps)")
    fig.tight_layout()
    return fig


def _make_on_log(
    model: nn.Module,
    device: torch.device,
    dataset: str,
    run: "wandb.sdk.wandb_run.Run",
    n_fixed: int = 512,
) -> Callable[[int, float, nn.Module], None]:
    """Return a callback for train(on_log=...) that logs loss + sample figure to W&B."""
    torch.manual_seed(42)
    x_fixed = torch.randn(n_fixed, *model.sample_shape)  # fixed noise, CPU

    make_fig = _wandb_figure_two_moons if dataset == "two_moons" else _wandb_figure_mnist

    def on_log(it: int, loss: float, model: nn.Module) -> None:
        model.eval()
        fig = make_fig(model, x_fixed)
        run.log({"train/loss": loss, "samples": wandb.Image(fig)}, step=it)
        plt.close(fig)
        model.train()

    return on_log


# --------------------------------------------------------------------------- #
# Per-dataset mains
# --------------------------------------------------------------------------- #
def main_two_moons(out_dir: str, n_iters: int, batch_size: int, lr: float,
                   log_every: int = 500, wandb_project: str | None = None,
                   wandb_entity: str | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.set_default_dtype(torch.float32)

    device = pick_device()
    print(f"Using device: {device}")

    model = FlowMapMLP(data_dim=2, hidden=256, n_layers=4).to(device)
    sample_data = lambda n: sample_two_moons(n).to(device, non_blocking=True)

    on_log = None
    if wandb_project:
        run = wandb.init(project=wandb_project, entity=wandb_entity,
                         config=dict(dataset="two_moons", n_iters=n_iters,
                                     batch_size=batch_size, lr=lr, log_every=log_every))
        on_log = _make_on_log(model, device, "two_moons", run)

    print("\nTraining flow map matching model (two-moons, MLP) ...")
    model, losses = train(
        model, sample_data,
        n_iters=n_iters, batch_size=batch_size, lr=lr,
        strip_width=0.25, min_gap=0.02, device=device,
        log_every=log_every, on_log=on_log,
    )

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
    ax.set_title("Training loss (two-moons)")
    fig.tight_layout()
    loss_path = os.path.join(out_dir, "fmm_two_moons_loss.png")
    fig.savefig(loss_path, dpi=130)
    plt.close(fig)
    print(f"Saved loss plot to {loss_path}")

    if wandb_project:
        wandb.finish()


def main_mnist(out_dir: str, n_iters: int, batch_size: int, lr: float,
               log_every: int = 500, wandb_project: str | None = None,
               wandb_entity: str | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.set_default_dtype(torch.float32)

    device = pick_device()
    print(f"Using device: {device}")

    # Pre-load MNIST cache before training starts so timing is clean.
    print("Loading MNIST ...")
    sample_mnist(1, device=device)

    model = FlowMapUNet(in_ch=1, c1=64, c2=128, c3=256, time_dim=128).to(device)
    sample_data = lambda n: sample_mnist(n, device=device)

    on_log = None
    if wandb_project:
        run = wandb.init(project=wandb_project, entity=wandb_entity,
                         config=dict(dataset="mnist", n_iters=n_iters,
                                     batch_size=batch_size, lr=lr, log_every=log_every))
        on_log = _make_on_log(model, device, "mnist", run, n_fixed=8)

    print(f"\nTraining flow map matching model (MNIST, U-Net) ...")
    model, losses = train(
        model, sample_data,
        n_iters=n_iters, batch_size=batch_size, lr=lr,
        strip_width=0.25, min_gap=0.02, device=device,
        log_every=log_every, on_log=on_log,
    )

    # ---- samples grid ----
    n_gen = 64
    gen = sample(model, n_gen, n_steps=1, device=device).cpu()  # (64, 1, 28, 28)
    # Undo standardization and clamp to [0, 1] for display.
    gen_vis = (gen * _MNIST_STD + _MNIST_MEAN).clamp(0.0, 1.0)
    grid = torchvision.utils.make_grid(gen_vis, nrow=8, padding=2)  # (3, H, W) or (1,H,W)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).squeeze(-1).numpy(), cmap="gray")
    ax.axis("off")
    ax.set_title(f"FMM generated MNIST digits (1-step, after {n_iters} iters)")
    fig.tight_layout()
    samples_path = os.path.join(out_dir, "fmm_mnist_samples.png")
    fig.savefig(samples_path, dpi=130)
    plt.close(fig)
    print(f"Saved samples grid to {samples_path}")

    # ---- loss curve ----
    fig, ax = plt.subplots(figsize=(6, 4))
    iters, loss_plot = _loss_series_for_plot(losses)
    ax.loglog(iters, loss_plot, lw=0.6)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("FMM loss")
    ax.set_title("Training loss (MNIST)")
    fig.tight_layout()
    loss_path = os.path.join(out_dir, "fmm_mnist_loss.png")
    fig.savefig(loss_path, dpi=130)
    plt.close(fig)
    print(f"Saved loss plot to {loss_path}")

    if wandb_project:
        wandb.finish()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flow Map Matching: two-moons (MLP) or MNIST (U-Net)"
    )
    parser.add_argument(
        "--dataset", choices=["two_moons", "mnist"], default="two_moons",
        help="Dataset and model architecture to use (default: two_moons)",
    )
    parser.add_argument(
        "--n_iters", type=int, default=None,
        help="Training iterations (default: 8000 for two_moons, 30000 for mnist)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (default: 1024 for two_moons, 256 for mnist)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--log_every", type=int, default=25,
        help="Log + upload W&B samples every this many iterations (default: 500)",
    )
    parser.add_argument("--out_dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument(
        "--wandb_project", default=None,
        help="W&B project name. If omitted, W&B logging is disabled.",
    )
    parser.add_argument("--wandb_entity", default=None, help="W&B entity (team/user)")
    args = parser.parse_args()

    shared = dict(
        out_dir=args.out_dir, lr=args.lr,
        log_every=args.log_every,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
    )

    if args.dataset == "two_moons":
        main_two_moons(n_iters=args.n_iters or 8_000, batch_size=args.batch_size or 1024,
                       **shared)
    else:
        main_mnist(n_iters=args.n_iters or 30_000, batch_size=args.batch_size or 256,
                   **shared)


if __name__ == "__main__":
    main()
