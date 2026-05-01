"""
Self-Consistent Stochastic Interpolants — MRA channel with Symmetry-Decoupled Drift Net
========================================================================================
Based on: https://arxiv.org/abs/2512.10857

Channel:  Y = T(X) + σ·w   (random circular shift + AWGN)

Implements a drift field b_t(x_t | y) with decoupled symmetries:

  - Invariant in y:   b_t(x_t | T_g y) = b_t(x_t | y)
  - Equivariant in x: b_t(T_g x_t | y) = T_g b_t(x_t | y)

y-stream: power spectrum + autocorrelation (both exactly shift-invariant)
          fed through a plain CNN → global FiLM conditioning vector.

x-stream: all-circular-conv stack (padding_mode='circular') with no
          downsampling. FiLM from the combined (y, t) vector scales/shifts
          each channel uniformly across space, preserving equivariance.

Inputs are NOT concatenated — that would impose the wrong joint group action.

Supports CUDA, Apple Silicon MPS, and CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

# ──────────────────────────────────────────────────────────────────────
# Device  (CUDA > MPS > CPU)
# ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 1.  Forward channel:  Y = T(X) + σ·w   (MRA)
# ══════════════════════════════════════════════════════════════════════
def random_circular_shift(x: torch.Tensor) -> torch.Tensor:
    """Vectorised random 2D circular shift via the FFT shift theorem."""
    B, C, H, W = x.shape
    sh = torch.randint(0, H, (B,), device=x.device)
    sw = torch.randint(0, W, (B,), device=x.device)
    ky = torch.arange(H, device=x.device, dtype=torch.float32)
    kx = torch.arange(W, device=x.device, dtype=torch.float32)
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")
    phase = -2.0 * math.pi * (
        sh[:, None, None].float() * ky[None] / H
        + sw[:, None, None].float() * kx[None] / W
    )
    kernel = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)
    return torch.fft.ifft2(torch.fft.fft2(x) * kernel).real


def forward_channel(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Y = T(X) + σ·w, T ~ uniform random circular shift."""
    return random_circular_shift(x) + sigma * torch.randn_like(x)


# ══════════════════════════════════════════════════════════════════════
# 2.  Translation-invariant y-features
# ══════════════════════════════════════════════════════════════════════
def invariants(y: torch.Tensor) -> torch.Tensor:
    """
    Power spectrum + autocorrelation — both exactly invariant to circular shifts.

    T_g y  →  phase-only change in Fourier domain  →  |FFT|² unchanged.
    Returns (B, 2, H, W).
    """
    Y = torch.fft.fft2(y)
    P = (Y.conj() * Y).real          # power spectrum,   (B, 1, H, W)
    A = torch.fft.ifft2(P).real      # autocorrelation,  (B, 1, H, W)
    return torch.cat([P, A], dim=1)  # (B, 2, H, W)


# ══════════════════════════════════════════════════════════════════════
# 3.  Drift network:  y-invariant / x-equivariant
# ══════════════════════════════════════════════════════════════════════

class DriftNet(nn.Module):
    """
    Conditional drift b_t(x_t | y) with decoupled symmetries.

    y-stream: plain CNN on invariants(y) → global vector h_y.
              No circular padding needed — inputs are already invariant.

    x-stream: 6-block circular-conv stack, no downsampling.
              FiLM from concat(h_y, h_t) applies per-channel scale/bias
              broadcast uniformly over space — commutes with T_g on x.

    Symmetry check (run verify_symmetry after construction):
      y-invariance:    b(x, T_g y) == b(x, y)          [exact]
      x-equivariance:  b(T_g x, y) == T_g b(x, y)      [exact]
    """
    def __init__(self, c: int = 64, d: int = 128):
        super().__init__()

        # y-stream: invariant CNN → global conditioning vector
        self.phi_y = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, d), nn.SiLU(),
        )

        # time embedding: sinusoidal → MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d),
        )

        # x-stream: circular convs only, no downsampling
        self.in_conv  = nn.Conv2d(1, c, 3, padding=1, padding_mode='circular')
        self.blocks   = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=1, padding_mode='circular')
            for _ in range(6)
        ])
        self.norms    = nn.ModuleList([nn.GroupNorm(8, c) for _ in range(6)])
        self.film     = nn.ModuleList([nn.Linear(2 * d, 2 * c) for _ in range(6)])
        self.out_conv = nn.Conv2d(c, 1, 3, padding=1, padding_mode='circular')

    def _sinusoidal(self, t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        freqs = torch.exp(torch.linspace(0, 8, dim // 2, device=t.device))
        a = t[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, x_t: torch.Tensor, y: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        h_y  = self.phi_y(invariants(y))                   # (B, d)
        h_t  = self.t_mlp(self._sinusoidal(t.view(-1)))    # (B, d)
        cond = torch.cat([h_y, h_t], dim=-1)               # (B, 2d)

        h = self.in_conv(x_t)
        for conv, norm, film in zip(self.blocks, self.norms, self.film):
            h = F.silu(norm(conv(h)))
            gamma, beta = film(cond).chunk(2, dim=-1)
            h = gamma[:, :, None, None] * h + beta[:, :, None, None]
        return self.out_conv(h)


# ══════════════════════════════════════════════════════════════════════
# 4.  Symmetry verification
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def verify_symmetry(model: DriftNet, sh: int = 3, sw: int = 7,
                    tol: float = 1e-4) -> None:
    """
    Numerically check both symmetries on random inputs (CPU).

    y-invariance:    ||b(x, T_g y) - b(x, y)||_inf   should be ~1e-5
    x-equivariance:  ||b(T_g x, y) - T_g b(x, y)||_inf  should be ~1e-5
    """
    model_cpu = model.to("cpu").eval()
    x_t   = torch.randn(2, 1, 28, 28)
    y     = torch.randn(2, 1, 28, 28)
    t     = torch.rand(2)
    shift = lambda z: torch.roll(z, (sh, sw), dims=(-2, -1))

    base = model_cpu(x_t, y, t)

    inv_err  = (model_cpu(x_t, shift(y), t) - base).abs().max().item()
    equi_err = (model_cpu(shift(x_t), y, t) - shift(base)).abs().max().item()

    print(f"  y-invariance  error: {inv_err:.2e}"
          f"  ({'PASS' if inv_err  < tol else 'FAIL'})")
    print(f"  x-equivariance error: {equi_err:.2e}"
          f"  ({'PASS' if equi_err < tol else 'FAIL'})")

    model.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════
# 5.  Flow-matching loss & sampling
# ══════════════════════════════════════════════════════════════════════
def flow_matching_loss(model, x, y):
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    w = torch.randn_like(x)
    t4 = t[:, None, None, None]
    x_t = (1.0 - t4) * x + t4 * w
    target = w - x
    pred = model(x_t, y, t)
    return F.mse_loss(pred, target)


@torch.no_grad()
def sample_midpoint(model, y, n_steps=50):
    model.eval()
    B = y.size(0)
    x_t = torch.randn(B, 1, 28, 28, device=y.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = 1.0 - i * dt
        t = torch.full((B,), t_val, device=y.device)
        v1 = model(x_t, y, t)
        x_mid = x_t - v1 * (dt / 2.0)
        t_mid = torch.full((B,), t_val - dt / 2.0, device=y.device)
        v2 = model(x_mid, y, t_mid)
        x_t = x_t - v2 * dt
    return x_t


# ══════════════════════════════════════════════════════════════════════
# 6.  E-step: train conditional model
# ══════════════════════════════════════════════════════════════════════
def train_conditional(model, x_pool, sigma, epochs=10, batch_size=256,
                      lr=3e-4):
    loader = DataLoader(
        TensorDataset(x_pool),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for (x_batch,) in tqdm(loader, desc=f"  E-step epoch {epoch}/{epochs}",
                                leave=False):
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = forward_channel(x_batch, sigma=sigma)

            loss = flow_matching_loss(model, x_batch, y_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        sched.step()
        avg = running / len(loader)
        print(f"    epoch {epoch:2d}  |  loss = {avg:.5f}")

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════
# 7.  Robust noise-level estimator
# ══════════════════════════════════════════════════════════════════════
def estimate_noise_mad(x: torch.Tensor) -> float:
    """Estimate noise σ via MAD: σ̂ = median(|x_i|) / 0.6745"""
    return x.abs().median().item() / 0.6745


# ══════════════════════════════════════════════════════════════════════
# 8.  M-step: push real observations through model
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def update_prior(model, y_obs, n_steps=50, batch_size=512):
    model.eval()
    N = y_obs.size(0)
    result_gpu = torch.empty(N, 1, 28, 28, device=DEVICE)
    y_gpu = y_obs.to(DEVICE)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = sample_midpoint(model, y_gpu[start:end], n_steps=n_steps)
        result_gpu[start:end] = x_batch

    result = result_gpu.cpu()
    del result_gpu, y_gpu

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════
# 9.  Visualisation
# ══════════════════════════════════════════════════════════════════════
def visualise_em(y_obs, x_gt, prior_history, n=8,
                 path="em_mra_unet_equivariant_drift_results.png"):
    import matplotlib.pyplot as plt

    def to_img(t):
        lo, hi = t.min(), t.max()
        if hi - lo < 1e-8:
            return torch.zeros_like(t)
        return (t - lo) / (hi - lo)

    rows = [
        ("π (GT)", x_gt[:n], x_gt),
        ("μ (obs)", y_obs[:n], y_obs),
    ]
    for k, x_pool in enumerate(prior_history):
        rows.append((f"π({k})", x_pool[:n], x_pool))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n + 1, figsize=(2 * (n + 1), 2.2 * n_rows),
                             gridspec_kw={"width_ratios": [1] * n + [1.5]})

    for row_idx, (label, imgs, full_tensor) in enumerate(rows):
        for j in range(n):
            axes[row_idx, j].imshow(
                to_img(imgs[j, 0]), cmap="gray", vmin=0, vmax=1,
            )
            axes[row_idx, j].axis("off")

        ax_hist = axes[row_idx, n]
        vals = full_tensor.flatten().numpy()
        ax_hist.hist(vals, bins=100, density=True, color="steelblue",
                     alpha=0.7, edgecolor="none")
        sigma_hat = estimate_noise_mad(full_tensor)
        ax_hist.set_title(f"σ̂ = {sigma_hat:.3f}", fontsize=9)
        ax_hist.set_xlim(-4, 4)
        ax_hist.tick_params(labelsize=7)
        ax_hist.set_yticks([])

        axes[row_idx, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(
        "Self-Consistent Flow Matching — MRA channel (DriftNet: y-invariant / x-equivariant)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# 10.  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}\n")

    # ── Hyperparameters ───────────────────────────────────────────────
    sigma_noise = 0.5
    n_em_steps = 20
    epochs_per_em = 100
    sample_steps = 50
    batch_size = 516
    lr = 3e-4
    n_obs = 516

    # ── Generate fixed observations ──────────────────────────────────
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform,
    )
    full_loader = DataLoader(dataset, batch_size=n_obs, shuffle=True)
    x_gt_all, _ = next(iter(full_loader))

    # Standardise GT to mean=0, std=1
    gt_mean = x_gt_all.mean()
    gt_std = x_gt_all.std().clamp(min=1e-6)
    x_gt_all = (x_gt_all - gt_mean) / gt_std
    print(f"GT normalised: mean={x_gt_all.mean():.4f}, std={x_gt_all.std():.4f}")

    y_obs = forward_channel(x_gt_all, sigma=sigma_noise)

    print(f"Observations:  {y_obs.shape[0]}")
    print(f"Channel:       MRA (shift + AWGN)  σ={sigma_noise}")
    print(f"EM steps:      {n_em_steps}")
    print(f"Epochs / step: {epochs_per_em}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = DriftNet(c=64, d=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    # ── Verify symmetries ─────────────────────────────────────────────
    print("Symmetry check:")
    verify_symmetry(model)
    print()

    # ── Bootstrap: pi^(0) = mu (observations as initial X-pool) ─────
    x_pool = y_obs.clone()
    prior_history = [x_pool.clone()]

    # ── Reference noise levels ────────────────────────────────────────
    sigma_gt = estimate_noise_mad(x_gt_all)
    sigma_obs = estimate_noise_mad(y_obs)
    print(f"Noise estimates (MAD):")
    print(f"  π (GT):    σ̂ = {sigma_gt:.3f}")
    print(f"  μ (obs):   σ̂ = {sigma_obs:.3f}")
    print(f"  π(0):      σ̂ = {estimate_noise_mad(x_pool):.3f}\n")

    # ── EM loop ───────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints_em_mra_unet_equivariant_drift")
    prior_dir = Path("priors_mra_unet_equivariant_drift")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    for k in range(n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k}")
        print("=" * 60)

        torch.save(x_pool, prior_dir / f"prior_em{k:02d}.pt")

        # E-step: fresh model, train on current prior
        model = DriftNet(c=64, d=128).to(DEVICE)

        train_conditional(
            model, x_pool, sigma=sigma_noise,
            epochs=epochs_per_em, batch_size=batch_size, lr=lr,
        )

        p = ckpt_dir / f"drift_em{k:02d}.pt"
        torch.save({
            "model": model.state_dict(), "em_step": k,
            "gt_mean": gt_mean, "gt_std": gt_std,
        }, p)
        print(f"  ✓ saved {p}")

        # M-step
        print(f"  M-step: sampling π({k+1}) ...")
        x_pool = update_prior(model, y_obs, n_steps=sample_steps,
                              batch_size=512)
        prior_history.append(x_pool.clone())
        sigma_k = estimate_noise_mad(x_pool)
        print(f"  π({k+1}) mean={x_pool.mean():.4f}  std={x_pool.std():.4f}"
              f"  σ̂(MAD)={sigma_k:.3f}\n")

    # ── Visualise ─────────────────────────────────────────────────────
    visualise_em(y_obs, x_gt_all, prior_history, n=8)


if __name__ == "__main__":
    main()
