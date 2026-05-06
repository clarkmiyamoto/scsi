"""
Self-Consistent Stochastic Interpolants — MRA channel with U-Net
=======================================================
Based on: https://arxiv.org/abs/2512.10857

Channel:  Y = T(X) + σ·w   (random circular shift + AWGN)

Uses the same standard U-Net architecture as the AWGN variant
(no built-in equivariance/invariance). This tests whether the
EM iteration can make progress on the harder MRA inverse problem
without architectural symmetry constraints.

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
# 2.  U-Net architecture
# ══════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """ResNet block with FiLM time conditioning."""
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
        # FiLM modulation from time
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = F.gelu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CleanUNet(nn.Module):
    """
    Clean U-Net for MNIST 28×28.

    Input:  (x_t, y) concatenated → 2 channels
    Output: velocity prediction → 1 channel

    Architecture:
      Encoder:  28×28 → 14×14 → 7×7
      Decoder:  7×7  → 14×14 → 28×28  (with skip connections)
    """
    def __init__(self, base_ch: int = 64, t_dim: int = 128):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.time_emb = SinusoidalTimeEmb(t_dim)

        # Encoder
        self.enc1a = ResBlock(2, c1, t_dim)
        self.enc1b = ResBlock(c1, c1, t_dim)
        self.down1 = Downsample(c1)                  # 28→14

        self.enc2a = ResBlock(c1, c2, t_dim)
        self.enc2b = ResBlock(c2, c2, t_dim)
        self.down2 = Downsample(c2)                  # 14→7

        # Bottleneck
        self.mid1 = ResBlock(c2, c3, t_dim)
        self.mid2 = ResBlock(c3, c3, t_dim)

        # Decoder
        self.up2 = Upsample(c3)                      # 7→14
        self.dec2a = ResBlock(c3 + c2, c2, t_dim)    # skip from enc2
        self.dec2b = ResBlock(c2, c2, t_dim)

        self.up1 = Upsample(c2)                      # 14→28
        self.dec1a = ResBlock(c2 + c1, c1, t_dim)    # skip from enc1
        self.dec1b = ResBlock(c1, c1, t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, c1), nn.GELU(),
            nn.Conv2d(c1, 1, 3, padding=1),
        )

    def forward(self, x_t, y, t):
        t_emb = self.time_emb(t)
        inp = torch.cat([x_t, y], dim=1)             # [B, 2, 28, 28]

        # Encoder
        h1 = self.enc1b(self.enc1a(inp, t_emb), t_emb)   # [B, c1, 28, 28]
        h2 = self.enc2b(self.enc2a(self.down1(h1), t_emb), t_emb)  # [B, c2, 14, 14]

        # Bottleneck
        h = self.mid2(self.mid1(self.down2(h2), t_emb), t_emb)     # [B, c3, 7, 7]

        # Decoder
        h = self.up2(h)                               # [B, c3, 14, 14]
        h = self.dec2b(self.dec2a(torch.cat([h, h2], dim=1), t_emb), t_emb)

        h = self.up1(h)                               # [B, c2, 28, 28]
        h = self.dec1b(self.dec1a(torch.cat([h, h1], dim=1), t_emb), t_emb)

        return self.out(h)


# ══════════════════════════════════════════════════════════════════════
# 3.  Flow-matching loss & sampling
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
# 4.  E-step: train conditional model
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
# 5.  Robust noise-level estimator
# ══════════════════════════════════════════════════════════════════════
def estimate_noise_mad(x: torch.Tensor) -> float:
    """Estimate noise σ via MAD: σ̂ = median(|x_i|) / 0.6745"""
    return x.abs().median().item() / 0.6745


# ══════════════════════════════════════════════════════════════════════
# 6.  M-step: push real observations through model
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def update_prior(model, y_obs, n_steps=50, batch_size=512):
    model.eval()
    N = y_obs.size(0)
    result_gpu = torch.empty(N, 1, 28, 28, device=DEVICE)
    y_gpu = y_obs.to(DEVICE)

    for start in tqdm(range(0, N, batch_size), desc="  M-step", leave=False):
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
# 7.  Visualisation
# ══════════════════════════════════════════════════════════════════════
def visualise_em(y_obs, x_gt, prior_history, n=8, path="em_mra_unet_results.png"):
    import matplotlib.pyplot as plt

    def to_img(t):
        lo, hi = t.min(), t.max()
        if hi - lo < 1e-8:
            return torch.zeros_like(t)
        return (t - lo) / (hi - lo)

    # Build rows: (label, display_imgs, full_tensor_for_histogram)
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

        # Histogram + MAD noise estimate
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

    fig.suptitle("Self-Consistent Flow Matching — MRA channel (U-Net)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# 8.  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}\n")

    # ── Hyperparameters ───────────────────────────────────────────────
    # Problem
    sigma_noise = 0.5

    # EM
    n_em_steps = 1
    epochs_per_em = 0
    first_pass_epochs = 600
    sample_steps = 50

    # Training
    batch_size = 516
    lr = 3e-4
    n_obs = 2_000 # Number of observations, instead of full dataset

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
    print(f"Epochs (pass 0): {first_pass_epochs}")
    print(f"Epochs / step: {epochs_per_em}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = CleanUNet(base_ch=64, t_dim=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

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
    ckpt_dir = Path("checkpoints_em_mra_unet")
    prior_dir = Path("priors_mra_unet")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    for k in range(n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k}")
        print("=" * 60)

        # Save prior for diagnostics
        torch.save(x_pool, prior_dir / f"prior_em{k:02d}.pt")

        # E-step: fresh model, train on current prior
        model = CleanUNet(base_ch=64, t_dim=128).to(DEVICE)
        current_epochs = first_pass_epochs if k == 0 else epochs_per_em

        train_conditional(
            model, x_pool, sigma=sigma_noise,
            epochs=current_epochs, batch_size=batch_size, lr=lr,
        )

        # Save checkpoint
        p = ckpt_dir / f"drift_em{k:02d}.pt"
        torch.save({
            "model": model.state_dict(), "em_step": k,
            "gt_mean": gt_mean, "gt_std": gt_std,
        }, p)
        print(f"  ✓ saved {p}")

        # M-step
        print(f"  M-step: sampling π({k+1}) ...")
        x_pool = update_prior(model, y_obs, n_steps=sample_steps,
                              batch_size=min(1024, n_obs))
        prior_history.append(x_pool.clone())
        sigma_k = estimate_noise_mad(x_pool)
        print(f"  π({k+1}) mean={x_pool.mean():.4f}  std={x_pool.std():.4f}"
              f"  σ̂(MAD)={sigma_k:.3f}\n")

    # ── Visualise ─────────────────────────────────────────────────────
    visualise_em(y_obs, x_gt_all, prior_history, n=8)


if __name__ == "__main__":
    main()
