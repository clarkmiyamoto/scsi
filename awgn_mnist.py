"""
Self-Consistent Flow Matching — AWGN channel variant
=====================================================
Based on: https://arxiv.org/abs/2512.10857

Channel:  Y = X + σ·w   (additive white Gaussian noise, no shift)

Since there is no translation ambiguity, the drift network can be a
standard U-Net that takes (x_t, y) as a 2-channel input, conditioned
on t.  No equivariance/invariance constraints needed.

EM algorithm:
  pi^(0) = mu  (bootstrap with observations)
  For k = 0, 1, ...:
    E-step: train conditional flow X|Y with X ~ pi^(k)
    M-step: push real Y_obs through sampler → pi^(k+1)

Optimised for Apple Silicon MPS.
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
# Device
# ──────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 1.  Forward channel:  Y = X + σ·w
# ══════════════════════════════════════════════════════════════════════
def forward_channel(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


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
        num_workers=0, pin_memory=False, drop_last=True,
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

    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════
# 5.  M-step: push real observations through model
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

    # Standardise to mean=0, std=1
    mu = result.mean()
    std = result.std()
    print(f"    pre-normalisation: mean={mu:.4f}, std={std:.4f}")
    result = (result - mu) / std.clamp(min=1e-6)

    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════
# 6.  Visualisation
# ══════════════════════════════════════════════════════════════════════
def visualise_em(y_obs, x_gt, prior_history, n=8, path="em_awgn_results.png"):
    import matplotlib.pyplot as plt

    def to_img(t):
        lo, hi = t.min(), t.max()
        if hi - lo < 1e-8:
            return torch.zeros_like(t)
        return (t - lo) / (hi - lo)

    n_em = len(prior_history)
    n_rows = 2 + n_em
    fig, axes = plt.subplots(n_rows, n, figsize=(2 * n, 2.2 * n_rows))

    for j in range(n):
        axes[0, j].imshow(to_img(x_gt[j, 0]), cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("GT  X", fontsize=10)

    for j in range(n):
        axes[1, j].imshow(to_img(y_obs[j, 0]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].axis("off")
    axes[1, 0].set_ylabel("Obs  Y", fontsize=10)

    for k, x_pool in enumerate(prior_history):
        for j in range(n):
            axes[2 + k, j].imshow(
                to_img(x_pool[j, 0]), cmap="gray", vmin=0, vmax=1,
            )
            axes[2 + k, j].axis("off")
        axes[2 + k, 0].set_ylabel(f"π({k})", fontsize=10)

    fig.suptitle("Self-Consistent Flow Matching — AWGN channel", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}\n")

    # ── Hyperparameters ───────────────────────────────────────────────
    sigma_noise = 0.8
    n_em_steps = 5
    epochs_per_em = 1
    sample_steps = 50
    batch_size = 256
    lr = 3e-4
    n_obs = 10_000

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
    print(f"Channel:       AWGN  σ={sigma_noise}")
    print(f"EM steps:      {n_em_steps}")
    print(f"Epochs / step: {epochs_per_em}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = CleanUNet(base_ch=64, t_dim=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    # ── Bootstrap: pi^(0) = mu ────────────────────────────────────────
    x_pool = y_obs.clone()
    x_pool = (x_pool - x_pool.mean()) / x_pool.std().clamp(min=1e-6)
    prior_history = [x_pool[:8].clone()]

    # ── EM loop ───────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints_em_awgn")
    prior_dir = Path("priors_awgn")
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

        train_conditional(
            model, x_pool, sigma=sigma_noise,
            epochs=epochs_per_em, batch_size=batch_size, lr=lr,
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
                              batch_size=512)
        prior_history.append(x_pool[:8].clone())
        print(f"  π({k+1}) mean={x_pool.mean():.4f}  std={x_pool.std():.4f}"
              f"  range=[{x_pool.min():.2f}, {x_pool.max():.2f}]\n")

    # ── Visualise ─────────────────────────────────────────────────────
    visualise_em(y_obs, x_gt_all, prior_history, n=8)


if __name__ == "__main__":
    main()
