"""
Self-Consistent Flow Map Matching — AWGN channel variant
=========================================================
Implements the FMM consistency training scheme from lifted_2dtoys_cm.ipynb,
applied to the AWGN inverse problem on MNIST.

Channel:  Y = X + σ·w   (additive white Gaussian noise)

Model: ConditionalFlowMapUNet learns a two-time flow map
       X_{s,t}(x | y)  =  x + (t - s) · v_θ(x, y, s, t)

EM algorithm:
  π^(0) = μ  (bootstrap with observations)
  For k = 0, 1, ...:
    E-step: train conditional flow map X|Y with X ~ π^(k) via FMM loss
    M-step: push real Y_obs through sampler → π^(k+1)

FMM loss (Proposition 3.11, Boffi et al.):
    L = |d/dt X_{s,t}(X_{t,s}(I_t | y) | y) - dI_t/dt|²   (consistency)
      + |X_{s,t}(X_{t,s}(I_t | y) | y) - I_t|²             (invertibility)
where I_t = (1-t)·z' + t·x_pool  is the linear interpolant.
"""

import torch
import torch.nn as nn
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

from consistency import (
    SinusoidalTimeEmb,
    _ResBlock,
    _Downsample,
    _Upsample,
    pick_device,
)

DEVICE = pick_device()


# ══════════════════════════════════════════════════════════════════════
# 1.  Forward channel:  Y = X + σ·w
# ══════════════════════════════════════════════════════════════════════
def forward_channel(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


# ══════════════════════════════════════════════════════════════════════
# 2.  Conditional flow map U-Net
# ══════════════════════════════════════════════════════════════════════
class ConditionalFlowMapUNet(nn.Module):
    """
    U-Net flow map for MNIST 28×28, conditioned on an observation y.

    Input:  x (B,1,28,28) and y (B,1,28,28) concatenated → (B,2,28,28)
    Times:  s, t  (two 1-D tensors of shape (B,))
    Output: velocity (B,1,28,28)

    Interface:
        .velocity(x, y, s, t)  →  v_θ(x, y, s, t)
        .flow_map(x, y, s, t)  →  x + (t - s) · v_θ   [boundary X_{s,s} = id]
    """

    def __init__(self, in_ch: int = 1, c1: int = 64, c2: int = 128,
                 c3: int = 256, time_dim: int = 128):
        super().__init__()
        self.in_ch = in_ch
        self.sample_shape = (in_ch, 28, 28)

        self.s_emb = SinusoidalTimeEmb(time_dim)
        self.t_emb = SinusoidalTimeEmb(time_dim)
        self.time_proj = nn.Linear(2 * time_dim, time_dim)

        # Encoder — first block sees x ‖ y → 2*in_ch input channels
        self.enc1a = _ResBlock(2 * in_ch, c1, time_dim)
        self.enc1b = _ResBlock(c1, c1, time_dim)
        self.down1 = _Downsample(c1)                       # 28 → 14

        self.enc2a = _ResBlock(c1, c2, time_dim)
        self.enc2b = _ResBlock(c2, c2, time_dim)
        self.down2 = _Downsample(c2)                       # 14 → 7

        # Bottleneck
        self.mid1 = _ResBlock(c2, c3, time_dim)
        self.mid2 = _ResBlock(c3, c3, time_dim)

        # Decoder
        self.up2 = _Upsample(c3)                           # 7 → 14
        self.dec2a = _ResBlock(c3 + c2, c2, time_dim)
        self.dec2b = _ResBlock(c2, c2, time_dim)

        self.up1 = _Upsample(c2)                           # 14 → 28
        self.dec1a = _ResBlock(c2 + c1, c1, time_dim)
        self.dec1b = _ResBlock(c1, c1, time_dim)

        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, c1), nn.GELU(),
            nn.Conv2d(c1, in_ch, 3, padding=1),
        )

    def _fuse_time(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.time_proj(torch.cat([self.s_emb(s), self.t_emb(t)], dim=-1))

    def velocity(self, x: torch.Tensor, y: torch.Tensor,
                 s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self._fuse_time(s, t)
        inp = torch.cat([x, y], dim=1)                     # (B, 2, 28, 28)

        h1 = self.enc1b(self.enc1a(inp, te), te)                               # (B,c1,28,28)
        h2 = self.enc2b(self.enc2a(self.down1(h1), te), te)                    # (B,c2,14,14)
        h = self.mid2(self.mid1(self.down2(h2), te), te)                       # (B,c3,7,7)

        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), h2], dim=1), te), te)  # (B,c2,14,14)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), h1], dim=1), te), te)  # (B,c1,28,28)
        return self.out_proj(h)

    def flow_map(self, x: torch.Tensor, y: torch.Tensor,
                 s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """X_{s,t}(x | y) = x + (t - s) · v_θ(x, y, s, t)."""
        v = self.velocity(x, y, s, t)
        dt = (t - s).view(-1, *([1] * (x.dim() - 1)))
        return x + dt * v


# ══════════════════════════════════════════════════════════════════════
# 3.  FMM loss & sampling
# ══════════════════════════════════════════════════════════════════════
def conditional_fmm_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y: torch.Tensor,
    strip_width: float = 0.25,
    min_gap: float = 0.02,
) -> torch.Tensor:
    """
    Conditional FMM loss (eq. 3.17, Boffi et al.) adapted for y-conditioning.

    x0 : fresh noise z' ~ N(0, I)             shape (B, 1, 28, 28)
    x1 : target from current prior x_pool     shape (B, 1, 28, 28)
    y  : conditioning observation              shape (B, 1, 28, 28)

    Samples (s, t) with min_gap ≤ |t - s| ≤ strip_width, forms the linear
    interpolant I_t = (1-t) x0 + t x1, and trains the round-trip
        X_{s,t}(X_{t,s}(I_t | y) | y) ≈ I_t   with matching time derivative.
    """
    B = x0.shape[0]
    device = x0.device

    s = torch.rand(B, device=device)
    mag = min_gap + (strip_width - min_gap) * torch.rand(B, device=device)
    sign = torch.where(torch.rand(B, device=device) < 0.5,
                       -torch.ones_like(mag), torch.ones_like(mag))
    t = (s + sign * mag).clamp(0.0, 1.0)

    t_v = t.view(B, 1, 1, 1)
    I_t = (1.0 - t_v) * x0 + t_v * x1
    dI_dt = x1 - x0

    # Push back t → s; gradients flow through y_back for invertibility term
    y_back = model.flow_map(I_t, y, t, s)

    # Forward s → t, plus time derivative via forward-mode AD (one extra pass)
    def fwd(t_in: torch.Tensor) -> torch.Tensor:
        return model.flow_map(y_back, y, s, t_in)

    z, dz_dt = torch.func.jvp(fwd, (t,), (torch.ones_like(t),))

    consistency = ((dz_dt - dI_dt) ** 2).flatten(1).sum(dim=-1).mean()
    invertibility = ((z - I_t) ** 2).flatten(1).sum(dim=-1).mean()
    return consistency + invertibility


@torch.no_grad()
def conditional_sample(model: nn.Module, y: torch.Tensor,
                       n_steps: int = 5) -> torch.Tensor:
    """
    Generate x̂ by chaining n_steps flow-map applications from z ~ N(0, I).
    n_steps=1 is the one-step (consistency model) regime.
    """
    model.eval()
    B = y.size(0)
    x = torch.randn(B, *model.sample_shape, device=y.device)
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=y.device)
    for k in range(n_steps):
        s = times[k].expand(B)
        t = times[k + 1].expand(B)
        x = model.flow_map(x, y, s, t)
    return x


# ══════════════════════════════════════════════════════════════════════
# 4.  E-step: train conditional flow map
# ══════════════════════════════════════════════════════════════════════
def train_conditional(model, x_pool, sigma, epochs=10, batch_size=256, lr=3e-4,
                      strip_width=0.25, min_gap=0.02):
    loader = DataLoader(
        TensorDataset(x_pool),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_skipped = 0
        for (x_batch,) in tqdm(loader, desc=f"  E-step epoch {epoch}/{epochs}",
                                leave=False):
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = forward_channel(x_batch, sigma=sigma)
            z_prime = torch.randn_like(x_batch)

            loss = conditional_fmm_loss(model, z_prime, x_batch, y_batch,
                                        strip_width=strip_width, min_gap=min_gap)

            if not torch.isfinite(loss):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        sched.step()
        n_batches = len(loader) - n_skipped
        avg = running / max(n_batches, 1)
        skip_str = f"  ({n_skipped} skipped)" if n_skipped else ""
        print(f"    epoch {epoch:2d}  |  loss = {avg:.5f}{skip_str}")

    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════
# 5.  M-step: push real observations through model
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def update_prior(model, y_obs, n_steps=5, batch_size=512):
    model.eval()
    N = y_obs.size(0)
    result_gpu = torch.empty(N, 1, 28, 28, device=DEVICE)
    y_gpu = y_obs.to(DEVICE)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = conditional_sample(model, y_gpu[start:end], n_steps=n_steps)
        result_gpu[start:end] = x_batch

    result = result_gpu.cpu()
    del result_gpu, y_gpu

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
def visualise_em(y_obs, x_gt, prior_history, n=8, path="em_awgn_cm_results.png"):
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

    fig.suptitle("Self-Consistent Flow Map Matching — AWGN channel", fontsize=13)
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
    epochs_per_em = 10
    sample_steps = 5        # flow-map steps
    strip_width = 0.25
    min_gap = 0.02
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

    gt_mean = x_gt_all.mean()
    gt_std = x_gt_all.std().clamp(min=1e-6)
    x_gt_all = (x_gt_all - gt_mean) / gt_std
    print(f"GT normalised: mean={x_gt_all.mean():.4f}, std={x_gt_all.std():.4f}")

    y_obs = forward_channel(x_gt_all, sigma=sigma_noise)

    print(f"Observations:  {y_obs.shape[0]}")
    print(f"Channel:       AWGN  σ={sigma_noise}")
    print(f"EM steps:      {n_em_steps}")
    print(f"Epochs / step: {epochs_per_em}")
    print(f"Sample steps:  {sample_steps}\n")

    # ── Bootstrap: π^(0) = μ ─────────────────────────────────────────
    x_pool = y_obs.clone()
    x_pool = (x_pool - x_pool.mean()) / x_pool.std().clamp(min=1e-6)
    prior_history = [x_pool[:8].clone()]

    # ── EM loop ───────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints_em_awgn_cm")
    prior_dir = Path("priors_awgn_cm")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    for k in range(n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k}")
        print("=" * 60)

        torch.save(x_pool, prior_dir / f"prior_em{k:02d}.pt")

        model = ConditionalFlowMapUNet(in_ch=1, c1=64, c2=128, c3=256,
                                       time_dim=128).to(DEVICE)
        if k == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}\n")

        train_conditional(
            model, x_pool, sigma=sigma_noise,
            epochs=epochs_per_em, batch_size=batch_size, lr=lr,
            strip_width=strip_width, min_gap=min_gap,
        )

        p = ckpt_dir / f"drift_em{k:02d}.pt"
        torch.save({
            "model": model.state_dict(), "em_step": k,
            "gt_mean": gt_mean, "gt_std": gt_std,
        }, p)
        print(f"  ✓ saved {p}")

        print(f"  M-step: sampling π({k+1}) ...")
        x_pool = update_prior(model, y_obs, n_steps=sample_steps, batch_size=512)
        prior_history.append(x_pool[:8].clone())
        print(f"  π({k+1}) mean={x_pool.mean():.4f}  std={x_pool.std():.4f}"
              f"  range=[{x_pool.min():.2f}, {x_pool.max():.2f}]\n")

    # ── Visualise ─────────────────────────────────────────────────────
    visualise_em(y_obs, x_gt_all, prior_history, n=8)


if __name__ == "__main__":
    main()
