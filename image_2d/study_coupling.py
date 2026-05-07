"""
Coupling fraction ablation study.

For each coupled_fraction in FRACTIONS, runs a short EM loop and tracks:
  - Reconstruction MSE against ground truth at each M-step
  - Pixel std of the prior pool (should approach GT std)

Saves:
  results/coupling_metrics.png    — MSE + std curves
  results/coupling_images.png     — prior grids at selected EM steps
  results/coupling_histograms.png — pixel intensity histograms, same grid layout
"""
import os
os.makedirs("results", exist_ok=True)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_mnist
from corruption import forward_channel
from model import ConditionalDiT
from em import train_estep, update_prior

# ── Study settings ─────────────────────────────────────────────────────────
FRACTIONS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_OBS        = 1000
N_EM         = 10
EPOCHS_FIRST = 3
EPOCHS_PER   = 2
BATCH_SIZE   = 64
LR           = 3e-4
ODE_STEPS    = 20
CORRUPTION   = "awgn"
NOISE_STD    = 0.3
P_DROP       = 0.1
STYLE        = "linear"
N_IMG        = 8           # images per row in grid plot
SNAP_STEPS   = [0, 5, 10]  # which prior snapshots to show (0 = before any EM)

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"Device: {device}\n")

# ── Data (shared across all runs) ──────────────────────────────────────────
torch.manual_seed(0)
x_gt  = load_mnist(N_OBS)
y_obs = forward_channel(x_gt, noise_std=NOISE_STD, p_drop=P_DROP, corruption=CORRUPTION)

gt_std  = float(x_gt.std())
obs_mse = float(((y_obs - x_gt) ** 2).mean())
print(f"GT std:  {gt_std:.4f}")
print(f"Obs MSE vs GT (baseline): {obs_mse:.4f}\n")

# ── Run each fraction ──────────────────────────────────────────────────────
# results[frac]["priors"] stores FULL x_pool (not just N_IMG) for histograms
results = {}

for frac in FRACTIONS:
    print("=" * 55)
    print(f"coupled_fraction = {frac}")
    print("=" * 55)

    torch.manual_seed(42)
    model  = ConditionalDiT().to(device)
    x_pool = y_obs.clone()
    z_pool = None

    mses   = []
    stds   = []
    # Snapshot index 0 = before any EM (y_obs bootstrap)
    prior_imgs  = [x_pool[:N_IMG].clone()]       # for image grid
    prior_pools = [x_pool.clone()]               # full pool for histograms

    for k in range(N_EM):
        print(f"  EM step {k + 1}/{N_EM}")
        epochs = EPOCHS_FIRST if k == 0 else EPOCHS_PER

        train_estep(model, x_pool,
                    noise_std=NOISE_STD, p_drop=P_DROP, corruption=CORRUPTION,
                    style=STYLE, epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
                    global_step=None, device=device,
                    z_pool=z_pool, coupled_fraction=frac)

        x_pool, z_pool = update_prior(model, y_obs,
                                      n_steps=ODE_STEPS,
                                      batch_size=256,
                                      device=device)

        m = float(((x_pool - x_gt) ** 2).mean())
        s = float(x_pool.std())
        mses.append(m)
        stds.append(s)
        prior_imgs.append(x_pool[:N_IMG].clone())
        prior_pools.append(x_pool.clone())
        print(f"    → MSE={m:.4f}  std={s:.4f}")

    results[frac] = {"mse": mses, "std": stds,
                     "prior_imgs": prior_imgs, "prior_pools": prior_pools}
    print()

# ── Shared helpers ─────────────────────────────────────────────────────────
snap_indices = [s for s in SNAP_STEPS if s <= N_EM]
n_snap       = len(snap_indices)
colors       = plt.cm.plasma(np.linspace(0.15, 0.85, len(FRACTIONS)))
steps        = list(range(1, N_EM + 1))

def tile_images(imgs):
    """imgs: (N, 1, H, W) → (H, N*W) numpy array."""
    return torch.cat([imgs[i, 0] for i in range(len(imgs))], dim=1).numpy()

# ── Plot 1: metrics curves ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for color, frac in zip(colors, FRACTIONS):
    ax1.plot(steps, results[frac]["mse"], marker="o", markersize=4,
             label=f"cf={frac}", color=color)
ax1.axhline(obs_mse, color="gray", linestyle="--", label="obs baseline")
ax1.set_xlabel("EM step")
ax1.set_ylabel("MSE vs ground truth")
ax1.set_title("Reconstruction MSE ↓")
ax1.legend(fontsize=7, ncol=3)
ax1.grid(True, alpha=0.3)

for color, frac in zip(colors, FRACTIONS):
    ax2.plot(steps, results[frac]["std"], marker="o", markersize=4,
             label=f"cf={frac}", color=color)
ax2.axhline(gt_std, color="k", linestyle="--", label="GT std")
ax2.axhline(float(y_obs.std()), color="gray", linestyle="--", label="obs std")
ax2.set_xlabel("EM step")
ax2.set_ylabel("Pixel std of prior")
ax2.set_title("Prior std (→ GT std is better)")
ax2.legend(fontsize=7, ncol=3)
ax2.grid(True, alpha=0.3)

fig.suptitle(f"AWGN ablation: coupled_fraction  (n={N_OBS}, {N_EM} EM steps)", fontsize=12)
fig.tight_layout()
fig.savefig("results/coupling_metrics.png", dpi=150)
plt.close(fig)
print("Saved results/coupling_metrics.png")

# ── Plot 2: prior image grids ──────────────────────────────────────────────
# Rows: GT + one per fraction.  Cols: one per snapshot step.
# Tiled images are 28×224 px (8:1 aspect ratio).  Wide figure keeps them from being squished.

vmin = float(x_gt[:N_IMG, 0].min())
vmax = float(x_gt[:N_IMG, 0].max())

n_img_rows = 1 + len(FRACTIONS)   # GT + fractions

fig2, axes2 = plt.subplots(n_img_rows, n_snap,
                           figsize=(10, 4.5), squeeze=False)

for ci, step_idx in enumerate(snap_indices):
    axes2[0, ci].imshow(tile_images(x_gt[:N_IMG]), cmap="gray",
                        vmin=vmin, vmax=vmax, aspect="auto")
    axes2[0, ci].set_xticks([]); axes2[0, ci].set_yticks([])
    axes2[0, ci].set_title("Before EM" if step_idx == 0 else f"After EM {step_idx}",
                            fontsize=10)
axes2[0, 0].set_ylabel("Ground truth", fontsize=8)

for ri, frac in enumerate(FRACTIONS):
    for ci, step_idx in enumerate(snap_indices):
        ax = axes2[ri + 1, ci]
        ax.imshow(tile_images(results[frac]["prior_imgs"][step_idx]),
                  cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
    axes2[ri + 1, 0].set_ylabel(f"cf={frac}", fontsize=8)

fig2.suptitle("Prior samples at selected EM steps", fontsize=11)
fig2.subplots_adjust(hspace=0.15, wspace=0.15)  # generous spacing
fig2.savefig("results/coupling_images.png", dpi=150)
plt.close(fig2)
print("Saved results/coupling_images.png")

# ── Plot 3: pixel intensity histograms ────────────────────────────────────
# One subplot per snapshot step; all fractions overlaid in each, + GT.
gt_pixels = x_gt.numpy().ravel()
HIST_BINS  = 80
HIST_RANGE = (-3.0, 3.0)

fig3, axes3 = plt.subplots(1, n_snap, figsize=(5 * n_snap, 4.5), sharey=False)
if n_snap == 1:
    axes3 = [axes3]

for ci, step_idx in enumerate(snap_indices):
    ax = axes3[ci]
    title = "Before EM" if step_idx == 0 else f"After EM {step_idx}"
    ax.set_title(title, fontsize=11)

    # GT reference (dark, thin)
    ax.hist(gt_pixels, bins=HIST_BINS, range=HIST_RANGE,
            density=True, color="#333333", alpha=0.35,
            edgecolor="none", label="GT")

    for ri, frac in enumerate(FRACTIONS):
        pool_pixels = results[frac]["prior_pools"][step_idx].numpy().ravel()
        ax.hist(pool_pixels, bins=HIST_BINS, range=HIST_RANGE,
                density=True, color=colors[ri], alpha=0.85,
                edgecolor="white", linewidth=0.8, label=f"cf={frac}")

    ax.set_xlabel("pixel value", fontsize=9)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.2)
    if ci == 0:
        ax.legend(fontsize=8, ncol=2)

fig3.suptitle("Pixel intensity histograms at selected EM steps\n"
              "(black = GT, colored = coupled fractions)", fontsize=11)
fig3.tight_layout()
fig3.savefig("results/coupling_histograms.png", dpi=150)
plt.close(fig3)
print("Saved results/coupling_histograms.png")
print("\nDone.")
