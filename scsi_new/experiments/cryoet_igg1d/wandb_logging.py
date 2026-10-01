"""
IgG-1D counterpart of cryoet_mnist3d/wandb_logging.py. Panels come from the held-out eval pool
(real images with GT, data.build_eval_pool): each column is one image, with its observation y, the
channel's noiseless image of x_hat in the image frame (vol_gain * |CTF| * P(x_hat), on y's colour
scale), and GT vs x_hat as z-projections and central z-slices. GT and x_hat are in different units
(GT density vs model units), so each volume panel has its own 1-99 percentile contrast.

log_reconstruction_grid also emits the interactive point-cloud twin at
viz/{panel}/reconstruction_pc, and log_trajectory_grid shows z-projection snapshots of the ODE.
"""

import math

import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corruption import image_formation
from ode import euler_integration, euler_integration_trajectory

# Per-item entries of an eval pool (data.build_eval_pool); gt_volumes is shared.
ITEM_KEYS = ("x_gt", "y", "x0", "ctf", "label", "rot", "shift")

# Point-cloud colours, as in cryoet_mnist3d.
_GT_RGB = (76, 114, 176)     # "#4c72b0"
_HAT_RGB = (196, 78, 82)     # "#c44e52"


def _limits(*imgs: torch.Tensor) -> tuple[float, float]:
    vals = np.concatenate([i.float().numpy().ravel() for i in imgs])
    lo, hi = np.percentile(vals, [1, 99])
    return float(lo), float(max(hi, lo + 1e-8))


def _volume_to_points(vol: torch.Tensor, k: int) -> torch.Tensor:
    """(D, H, W) volume -> (k, 6) [x=W, y=H, z=D, r, g, b] for its k brightest voxels, recentred
    on the volume midpoint (colour left as zeros). Copied from cryoet_mnist3d: a fixed voxel
    budget via topk, never a threshold, because x_hat's units are not GT's."""
    v = vol.detach().float().cpu()
    k = min(v.numel(), max(1, k))
    idx = torch.topk(v.flatten(), k, sorted=False).indices
    occ = torch.stack(torch.unravel_index(idx, v.shape), dim=1).float()
    center = (torch.tensor(v.shape, dtype=torch.float32) - 1.0) / 2.0
    d, h, w = occ[:, 0] - center[0], occ[:, 1] - center[1], occ[:, 2] - center[2]
    xyz = torch.stack([w, h, d], dim=1)
    return torch.cat([xyz, torch.zeros_like(xyz)], dim=1)


@torch.no_grad()
def log_reconstruction_pointcloud(x_gt, x_hat, em_step, wandb_step, panel_name, iso_frac=0.2):
    """
    One rotatable wandb.Object3D scene: per example, GT (blue) and x_hat (red, offset along z) as
    point clouds, tiled on a near-square grid. Every cloud keeps the same number of voxels: the
    batch-mean count of GT voxels above iso_frac * that volume's max (GT density is continuous, so
    this plays the role of cryoet_mnist3d's near-binary ink count).
    """
    n, V = x_gt.size(0), x_gt.size(-1)
    peak = x_gt.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1)
    k = max(1, round((x_gt > iso_frac * peak).float().flatten(1).sum(1).mean().item()))
    gap = V * 1.6
    n_cols = max(1, math.ceil(math.sqrt(n)))
    clouds = []
    for j in range(n):
        for row, (vol, rgb) in enumerate(((x_gt[j, 0], _GT_RGB), (x_hat[j, 0], _HAT_RGB))):
            pc = _volume_to_points(vol, k)
            pc[:, 0] += (j % n_cols) * gap
            pc[:, 1] -= (j // n_cols) * gap
            pc[:, 2] += row * gap
            pc[:, 3:] = torch.tensor(rgb, dtype=torch.float32)
            clouds.append(pc)
    caption = (f"{panel_name} | EM step {em_step} | each cell: GT (blue) then x_hat (red, +z) | "
               f"top {k} voxels (GT > {iso_frac:g} x max)")
    wandb.log({f"viz/{panel_name}/reconstruction_pc": wandb.Object3D(torch.cat(clouds).numpy(),
                                                                     caption=caption),
               "em/step": em_step}, step=wandb_step)


@torch.no_grad()
def log_reconstruction_grid(model, src: dict, apix: float, vol_gain: float, n_steps_sampling: int,
                            em_step, wandb_step, panel_name, device):
    x_hat = euler_integration(model, src["x0"].to(device), src["y"].to(device), n_steps_sampling)
    reproj = vol_gain * image_formation(x_hat, src["ctf"].to(device), apix)
    x_hat, reproj = x_hat.cpu(), reproj.cpu()
    x_gt, y, label = src["x_gt"], src["y"], src["label"]
    n, D = x_gt.size(0), x_gt.size(-1)

    row_labels = ["y (observed)", "F(x_hat) noiseless", "GT z-proj", "x_hat z-proj",
                  "GT z-slice", "x_hat z-slice"]
    fig, axes = plt.subplots(len(row_labels), n, figsize=(2 * n, 2 * len(row_labels)), squeeze=False)
    for r, text in enumerate(row_labels):
        axes[r, 0].set_ylabel(text, fontsize=9)
    for j in range(n):
        y_lim = _limits(y[j, 0], reproj[j, 0])
        panels = [(y[j, 0], y_lim), (reproj[j, 0], y_lim)]
        for img in (x_gt[j, 0].sum(0), x_hat[j, 0].sum(0), x_gt[j, 0, D // 2], x_hat[j, 0, D // 2]):
            panels.append((img, _limits(img)))
        for r, (img, (lo, hi)) in enumerate(panels):
            axes[r, j].imshow(img.float().numpy(), cmap="gray", vmin=lo, vmax=hi)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, j].set_title(f"conf {int(label[j])}", fontsize=8)

    fig.suptitle(f"{panel_name} reconstruction | EM step {em_step}", fontsize=11)
    plt.tight_layout()
    wandb.log({f"viz/{panel_name}/reconstruction": wandb.Image(fig), "em/step": em_step},
              step=wandb_step)
    plt.close(fig)

    log_reconstruction_pointcloud(x_gt, x_hat, em_step, wandb_step, panel_name)


@torch.no_grad()
def log_trajectory_grid(model, x0, y, n_steps_sampling, n_snapshots, n_rows,
                        em_step, wandb_step, panel_name, device):
    x0, y = x0[:n_rows].to(device), y[:n_rows].to(device)
    traj = euler_integration_trajectory(model, x0, y, n_steps_sampling, n_snapshots)
    traj = traj.cpu()  # (S, n_rows, 1, D, H, W)
    S, n = traj.size(0), traj.size(1)
    ts = torch.linspace(0, 1, S).tolist()

    fig, axes = plt.subplots(n, S, figsize=(1.6 * S, 1.6 * n), squeeze=False)
    for i in range(n):
        frames = traj[:, i, 0].sum(dim=-3)  # (S, H, W) -- z-projection of each state
        lo, hi = frames.min().item(), frames.max().item()
        for s in range(S):
            axes[i, s].imshow(frames[s].numpy(), cmap="gray", vmin=lo, vmax=hi)
            axes[i, s].set_xticks([]); axes[i, s].set_yticks([])
            if i == 0:
                axes[i, s].set_title(f"t={ts[s]:.2f}", fontsize=8)
    fig.suptitle(f"{panel_name} trajectory (z-proj) | EM step {em_step}", fontsize=11)
    plt.tight_layout()
    wandb.log({f"viz/{panel_name}/trajectory": wandb.Image(fig), "em/step": em_step},
              step=wandb_step)
    plt.close(fig)


def select(pool: dict, idx) -> dict:
    """The pool items at idx (per-item keys only)."""
    return {k: pool[k][idx] for k in ITEM_KEYS}


def random_draw(pool: dict, n: int) -> dict:
    """n random pool items. The images are real, so unlike cryoet_mnist3d there is no fresh
    channel draw: this only varies which held-out images the "random" panel shows."""
    return select(pool, torch.randperm(pool["x_gt"].size(0))[:n])
