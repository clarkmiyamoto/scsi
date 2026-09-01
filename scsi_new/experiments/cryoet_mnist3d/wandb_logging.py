"""
3D counterpart of cryoet_mnist/wandb_logging.py. Volumes have no single natural 2D view, so
each volume is shown two ways: its central depth (Z) slice and its full depth projection
(sum along the projection axis -- what the channel would see at zero tilt). Tilt-series
observations are 2D images already; one representative tilt is shown.

log_reconstruction_grid also emits an interactive 3D point-cloud twin of its volumes at
viz/{panel}/reconstruction_pc -- GT and x_hat for every display example in one rotatable
wandb.Object3D scene. See log_reconstruction_pointcloud.
"""

import math

import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corruption import corruption_channel
from rotation import sample_tilt_series_rotations_so3
from ode import euler_integration, euler_integration_trajectory


def _pair_scale(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    lo, hi = min(a.min().item(), b.min().item()), max(a.max().item(), b.max().item())
    return (lo, lo + 1e-8) if hi - lo < 1e-8 else (lo, hi)


def _depth_proj(vol: torch.Tensor) -> torch.Tensor:
    # (..., D, H, W) -> (..., H, W): integrate along the projection (depth) axis.
    return vol.sum(dim=-3)


def _central_slice(vol: torch.Tensor) -> torch.Tensor:
    # (..., D, H, W) -> (..., H, W): the middle depth slice.
    return vol[..., vol.size(-3) // 2, :, :]


# Point-cloud colours match data.py's marching-cubes isosurface convention.
_GT_RGB = (76, 114, 176)     # "#4c72b0"
_HAT_RGB = (196, 78, 82)     # "#c44e52"


def _volume_to_points(vol: torch.Tensor, frac: float) -> torch.Tensor:
    """(D, H, W) volume -> (P, 6) [x=W, y=H, z=D, r, g, b] for the `frac`-brightest
    voxels, recentred on the volume midpoint (colour left as zeros for the caller).

    An exact value *budget* -- P = round(frac * D*H*W) voxels via topk, identical for
    every cloud -- rather than a threshold. `vol > c` breaks both ways here: the ODE's
    x_hat is unclamped and need not sit near [-1, 1] (a fixed c can select ~0 or ~all of
    it), and the near-binary GT collapses to an empty set whenever its ink fraction
    exceeds `frac` (every ink voxel ties at the max). topk is bounded away from 0 and
    D*H*W by construction and uses no RNG. Equal budget for GT and x_hat makes the panel
    a comparison of geometry, not intensity (the image panel already carries intensity).
    The midpoint offset is a fixed affine shift applied identically to every volume; it
    is deliberately NOT a per-cloud centroid, which would slide a displaced x_hat back
    onto GT and hide the error.
    """
    v = vol.detach().float().cpu()
    k = min(v.numel(), max(1, round(frac * v.numel())))
    idx = torch.topk(v.flatten(), k, sorted=False).indices
    occ = torch.stack(torch.unravel_index(idx, v.shape), dim=1).float()   # (P, 3) (d, h, w)
    center = (torch.tensor(v.shape, dtype=torch.float32) - 1.0) / 2.0
    d, h, w = occ[:, 0] - center[0], occ[:, 1] - center[1], occ[:, 2] - center[2]
    xyz = torch.stack([w, h, d], dim=1)
    return torch.cat([xyz, torch.zeros_like(xyz)], dim=1)


@torch.no_grad()
def log_reconstruction_pointcloud(x_gt, x_hat, em_step, wandb_step, panel_name, frac=0.05):
    """Interactive 3D point-cloud twin of viz/{panel_name}/reconstruction.

    One rotatable wandb.Object3D scene per call: for each display example, the GT volume
    (blue) and the model's x_hat (red) as two point clouds. Examples tile a near-square
    grid on the x/y plane (a single row would be an unviewable strip at n_display=6) and
    GT/x_hat split along z, so every cell reads GT-then-recon. Reuses the x_hat that
    log_reconstruction_grid already integrated (no second ODE solve). Only the volumes
    become point clouds; y / F(x_hat) are 2D tilts and stay in the image panel.

    Uses no RNG (the budget is topk, not a random subsample) so it cannot perturb the
    training stream.
    """
    n, V = x_gt.size(0), x_gt.size(-1)
    gap = V * 1.6
    n_cols = max(1, math.ceil(math.sqrt(n)))
    clouds = []
    for j in range(n):
        pairs = ((x_gt[j, 0], _GT_RGB), (x_hat[j, 0], _HAT_RGB))
        for row, (vol, rgb) in enumerate(pairs):
            pc = _volume_to_points(vol, frac)
            if pc.size(0) == 0:
                continue
            pc[:, 0] += (j % n_cols) * gap     # examples across x ...
            pc[:, 1] -= (j // n_cols) * gap    # ... and down y
            pc[:, 2] += row * gap             # GT at z~0, x_hat offset +z
            pc[:, 3:] = torch.tensor(rgb, dtype=torch.float32)
            clouds.append(pc)
    if not clouds:
        return
    arr = torch.cat(clouds, dim=0).numpy()
    caption = (f"{panel_name} | EM step {em_step} | each cell: GT (blue) then x_hat "
               f"(red, +z) | top {frac:.0%} of voxels")
    wandb.log({f"viz/{panel_name}/reconstruction_pc": wandb.Object3D(arr, caption=caption),
               "em/step": em_step},
              step=wandb_step)


@torch.no_grad()
def log_reconstruction_grid(model, x0, y, x_gt, config_dataset, n_steps_sampling,
                            em_step, wandb_step, panel_name, device):
    x0, y = x0.to(device), y.to(device)

    x_hat = euler_integration(model, x0, y, n_steps_sampling)
    y_recorrupt = corruption_channel(x_hat,
                                     num_tilts=config_dataset.num_tilts,
                                     tilt_increment_deg=config_dataset.tilt_increment_deg,
                                     noise_std=config_dataset.noise_std,
                                     tilt_axis=config_dataset.tilt_axis)

    n = x_gt.size(0)
    x_gt, x_hat = x_gt.cpu(), x_hat.cpu()
    y, y_recorrupt = y.cpu(), y_recorrupt.cpu()
    t_show = config_dataset.num_tilts // 2

    row_labels = ["GT z-slice", "GT depth-proj", "x_hat z-slice", "x_hat depth-proj",
                  f"y  tilt {t_show}", f"F(x_hat) tilt {t_show}"]
    fig, axes = plt.subplots(len(row_labels), n, figsize=(2 * n, 2 * len(row_labels)),
                             squeeze=False)
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9)

    for j in range(n):
        gt_slice, hat_slice = _central_slice(x_gt[j, 0]), _central_slice(x_hat[j, 0])
        gt_proj, hat_proj = _depth_proj(x_gt[j, 0]), _depth_proj(x_hat[j, 0])
        y_tilt, yhat_tilt = y[j, t_show, 0], y_recorrupt[j, t_show, 0]

        proj_lo, proj_hi = _pair_scale(gt_proj, hat_proj)
        tilt_lo, tilt_hi = _pair_scale(y_tilt, yhat_tilt)
        panels = [
            (gt_slice, -1.0, 1.0), (gt_proj, proj_lo, proj_hi),
            (hat_slice, -1.0, 1.0), (hat_proj, proj_lo, proj_hi),
            (y_tilt, tilt_lo, tilt_hi), (yhat_tilt, tilt_lo, tilt_hi),
        ]
        for r, (img, lo, hi) in enumerate(panels):
            axes[r, j].imshow(img.numpy(), cmap="gray", vmin=lo, vmax=hi)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])

    fig.suptitle(f"{panel_name} reconstruction | EM step {em_step}", fontsize=11)
    plt.tight_layout()
    wandb.log({f"viz/{panel_name}/reconstruction": wandb.Image(fig), "em/step": em_step},
              step=wandb_step)
    plt.close(fig)

    # Interactive 3D twin of the volumes in this panel (GT + x_hat), same wandb step.
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
        frames = _depth_proj(traj[:, i, 0])  # (S, H, W) -- depth projection of each state
        lo, hi = frames.min().item(), frames.max().item()
        for s in range(S):
            axes[i, s].imshow(frames[s].numpy(), cmap="gray", vmin=lo, vmax=hi)
            axes[i, s].set_xticks([]); axes[i, s].set_yticks([])
            if i == 0:
                axes[i, s].set_title(f"t={ts[s]:.2f}", fontsize=8)
    fig.suptitle(f"{panel_name} trajectory (depth proj) | EM step {em_step}", fontsize=11)
    plt.tight_layout()
    wandb.log({f"viz/{panel_name}/trajectory": wandb.Image(fig), "em/step": em_step},
              step=wandb_step)
    plt.close(fig)


def random_draw(pool: dict, config_dataset, n: int) -> dict:
    idx = torch.randperm(pool["x_gt"].size(0))[:n]
    x_gt = pool["x_gt"][idx]
    V = config_dataset.vol_size
    x0 = torch.randn(n, 1, V, V, V)
    tilt_increment_rad = config_dataset.tilt_increment_deg * torch.pi / 180.0
    rotations = sample_tilt_series_rotations_so3(n, config_dataset.num_tilts,
                                                tilt_increment_rad,
                                                tilt_axis=config_dataset.tilt_axis)
    y = corruption_channel(x_gt, rotations=rotations, noise_std=config_dataset.noise_std)
    return {"x_gt": x_gt, "x0": x0, "rotations": rotations, "y": y}
