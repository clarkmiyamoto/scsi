"""
3D counterpart of cryoet_mnist/wandb_logging.py. Volumes have no single natural 2D view, so
each volume is shown two ways: its central depth (Z) slice and its full depth projection
(sum along the projection axis -- what the channel would see at zero tilt). Tilt-series
observations are 2D images already; one representative tilt is shown.
"""

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
