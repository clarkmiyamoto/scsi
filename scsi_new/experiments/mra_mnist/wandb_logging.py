import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corruption import corruption_channel
from ode import euler_integration, euler_integration_trajectory


def _pair_scale(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    lo, hi = min(a.min().item(), b.min().item()), max(a.max().item(), b.max().item())
    return (lo, lo + 1e-8) if hi - lo < 1e-8 else (lo, hi)


@torch.no_grad()
def log_reconstruction_grid(model, x0, y, x_gt, noise_std, n_steps_sampling,
                            em_step, wandb_step, panel_name, device):
    x0, y = x0.to(device), y.to(device)

    x_hat = euler_integration(model, x0, y, n_steps_sampling)
    y_recorrupt = corruption_channel(x_hat, noise_std=noise_std)

    n = x_gt.size(0)
    x_gt, x_hat_c = x_gt.cpu(), x_hat.cpu()
    y_c, y_recorrupt_c = y.cpu(), y_recorrupt.cpu()

    # Visualize
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
    for r, label in enumerate(["GT", "Obs y", "x_hat", "F(x_hat)"]):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for j in range(n):
        img_lo, img_hi = _pair_scale(x_gt[j], x_hat_c[j])
        obs_lo, obs_hi = _pair_scale(y_c[j], y_recorrupt_c[j])
        axes[0, j].imshow(x_gt[j, 0].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[1, j].imshow(y_c[j, 0].numpy(), cmap="gray", vmin=obs_lo, vmax=obs_hi)
        axes[2, j].imshow(x_hat_c[j, 0].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[3, j].imshow(y_recorrupt_c[j, 0].numpy(), cmap="gray", vmin=obs_lo, vmax=obs_hi)
        for r in range(4):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    fig.suptitle(f"{panel_name} reconstruction | EM step {em_step}", fontsize=11)
    plt.tight_layout()

    wandb.log({f"viz/{panel_name}/reconstruction": wandb.Image(fig), "em/step": em_step}, step=wandb_step)
    plt.close(fig)


@torch.no_grad()
def log_trajectory_grid(model, x0, y, n_steps_sampling, n_snapshots, n_rows,
                        em_step, wandb_step, panel_name, device):
    x0, y = x0[:n_rows].to(device), y[:n_rows].to(device)
    traj = euler_integration_trajectory(model, x0, y, n_steps_sampling, n_snapshots)
    traj = traj.cpu()  # (S, n_rows, 1, H, W)
    S, n = traj.size(0), traj.size(1)
    ts = torch.linspace(0, 1, S).tolist()

    fig, axes = plt.subplots(n, S, figsize=(1.6 * S, 1.6 * n), squeeze=False)
    for i in range(n):
        lo, hi = traj[:, i].min().item(), traj[:, i].max().item()
        for s in range(S):
            axes[i, s].imshow(traj[s, i, 0].numpy(), cmap="gray", vmin=lo, vmax=hi)
            axes[i, s].set_xticks([]); axes[i, s].set_yticks([])
            if i == 0:
                axes[i, s].set_title(f"t={ts[s]:.2f}", fontsize=8)
    fig.suptitle(f"{panel_name} trajectory | EM step {em_step}", fontsize=11)
    plt.tight_layout()

    wandb.log({f"viz/{panel_name}/trajectory": wandb.Image(fig), "em/step": em_step}, step=wandb_step)
    plt.close(fig)


def random_draw(pool: dict, config_dataset, n: int) -> dict:
    idx = torch.randperm(pool["x_gt"].size(0))[:n]
    x_gt = pool["x_gt"][idx]
    x0 = torch.randn(n, 1, config_dataset.image_size, config_dataset.image_size)
    y = corruption_channel(x_gt, noise_std=config_dataset.noise_std)
    return {"x_gt": x_gt, "x0": x0, "y": y}
