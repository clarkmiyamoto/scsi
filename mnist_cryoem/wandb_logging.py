import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from si import wrap_to_pi
from corruption import forward_channel

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def log_train_step(loss_img: torch.Tensor, loss_pose: torch.Tensor,
                   grad_norm: torch.Tensor, global_step: int, use_wandb: bool) -> None:
    """Per-SGD-step scalars from the M-step: image loss, pose loss, grad norm."""
    if not use_wandb:
        return
    wandb.log({
        "train/loss_image": loss_img.item(),
        "train/loss_pose": loss_pose.item(),
        "train/grad_norm": grad_norm.item(),
    }, step=global_step)


def log_em_step(
    x_gt: torch.Tensor,        # (n_images, 1, H, W)
    y_obs: torch.Tensor,       # (N_obs, 1, W)
    theta_star: torch.Tensor,  # (N_obs,)  diagnostic only, never used in training
    image_idx: torch.Tensor,   # (N_obs,)
    x_pool: torch.Tensor,      # (n, 1, H, W)  this iteration's pool (subset of N_obs)
    theta_pool: torch.Tensor,  # (n,)
    pool_indices: torch.Tensor,  # (n,) indices into y_obs/theta_star/image_idx
    em_step: int,
    use_wandb: bool,
    n: int = 6,
):
    """
    4-row panel (n columns):
      Row 0 — GT digit the pool sample's observation came from
      Row 1 — observation y (rendered as a thin strip)
      Row 2 — pool sample x_pool[i] (pi(k) reconstruction)
      Row 3 — F(pool sample) re-projected 1D signal (consistency check, also a strip)
    Plus a logged scalar: mean circular error |wrap(theta_pool - theta_star)| — a health
    diagnostic exploiting the fact that we control data generation, NOT a training signal.
    """
    n = min(n, x_pool.size(0))
    sel = pool_indices[:n]
    gt_imgs = x_gt[image_idx[sel]]
    obs = y_obs[sel]
    pool_imgs = x_pool[:n]
    theta_p = theta_pool[:n]
    theta_s = theta_star[sel]

    with torch.no_grad():
        pool_proj, _ = forward_channel(pool_imgs, noise_std=0.0, theta=theta_p)

    circ_err = wrap_to_pi(theta_pool - theta_star[pool_indices]).abs().mean().item()

    def strip(y_1d, height=6):
        # (n, 1, W) -> (n, height, W) for imshow
        return y_1d[:, 0, :].unsqueeze(1).expand(-1, height, -1)

    obs_strip = strip(obs).cpu()
    proj_strip = strip(pool_proj).cpu()

    vmin_img, vmax_img = -1.0, 1.0
    all_1d = torch.cat([obs_strip.flatten(), proj_strip.flatten()])
    vmin_1d, vmax_1d = float(all_1d.min()), float(all_1d.max())
    if vmax_1d - vmin_1d < 1e-8:
        vmax_1d = vmin_1d + 1e-8

    rows = [
        (gt_imgs[:, 0].cpu(), "GT digit", vmin_img, vmax_img),
        (obs_strip, "Obs y", vmin_1d, vmax_1d),
        (pool_imgs[:, 0].cpu(), f"pi({em_step})", vmin_img, vmax_img),
        (proj_strip, "F(pool)", vmin_1d, vmax_1d),
    ]
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
    for r, (data, label, vmin, vmax) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
        for j in range(n):
            axes[r, j].imshow(data[j].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])

    fig.suptitle(f"EM step {em_step}  |  mean circular error={circ_err:.3f} rad (diagnostic)",
                fontsize=11)
    plt.tight_layout()

    if use_wandb:
        wandb.log({"em/reconstruction": wandb.Image(fig),
                  "em/circular_error": circ_err}, step=em_step)
    else:
        from pathlib import Path
        out = Path("mnist_cryoem_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_step_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
