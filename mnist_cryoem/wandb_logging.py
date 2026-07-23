import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from si import wrap_to_pi
from corruption import forward_channel, sample_uniform_angle
from ode import sample_joint

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


def log_overfit_step(loss: torch.Tensor, loss_img: torch.Tensor, loss_pose: torch.Tensor,
                     grad_norm: torch.Tensor, step: int, use_wandb: bool) -> None:
    """Per-step scalars from the --overfit single-batch sanity check (overfit.py)."""
    if not use_wandb:
        return
    wandb.log({
        "overfit/loss": loss.item(),
        "overfit/loss_image": loss_img.item(),
        "overfit/loss_pose": loss_pose.item(),
        "overfit/grad_norm": grad_norm.item(),
    }, step=step)


@torch.no_grad()
def log_pretrain_reconstruction(
    model: nn.Module,
    x_gt: torch.Tensor,    # (n, 1, H, W) canonical GT digits to visualize
    noise_std: float,
    sample_steps: int,
    step: int,
    use_wandb: bool,
    n: int = 6,
) -> None:
    """
    Periodic qualitative check for pretrain.py (which otherwise only logs scalars via
    log_train_step). Draws a fresh random rotation for n GT digits, builds the observation
    y = F(x_gt; theta), and asks the CURRENT model to invert it via the E-step integrator
    (ode.sample_joint, z ~ N(0,I) | y) -> (x_hat, theta_hat). Logs a 4-row panel (GT digit /
    Obs y / model x_hat / F(x_hat; theta_hat)) plus mean circular error |wrap(theta_hat -
    theta)| — here a real accuracy metric (pretrain.py has true GT rotations), unlike
    log_em_step's pool-vs-pool circular error which is diagnostic-only.

    sample_joint leaves the model in eval() mode (it doesn't restore it) — this function
    restores model.train() before returning so the caller's training loop is unaffected.
    """
    n = min(n, x_gt.size(0))
    x_gt = x_gt[:n]
    device = x_gt.device

    theta_true = sample_uniform_angle(n, device)
    y_obs, _ = forward_channel(x_gt, noise_std=noise_std, theta=theta_true)

    z_image = torch.randn_like(x_gt)
    x_hat, theta_hat = sample_joint(model, z_image, y_obs, n_steps=sample_steps)
    model.train()

    proj_hat, _ = forward_channel(x_hat, noise_std=0.0, theta=theta_hat)
    circ_err = wrap_to_pi(theta_hat - theta_true).abs().mean().item()

    def strip(y_1d, height=6):
        return y_1d[:, 0, :].unsqueeze(1).expand(-1, height, -1)

    obs_strip = strip(y_obs).cpu()
    proj_strip = strip(proj_hat).cpu()

    vmin_img, vmax_img = -1.0, 1.0
    all_1d = torch.cat([obs_strip.flatten(), proj_strip.flatten()])
    vmin_1d, vmax_1d = float(all_1d.min()), float(all_1d.max())
    if vmax_1d - vmin_1d < 1e-8:
        vmax_1d = vmin_1d + 1e-8

    rows = [
        (x_gt[:, 0].cpu(), "GT digit", vmin_img, vmax_img),
        (obs_strip, "Obs y", vmin_1d, vmax_1d),
        (x_hat[:, 0].cpu(), "Model x_hat", vmin_img, vmax_img),
        (proj_strip, "F(x_hat)", vmin_1d, vmax_1d),
    ]
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
    for r, (data, label, vmin, vmax) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
        for j in range(n):
            axes[r, j].imshow(data[j].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])

    fig.suptitle(f"pretrain step {step}  |  mean circular error={circ_err:.3f} rad",
                fontsize=11)
    plt.tight_layout()

    if use_wandb:
        wandb.log({"pretrain/reconstruction": wandb.Image(fig),
                  "pretrain/circular_error": circ_err}, step=step)
    else:
        from pathlib import Path
        out = Path("mnist_cryoem_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"pretrain_step_{step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def log_em_step(
    x_gt: torch.Tensor,        # (n_images, 1, H, W)
    y_obs: torch.Tensor,       # (N_obs, 1, W)
    theta_star: torch.Tensor,  # (N_obs,)  diagnostic only, never used in training
    image_idx: torch.Tensor,   # (N_obs,)
    x_pool: torch.Tensor,      # (n, 1, H, W)  this iteration's pool (subset of N_obs)
    theta_pool: torch.Tensor,  # (n,)
    pool_indices: torch.Tensor,  # (n,) indices into y_obs/theta_star/image_idx
    em_step: int,
    wandb_step: int,
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

    Logged at `wandb_step` (the same monotonically-increasing SGD-step counter used by
    `log_train_step`), NOT `em_step` — wandb requires non-decreasing steps per run, and
    `em_step` (0, 1, 2, ...) is far behind the counter `log_train_step` has already advanced
    to by the time this fires. Logging at `step=em_step` doesn't error, but wandb silently
    drops the whole history row (image included) since it regresses the step. `em_step` is
    still used for the figure's title/filename, which have no such constraint.
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
                  "em/circular_error": circ_err}, step=wandb_step)
    else:
        from pathlib import Path
        out = Path("mnist_cryoem_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_step_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
