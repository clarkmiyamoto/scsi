import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from si import wrap_to_pi
from corruption import forward_channel, sample_uniform_angle
from ode import sample_joint
from model import IMAGE_SIZE

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
    log_em_pool_diagnostics's pool-vs-pool circular error which is diagnostic-only.

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

    # Each column gets its OWN vmin/vmax (over its Obs y + F(x_hat) pair only), not one shared
    # scale across the whole panel. The projection's huge per-column background DC offset
    # (~-H from summing H rows of -1 background) varies sample to sample and otherwise
    # dominates a shared scale, crushing the much smaller rotation-dependent signal into a
    # near-flat gray band — i.e. every column looks "the same" even though the underlying
    # signals differ.
    rows = [
        (x_gt[:, 0].cpu(), "GT digit", "img"),
        (obs_strip, "Obs y", "1d"),
        (x_hat[:, 0].cpu(), "Model x_hat", "img"),
        (proj_strip, "F(x_hat)", "1d"),
    ]
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
    for r, (_, label, _) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for j in range(n):
        lo = min(obs_strip[j].min().item(), proj_strip[j].min().item())
        hi = max(obs_strip[j].max().item(), proj_strip[j].max().item())
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        for r, (data, _, kind) in enumerate(rows):
            vmin, vmax = (vmin_img, vmax_img) if kind == "img" else (lo, hi)
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


def log_em_pool_diagnostics(
    theta_pool: torch.Tensor,    # (n,)      this iteration's Phi^(k-1)-recovered rotations
    theta_star: torch.Tensor,    # (N_obs,)  diagnostic only, never used in training
    pool_indices: torch.Tensor,  # (n,) indices into theta_star for this iteration's pool
    wandb_step: int,
    use_wandb: bool,
) -> None:
    """
    Scalar-only health diagnostic: mean circular error |wrap(theta_pool - theta_star)|
    between the whole pool's Phi^(k-1)-recovered rotations and the true rotations for the
    SAME sampled observations. Exploits the fact that we control data generation
    (theta_star) — NOT a training signal, never fed back into the loss. Logged at
    `wandb_step` (see log_train_step) for the same non-decreasing-step reason documented on
    log_reconstruction_grid below.
    """
    if not use_wandb:
        return
    circ_err = wrap_to_pi(theta_pool - theta_star[pool_indices]).abs().mean().item()
    wandb.log({"em/circular_error": circ_err}, step=wandb_step)


@torch.no_grad()
def log_reconstruction_grid(
    model: nn.Module,
    x_gt: torch.Tensor,        # (n_images, 1, H, W)
    y_obs: torch.Tensor,       # (N_obs, 1, W)
    theta_star: torch.Tensor,  # (N_obs,)  diagnostic only, never used in training
    image_idx: torch.Tensor,   # (N_obs,)
    acq_idx: torch.Tensor,     # (N_obs,)  which acquisition each observation belongs to
    fixed_acq_id: int,
    n_acq: int,
    sample_steps: int,
    em_step: int,
    wandb_step: int,
    use_wandb: bool,
    device: torch.device,
    n_problems: int = 6,
    max_tilt_rows: int = 32,
) -> None:
    """
    ONE 4-row panel, where each COLUMN is a different "problem" (a whole acquisition / tilt
    series), not a different tilt — the previous per-tilt-column layout couldn't show more
    than one acquisition at a time. Column 0 is always `fixed_acq_id` (tracked across the
    whole run, so you can watch the same particle improve over EM steps); the remaining
    columns are freshly re-randomized acquisitions each call (variety across digits/poses).

      Row 0 — GT digit (single canonical image, one per problem)
      Row 1 — the acquisition's real observations, STACKED into one (T, W) sinogram-style
              image (T = n_tilts, rows = tilt index, columns = projection position) instead
              of spread across sub-columns — this is what actually shows the tilt-series
              structure --n_tilts/--corruptions_per_object/--tilt_increment_deg control.
      Row 2 — ONE generated sample x_hat, conditioned on that acquisition's MIDDLE tilt's
              observation (the model only ever conditions on a single y; there's no joint
              multi-tilt conditioning to fall back on, so one representative tilt is chosen).
      Row 3 — that SAME x_hat re-projected (F, noise_std=0) at every tilt's TRUE angle,
              stacked into another (T, W) sinogram — directly comparable to Row 1. This
              isolates "is the reconstructed image right" from "is the recovered pose
              right" (the latter already has its own diagnostic in log_em_pool_diagnostics)
              by reprojecting at the known true angles rather than a recovered one.
    Column titles show the conditioning tilt's true vs. recovered angle in degrees.

    Runs its own small E-step (ode.sample_joint) directly rather than reusing
    scsi.py::propose_estep: scsi.py already imports from this module (log_train_step), so
    importing scsi.py back here would be circular. sample_joint sets model.eval() internally
    each call but never restores it — model.train() is restored once at the end.

    Logged at `wandb_step`, NOT `em_step` — see log_train_step for why (wandb requires
    non-decreasing steps per run).
    """
    other_ids = torch.randperm(n_acq)
    other_ids = other_ids[other_ids != fixed_acq_id][:n_problems - 1]
    acq_ids = [fixed_acq_id] + other_ids.tolist()
    n = len(acq_ids)

    gt_imgs, obs_sinos, gen_imgs, recorrupt_sinos, col_titles = [], [], [], [], []

    for a in acq_ids:
        mask = acq_idx == a
        y_acq = y_obs[mask]
        theta_star_acq = theta_star[mask]
        img_idx = image_idx[mask][0].item()
        T = y_acq.size(0)

        if T > max_tilt_rows:
            rsel = torch.linspace(0, T - 1, max_tilt_rows).round().long()
            y_acq = y_acq[rsel]
            theta_star_acq = theta_star_acq[rsel]
            T = max_tilt_rows

        cond_i = T // 2
        z_image = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        x_hat, theta_hat = sample_joint(
            model, z_image, y_acq[cond_i:cond_i + 1].to(device), n_steps=sample_steps,
        )

        recorrupt, _ = forward_channel(
            x_hat.expand(T, -1, -1, -1).contiguous(), noise_std=0.0,
            theta=theta_star_acq.to(device),
        )  # (T, 1, W): the SAME single x_hat reprojected at every tilt's true angle

        gt_imgs.append(x_gt[img_idx, 0].cpu())
        obs_sinos.append(y_acq[:, 0, :].cpu())            # (T, W)
        gen_imgs.append(x_hat[0, 0].cpu())
        recorrupt_sinos.append(recorrupt[:, 0, :].cpu())  # (T, W)

        true_deg = theta_star_acq[cond_i].item() * 180.0 / torch.pi
        rec_deg = theta_hat[0].item() * 180.0 / torch.pi
        tag = " (fixed)" if a == fixed_acq_id else ""
        col_titles.append(f"acq={a}{tag}\ncond θ: true {true_deg:.0f}° rec {rec_deg:.0f}°")

    model.train()

    fig, axes = plt.subplots(4, n, figsize=(2.4 * n, 9), squeeze=False)
    row_labels = ["GT digit", f"Obs y ({T} tilts)", f"pi({em_step}) sample",
                  "Recorrupt (true θ)"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9)

    # Each column gets its own vmin/vmax, shared within a row PAIR — (Obs y, Recorrupt) get
    # one scale, (GT digit, pi(k) sample) get another — rather than one scale across the
    # whole panel, so an out-of-range/undertrained pi(k) sample doesn't wash out under a
    # scale fixed to GT's [-1,1], and so the two members of each pair stay directly
    # comparable to each other (see log_pretrain_reconstruction for the shared-scale pitfall
    # this avoids).
    for j in range(n):
        sino_lo = min(obs_sinos[j].min().item(), recorrupt_sinos[j].min().item())
        sino_hi = max(obs_sinos[j].max().item(), recorrupt_sinos[j].max().item())
        if sino_hi - sino_lo < 1e-8:
            sino_hi = sino_lo + 1e-8

        img_lo = min(gt_imgs[j].min().item(), gen_imgs[j].min().item())
        img_hi = max(gt_imgs[j].max().item(), gen_imgs[j].max().item())
        if img_hi - img_lo < 1e-8:
            img_hi = img_lo + 1e-8

        axes[0, j].imshow(gt_imgs[j].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[1, j].imshow(obs_sinos[j].numpy(), cmap="gray", vmin=sino_lo, vmax=sino_hi,
                          aspect="auto")
        axes[2, j].imshow(gen_imgs[j].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[3, j].imshow(recorrupt_sinos[j].numpy(), cmap="gray", vmin=sino_lo, vmax=sino_hi,
                          aspect="auto")
        axes[0, j].set_title(col_titles[j], fontsize=7)
        for r in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])

    fig.suptitle(f"em/reconstruction  |  EM step {em_step}", fontsize=11)
    plt.tight_layout()

    if use_wandb:
        wandb.log({"em/reconstruction": wandb.Image(fig)}, step=wandb_step)
    else:
        from pathlib import Path
        out = Path("mnist_cryoem_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_reconstruction_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
