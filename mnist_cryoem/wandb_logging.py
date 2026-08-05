import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from si import wrap_to_pi, rotation_geodesic_angle, gram_schmidt_to_matrix, matrix_to_gram_schmidt
from corruption import (
    forward_channel, sample_uniform_angle,
    forward_channel_3d, sample_uniform_rotation_so3, project_2d,
    forward_channel_mra,
)
from ode import sample_joint, sample_joint_3d
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


########################################################
# Rotational-MRA diagnostics. log_em_pool_diagnostics is reused UNMODIFIED (it only ever touches
# theta_pool/theta_star/pool_indices — zero image tensors — so it's already channel-agnostic).
# Only the reconstruction panel needs an MRA-specific version, since there's no tilt series to
# stack into a sinogram here: every row is already a plain (H,W) image.
########################################################

@torch.no_grad()
def log_reconstruction_grid_mra(
    model: nn.Module,
    x_gt: torch.Tensor,        # (n_images, 1, H, W)
    y_obs: torch.Tensor,       # (N_obs, 1, H, W)
    theta_star: torch.Tensor,  # (N_obs,)  diagnostic only, never used in training
    image_idx: torch.Tensor,   # (N_obs,)
    fixed_obs_id: int,
    sample_steps: int,
    em_step: int,
    wandb_step: int,
    use_wandb: bool,
    device: torch.device,
    n_problems: int = 6,
) -> None:
    """
    MRA analogue of log_reconstruction_grid / log_reconstruction_grid_3d. There is no
    tilt-series / acquisition grouping in this channel (see data.build_observations_mra), so
    each COLUMN here is one OBSERVATION, not a group — no sinogram/strip-stacking row is needed
    at all (unlike both other pipelines), since every row is already a plain (H,W) image.
    Column 0 is always `fixed_obs_id` (tracked across the whole run, so you can watch the same
    particle improve over EM steps); the remaining columns are freshly re-randomized observation
    indices each call.

      Row 0 — GT digit the observation came from (single canonical, disk-masked image)
      Row 1 — the observation y itself
      Row 2 — ONE generated sample x_hat, conditioned on that observation's y
      Row 3 — that SAME x_hat re-corrupted (F_MRA, noise_std=0) at the observation's TRUE
              angle — directly comparable to Row 1. Isolates "is the reconstructed image
              right" from "is the recovered pose right" (the latter already has its own
              diagnostic in log_em_pool_diagnostics — note that metric is gauge-ambiguous: the
              learned (image, pose) prior is only identifiable up to a global rotation, so a
              plateau at a nonzero constant isn't necessarily a training failure) by
              recorrupting at the known true angle rather than the recovered one.
    Column titles show the true vs. recovered angle in degrees.

    Runs its own small E-step (ode.sample_joint) directly rather than reusing
    scsi.py::propose_estep — same circular-import reasoning as log_reconstruction_grid
    (scsi.py already imports this module for log_train_step).
    """
    N_obs = y_obs.size(0)
    other_ids = torch.randperm(N_obs)
    other_ids = other_ids[other_ids != fixed_obs_id][:n_problems - 1]
    obs_ids = [fixed_obs_id] + other_ids.tolist()
    n = len(obs_ids)

    gt_imgs, obs_imgs, gen_imgs, recorrupt_imgs, col_titles = [], [], [], [], []

    for o in obs_ids:
        y_o = y_obs[o:o + 1].to(device)
        theta_o = theta_star[o:o + 1].to(device)
        img_idx = image_idx[o].item()

        z_image = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        x_hat, theta_hat = sample_joint(model, z_image, y_o, n_steps=sample_steps)

        recorrupt, _ = forward_channel_mra(x_hat, noise_std=0.0, theta=theta_o)

        gt_imgs.append(x_gt[img_idx, 0].cpu())
        obs_imgs.append(y_obs[o, 0].cpu())
        gen_imgs.append(x_hat[0, 0].cpu())
        recorrupt_imgs.append(recorrupt[0, 0].cpu())

        true_deg = theta_star[o].item() * 180.0 / torch.pi
        rec_deg = theta_hat[0].item() * 180.0 / torch.pi
        tag = " (fixed)" if o == fixed_obs_id else ""
        col_titles.append(f"obs={o}{tag}\ntrue θ {true_deg:.0f}° rec θ {rec_deg:.0f}°")

    model.train()

    fig, axes = plt.subplots(4, n, figsize=(2.4 * n, 9), squeeze=False)
    row_labels = ["GT digit", "Obs y", f"pi({em_step}) sample", "Recorrupt (true θ)"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9)

    # Row-PAIR scaling, same pattern as log_reconstruction_grid/log_reconstruction_grid_3d:
    # (Obs y, Recorrupt) share one scale, (GT digit, pi(k) sample) share another, rather than
    # one scale across the whole panel — keeps an out-of-range/undertrained sample from washing
    # out under GT's fixed [-1,1] scale, and keeps each pair directly comparable to each other.
    for j in range(n):
        img_lo = min(gt_imgs[j].min().item(), gen_imgs[j].min().item())
        img_hi = max(gt_imgs[j].max().item(), gen_imgs[j].max().item())
        if img_hi - img_lo < 1e-8:
            img_hi = img_lo + 1e-8

        obs_lo = min(obs_imgs[j].min().item(), recorrupt_imgs[j].min().item())
        obs_hi = max(obs_imgs[j].max().item(), recorrupt_imgs[j].max().item())
        if obs_hi - obs_lo < 1e-8:
            obs_hi = obs_lo + 1e-8

        axes[0, j].imshow(gt_imgs[j].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[1, j].imshow(obs_imgs[j].numpy(), cmap="gray", vmin=obs_lo, vmax=obs_hi)
        axes[2, j].imshow(gen_imgs[j].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[3, j].imshow(recorrupt_imgs[j].numpy(), cmap="gray", vmin=obs_lo, vmax=obs_hi)
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
        out = Path("mnist_mra_rotation_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_reconstruction_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def log_pretrain_reconstruction_mra(
    model: nn.Module,
    x_gt: torch.Tensor,    # (n, 1, H, W) canonical, disk-masked GT digits to visualize
    noise_std: float,
    sample_steps: int,
    step: int,
    use_wandb: bool,
    n: int = 6,
) -> None:
    """
    MRA analogue of log_pretrain_reconstruction, for pretrain_mra_rotation.py. Draws a fresh
    random rotation for n GT digits, builds the observation y = F_MRA(x_gt; theta) via
    forward_channel_mra, and asks the CURRENT model to invert it via the E-step integrator
    (ode.sample_joint, z ~ N(0,I) | y) -> (x_hat, theta_hat). Logs a 4-row panel (GT digit /
    Obs y / model x_hat / F_MRA(x_hat; theta_hat)) plus mean circular error
    |wrap(theta_hat - theta)| — here a real accuracy metric (pretrain_mra_rotation.py has true
    GT rotations), unlike log_em_pool_diagnostics's pool-vs-pool circular error which is
    diagnostic-only (same distinction log_pretrain_reconstruction draws for the CryoEM
    pipeline). Unlike log_pretrain_reconstruction, every row here is already a plain (H,W)
    image — no 1D-signal "strip" trick needed, since forward_channel_mra never projects.

    Row 3 recorrupts at the RECOVERED angle theta_hat (not the true angle) — unlike
    log_reconstruction_grid_mra's row 3, which recorrupts at the true angle to isolate image
    quality from pose quality using a separate pool-level diagnostic (log_em_pool_diagnostics).
    pretrain_mra_rotation.py has no such separate diagnostic to lean on, so this panel instead
    shows direct self-consistency: does the model's own (x_hat, theta_hat) reproduce something
    resembling the y it was conditioned on.

    sample_joint leaves the model in eval() mode (it doesn't restore it) — this function
    restores model.train() before returning so the caller's training loop is unaffected.
    """
    n = min(n, x_gt.size(0))
    x_gt = x_gt[:n]
    device = x_gt.device

    theta_true = sample_uniform_angle(n, device)
    y_obs, _ = forward_channel_mra(x_gt, noise_std=noise_std, theta=theta_true)

    z_image = torch.randn_like(x_gt)
    x_hat, theta_hat = sample_joint(model, z_image, y_obs, n_steps=sample_steps)
    model.train()

    recorrupt, _ = forward_channel_mra(x_hat, noise_std=0.0, theta=theta_hat)
    circ_err = wrap_to_pi(theta_hat - theta_true).abs().mean().item()

    gt_imgs = x_gt[:, 0].cpu()
    obs_imgs = y_obs[:, 0].cpu()
    gen_imgs = x_hat[:, 0].cpu()
    recorrupt_imgs = recorrupt[:, 0].cpu()

    rows = [
        (gt_imgs, "GT digit"),
        (obs_imgs, "Obs y"),
        (gen_imgs, "Model x_hat"),
        (recorrupt_imgs, "F(x_hat; rec θ)"),
    ]
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
    for r, (_, label) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)

    # Row-PAIR scaling, same pattern as log_reconstruction_grid_mra: (GT digit, Model x_hat)
    # share one scale, (Obs y, Recorrupt) share another, rather than one scale across the whole
    # panel — keeps an out-of-range/undertrained sample from washing out under GT's fixed
    # [-1,1] scale.
    for j in range(n):
        img_lo = min(gt_imgs[j].min().item(), gen_imgs[j].min().item())
        img_hi = max(gt_imgs[j].max().item(), gen_imgs[j].max().item())
        if img_hi - img_lo < 1e-8:
            img_hi = img_lo + 1e-8

        obs_lo = min(obs_imgs[j].min().item(), recorrupt_imgs[j].min().item())
        obs_hi = max(obs_imgs[j].max().item(), recorrupt_imgs[j].max().item())
        if obs_hi - obs_lo < 1e-8:
            obs_hi = obs_lo + 1e-8

        axes[0, j].imshow(gt_imgs[j].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[1, j].imshow(obs_imgs[j].numpy(), cmap="gray", vmin=obs_lo, vmax=obs_hi)
        axes[2, j].imshow(gen_imgs[j].numpy(), cmap="gray", vmin=img_lo, vmax=img_hi)
        axes[3, j].imshow(recorrupt_imgs[j].numpy(), cmap="gray", vmin=obs_lo, vmax=obs_hi)
        for r in range(4):
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
        out = Path("mnist_mra_rotation_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"pretrain_step_{step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


########################################################
# 3D / CryoET diagnostics. Mirror the 2D functions above, with two structural differences:
# (1) rotation_geodesic_angle (on gram_schmidt_to_matrix'd rotations) replaces wrap_to_pi-based
#     circular error throughout; (2) each observation here is a full (H,W) image, not a 1D
#     signal, so the sinogram-stacking trick doesn't apply — replaced with sum-projection
#     mosaics / horizontal strips (see log_reconstruction_grid_3d's docstring).
########################################################

def _volume_to_pointcloud(
    vol: torch.Tensor, keep_frac: float = 0.08, cmap_name: str = "inferno",
    max_points: int = 6000,
) -> np.ndarray:
    """
    (D, H, W) volume -> (N, 6) numpy point cloud [x y z r g b] for wandb.Object3D — wandb's
    native 3D viewer, orbit/zoom/rotate directly in the browser (vs. the flat xy|xz|yz
    projections _projection_mosaic makes). Keeps the top `keep_frac` of THIS volume's OWN
    voxels by value (a per-volume quantile, not a fixed cutoff): x_hat has no guaranteed
    range, so a fixed threshold either empties the cloud or fills the whole cube depending on
    how (mis)calibrated the current model happens to be — same shared-scale pitfall documented
    on log_pretrain_reconstruction, one level more dangerous here since a saturated cloud
    silently degrades into confetti after the max_points subsample below. An
    untrained/undertrained model legitimately renders as a noisy point cloud under this
    scheme — that's the diagnostic, not a bug to threshold away. Voxel value maps to point
    color via cmap_name. Subsamples uniformly past max_points (wandb's browser viewer bogs
    down well before tens of thousands of points).
    """
    vol = vol.detach().float().cpu()
    cutoff = torch.quantile(vol.reshape(-1), 1.0 - keep_frac).item()
    mask = vol > cutoff
    coords = mask.nonzero(as_tuple=False).float()
    values = vol[mask]

    if coords.size(0) > max_points:
        sel = torch.randperm(coords.size(0))[:max_points]
        coords, values = coords[sel], values[sel]

    lo, hi = values.min().item(), values.max().item()
    norm = ((values - lo) / max(hi - lo, 1e-8)).numpy()
    rgb = matplotlib.colormaps[cmap_name](norm)[:, :3] * 255.0

    center = (torch.tensor(vol.shape, dtype=torch.float32) - 1) / 2
    coords = (coords - center).numpy()

    return np.concatenate([coords, rgb], axis=1).astype(np.float32)


def _projection_mosaic(x: torch.Tensor) -> torch.Tensor:
    """
    (B, 1, D, H, W) -> (B, H, 3W): xy|xz|yz sum-projections concatenated horizontally.
    Assumes D=H=W (a cube), so the three projections share a common shape. The xy projection
    uses the same summing convention as corruption.project_2d, directly comparable to the
    observation channel.
    """
    xy = project_2d(x)     # (B, 1, H, W), sum over D
    xz = x.sum(dim=3)       # (B, 1, D, W), sum over H
    yz = x.sum(dim=4)       # (B, 1, D, H), sum over W
    return torch.cat([xy, xz, yz], dim=-1)[:, 0]  # (B, H, 3W)


@torch.no_grad()
def log_pretrain_reconstruction_3d(
    model: nn.Module,
    x_gt: torch.Tensor,    # (n, 1, D, H, W) canonical GT volumes to visualize
    noise_std: float,
    sample_steps: int,
    step: int,
    use_wandb: bool,
    n: int = 6,
) -> None:
    """
    3D analogue of log_pretrain_reconstruction. Draws a fresh random rotation for n GT volumes,
    builds the observation y = F(x_gt; R), and asks the CURRENT model to invert it via the
    E-step integrator (ode.sample_joint_3d, z ~ N(0,I) | y) -> (x_hat, pose_hat). Logs a 4-row
    panel (GT xy|xz|yz mosaic / Obs y / model x_hat's mosaic / F(x_hat) recorrupted) plus mean
    geodesic error — here a real accuracy metric (pretrain_3d.py has true GT rotations), unlike
    log_em_pool_diagnostics_3d's pool-vs-pool error which is diagnostic-only. ALSO logs an
    interactive companion for the SAME x_hat batch (no second sample_joint_3d call — reusing
    the mosaic panel's x_hat keeps the two views cross-referencable and avoids doubling the
    3D-UNet-pass cost): one wandb.Object3D point cloud per example, all under one key so
    wandb's media panel renders them as a steppable, individually-orbitable gallery — see
    _volume_to_pointcloud for the per-volume quantile thresholding this relies on. GT volumes
    are intentionally NOT logged as point clouds (the static mosaic already carries that
    comparison, and GT never changes call to call, so re-uploading it every --plot_every would
    be pure upload-budget waste — see _volume_to_pointcloud's docstring on cost).

    sample_joint_3d leaves the model in eval() mode (it doesn't restore it) — this function
    restores model.train() before returning so the caller's training loop is unaffected.
    """
    n = min(n, x_gt.size(0))
    x_gt = x_gt[:n]
    device = x_gt.device

    R_true = sample_uniform_rotation_so3(n, device)
    pose_true = matrix_to_gram_schmidt(R_true)
    y_obs, _ = forward_channel_3d(x_gt, noise_std=noise_std, pose6=pose_true)

    z_vol = torch.randn_like(x_gt)
    x_hat, pose_hat = sample_joint_3d(model, z_vol, y_obs, n_steps=sample_steps)
    model.train()

    proj_hat, _ = forward_channel_3d(x_hat, noise_std=0.0, pose6=pose_hat)
    geo_err = rotation_geodesic_angle(gram_schmidt_to_matrix(pose_hat), R_true).abs().mean().item()

    gt_mosaic = _projection_mosaic(x_gt).cpu()      # (n, H, 3W)
    obs = y_obs[:, 0].cpu()                          # (n, H, W)
    gen_mosaic = _projection_mosaic(x_hat).cpu()     # (n, H, 3W)
    proj = proj_hat[:, 0].cpu()                       # (n, H, W)

    # All four rows are sum-projections (unlike the 2D version's split between raw [-1,1]
    # image rows and 1D-signal rows) — one dynamic range per column, shared across all 4 rows.
    rows = [
        (gt_mosaic, "GT xy|xz|yz"),
        (obs, "Obs y"),
        (gen_mosaic, "Model x_hat xy|xz|yz"),
        (proj, "F(x_hat)"),
    ]
    fig, axes = plt.subplots(4, n, figsize=(2.4 * n, 8), squeeze=False)
    for r, (_, label) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for j in range(n):
        lo = min(data[j].min().item() for data, _ in rows)
        hi = max(data[j].max().item() for data, _ in rows)
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        for r, (data, _) in enumerate(rows):
            axes[r, j].imshow(data[j].numpy(), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])

    fig.suptitle(f"pretrain step {step}  |  mean geodesic error={geo_err:.3f} rad",
                fontsize=11)
    plt.tight_layout()

    # Built unconditionally (not gated on use_wandb) so `--debug --no_wandb` still exercises
    # this code path — see _volume_to_pointcloud's docstring for why the threshold is
    # per-volume rather than fixed.
    gen_clouds = [_volume_to_pointcloud(x_hat[i, 0]) for i in range(n)]

    if use_wandb:
        wandb.log({"pretrain/reconstruction": wandb.Image(fig),
                  "pretrain/geodesic_error": geo_err,
                  "pretrain/generations_3d": [wandb.Object3D(c) for c in gen_clouds]},
                 step=step)
    else:
        from pathlib import Path
        out = Path("mnist_cryoet_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"pretrain_step_{step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def log_em_pool_diagnostics_3d(
    pose_pool: torch.Tensor,     # (n, 6)     this iteration's Phi^(k-1)-recovered poses
    R_star: torch.Tensor,        # (N_obs, 3, 3)  diagnostic only, never used in training
    pool_indices: torch.Tensor,  # (n,) indices into R_star for this iteration's pool
    wandb_step: int,
    use_wandb: bool,
) -> None:
    """
    3D analogue of log_em_pool_diagnostics: mean geodesic angle between the pool's
    Phi^(k-1)-recovered rotations and the true rotations for the SAME sampled observations.
    """
    if not use_wandb:
        return
    R_pool = gram_schmidt_to_matrix(pose_pool)
    geo_err = rotation_geodesic_angle(R_pool, R_star[pool_indices]).abs().mean().item()
    wandb.log({"em/geodesic_error": geo_err}, step=wandb_step)


@torch.no_grad()
def log_reconstruction_grid_3d(
    model: nn.Module,
    x_gt: torch.Tensor,        # (n_images, 1, D, H, W)
    y_obs: torch.Tensor,       # (N_obs, 1, H, W)
    R_star: torch.Tensor,      # (N_obs, 3, 3)  diagnostic only, never used in training
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
    n_tilt_preview: int = 4,
) -> None:
    """
    3D/CryoET analogue of log_reconstruction_grid. Since each observation here is a full (H,W)
    image (not a 1D signal), the 2D version's (T,W)-sinogram-stacking trick doesn't apply —
    replaced with:

      Row 0 — GT volume's xy|xz|yz sum-projection mosaic (see _projection_mosaic).
      Row 1 — a small strip of n_tilt_preview evenly-spaced REAL tilt observations from that
              acquisition, horizontally concatenated.
      Row 2 — ONE generated x_hat's xy|xz|yz mosaic, conditioned on the acquisition's middle
              preview tilt's observation (the model only ever conditions on a single y; there's
              no joint multi-tilt conditioning to fall back on, so one representative tilt is
              chosen — mirrors the 2D version's limitation).
      Row 3 — that SAME x_hat reprojected (F, noise_std=0) at every preview tilt's TRUE pose,
              same strip layout as Row 1 — directly comparable to Row 1.
    Column titles show the conditioning tilt's recovered-vs-true geodesic error in degrees.

    Runs its own small E-step (ode.sample_joint_3d) directly rather than reusing
    scsi.py::propose_estep_3d — same circular-import reasoning as log_reconstruction_grid.

    ALSO logs an interactive companion for the SAME per-problem x_hat computed above (no
    second sample_joint_3d call): one wandb.Object3D point cloud per problem, all under one
    key so wandb's media panel renders a steppable, individually-orbitable gallery you can
    rotate/zoom directly in the browser — see _volume_to_pointcloud for the per-volume
    quantile thresholding this relies on. GT volumes are intentionally NOT logged as point
    clouds here — the static mosaic already carries that comparison, and re-uploading a GT
    volume every EM step would be pure upload-budget waste since it never changes.
    """
    other_ids = torch.randperm(n_acq)
    other_ids = other_ids[other_ids != fixed_acq_id][:n_problems - 1]
    acq_ids = [fixed_acq_id] + other_ids.tolist()
    n = len(acq_ids)

    gt_mosaics, obs_strips, gen_mosaics, gen_volumes, recorrupt_strips, col_titles = (
        [], [], [], [], [], [])
    T = n_tilt_preview

    for a in acq_ids:
        mask = acq_idx == a
        y_acq = y_obs[mask]
        R_star_acq = R_star[mask]
        img_idx = image_idx[mask][0].item()
        T = y_acq.size(0)

        if T > n_tilt_preview:
            rsel = torch.linspace(0, T - 1, n_tilt_preview).round().long()
            y_acq = y_acq[rsel]
            R_star_acq = R_star_acq[rsel]
            T = n_tilt_preview

        cond_i = T // 2
        z_vol = torch.randn(1, 1, *x_gt.shape[-3:], device=device)
        x_hat, pose_hat = sample_joint_3d(
            model, z_vol, y_acq[cond_i:cond_i + 1].to(device), n_steps=sample_steps,
        )

        recorrupt, _ = forward_channel_3d(
            x_hat.expand(T, -1, -1, -1, -1).contiguous(), noise_std=0.0,
            pose6=matrix_to_gram_schmidt(R_star_acq.to(device)),
        )  # (T, 1, H, W): the SAME single x_hat reprojected at every preview tilt's true pose

        gt_mosaics.append(_projection_mosaic(x_gt[img_idx:img_idx + 1]).cpu()[0])  # (H, 3W)
        obs_strips.append(torch.cat(list(y_acq[:, 0]), dim=-1).cpu())             # (H, T*W)
        gen_mosaics.append(_projection_mosaic(x_hat).cpu()[0])                     # (H, 3W)
        gen_volumes.append(x_hat[0, 0].cpu())                                      # (D, H, W)
        recorrupt_strips.append(torch.cat(list(recorrupt[:, 0]), dim=-1).cpu())   # (H, T*W)

        R_hat = gram_schmidt_to_matrix(pose_hat)
        rec_err_deg = rotation_geodesic_angle(
            R_hat, R_star_acq[cond_i:cond_i + 1].to(device)
        ).item() * 180.0 / torch.pi
        tag = " (fixed)" if a == fixed_acq_id else ""
        col_titles.append(f"acq={a}{tag}\ncond geodesic err: {rec_err_deg:.0f}°")

    model.train()

    fig, axes = plt.subplots(4, n, figsize=(2.6 * n, 9), squeeze=False)
    row_labels = ["GT xy|xz|yz", f"Obs y ({T} tilts)", f"pi({em_step}) xy|xz|yz",
                  "Recorrupt (true pose)"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9)

    # Row-PAIR scaling, same pattern as log_reconstruction_grid: (Obs y, Recorrupt) share one
    # scale, (GT mosaic, pi(k) mosaic) share another.
    for j in range(n):
        strip_lo = min(obs_strips[j].min().item(), recorrupt_strips[j].min().item())
        strip_hi = max(obs_strips[j].max().item(), recorrupt_strips[j].max().item())
        if strip_hi - strip_lo < 1e-8:
            strip_hi = strip_lo + 1e-8

        mosaic_lo = min(gt_mosaics[j].min().item(), gen_mosaics[j].min().item())
        mosaic_hi = max(gt_mosaics[j].max().item(), gen_mosaics[j].max().item())
        if mosaic_hi - mosaic_lo < 1e-8:
            mosaic_hi = mosaic_lo + 1e-8

        axes[0, j].imshow(gt_mosaics[j].numpy(), cmap="gray", vmin=mosaic_lo, vmax=mosaic_hi)
        axes[1, j].imshow(obs_strips[j].numpy(), cmap="gray", vmin=strip_lo, vmax=strip_hi,
                          aspect="auto")
        axes[2, j].imshow(gen_mosaics[j].numpy(), cmap="gray", vmin=mosaic_lo, vmax=mosaic_hi)
        axes[3, j].imshow(recorrupt_strips[j].numpy(), cmap="gray", vmin=strip_lo, vmax=strip_hi,
                          aspect="auto")
        axes[0, j].set_title(col_titles[j], fontsize=7)
        for r in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])

    fig.suptitle(f"em/reconstruction  |  EM step {em_step}", fontsize=11)
    plt.tight_layout()

    # Built unconditionally (not gated on use_wandb) so `--debug --no_wandb` still exercises
    # this code path — see _volume_to_pointcloud's docstring for why the threshold is
    # per-volume rather than fixed.
    gen_clouds = [_volume_to_pointcloud(v) for v in gen_volumes]

    if use_wandb:
        wandb.log({"em/reconstruction": wandb.Image(fig),
                  "em/generations_3d": [wandb.Object3D(c) for c in gen_clouds]},
                 step=wandb_step)
    else:
        from pathlib import Path
        out = Path("mnist_cryoet_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_reconstruction_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
