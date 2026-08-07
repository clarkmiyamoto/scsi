import argparse
from pathlib import Path

import torch
import torch.nn as nn

from scsi import loss_func_joint
from wandb_logging import log_warmup_step, log_pretrain_reconstruction

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

########################################################
# --warmup_steps phase (dispatched from main.py, before the EM loop starts): a supervised
# stochastic-interpolant warm-start using CLASSICAL (filtered-backprojection) reconstructions of
# main.py's own D_Y as the interpolant target -- i.e. the same idea as
# pretrain.py::build_classical_recon_pool ("--warmstart_target classical_recon"), but run
# in-process against the SAME observation pool the EM loop is about to train on, instead of a
# separate pretraining script's own train-split pool + checkpoint hand-off.
#
# backproject_pool is the shared piece: pretrain.py's build_classical_recon_pool calls it too
# (after its own data.build_observations call), so the per-acquisition backproject+calibration
# logic exists exactly once.
########################################################


def backproject_pool(
    y_obs: torch.Tensor,        # (N_obs, 1, W)
    theta_star: torch.Tensor,   # (N_obs,)  TRUE angle per observation (a legitimate oracle here
                                 #           -- synthetic data; see classical_recon.py docstring)
    acq_idx: torch.Tensor,      # (N_obs,)  which acquisition each observation belongs to;
                                 #           observations from the same acquisition are
                                 #           contiguous and in tilt order (data.build_observations)
    x_gt: torch.Tensor,         # (n_images, 1, H, W) -- only used for image_size + calibration
                                 #                       target mean/std, never for x1 itself
    recon_filter_type: str,
    recon_calibration: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Backproject EVERY acquisition in (y_obs, theta_star, acq_idx) into its own canonical-frame
    classical reconstruction, broadcast across that acquisition's own observations, then
    calibrate the whole pool to x_gt's aggregate scale. Returns x_recon_pool: (N_obs, 1, H, W),
    index-aligned with y_obs/theta_star/acq_idx.

    This is the per-acquisition loop from pretrain.py::build_classical_recon_pool, decoupled from
    the data.build_observations call that builds (y_obs, theta_star, acq_idx) in the first place
    -- so it can run against observations built by EITHER pretrain.py's own disjoint train-split
    pool OR main.py's own EM-loop observation pool. See both callers' docstrings for the
    calibration rationale (single global affine fit from the whole pool's aggregate mean/std to
    x_gt's aggregate mean/std, then clamp to [-1, 1] -- deliberately NOT a per-acquisition fit).

    `device`, if given, moves each acquisition's slice to `device` for the backproject() call and
    the result back to CPU before concatenating -- the same "compute on device, store on CPU"
    pattern scsi.py::propose_estep already uses. Pass None (default) when y_obs/theta_star/x_gt
    are already resident on the device you want the computation to run on (pretrain.py's case,
    where the whole pool is GPU-resident already); pass device=<the run's device> when they're
    CPU-resident and you want GPU-accelerated backprojection without holding the whole
    reconstruction pool on the GPU (main.py's case).
    """
    from classical_recon import backproject  # lazy: keeps classical_recon.py's skimage import off
                                              # the default (no-warmup / gt-warmstart) path

    image_size = x_gt.size(-1)
    n_acq = int(acq_idx.max().item()) + 1

    x_recon_chunks = []
    for a in range(n_acq):
        mask = acq_idx == a
        y_a = y_obs[mask].to(device) if device is not None else y_obs[mask]
        theta_a = theta_star[mask].to(device) if device is not None else theta_star[mask]
        x_hat_a = backproject(y_a, theta_a, image_size,
                              filtered=True, filter_type=recon_filter_type)  # (1,1,H,W)
        if device is not None:
            x_hat_a = x_hat_a.cpu()
        x_recon_chunks.append(x_hat_a.expand(int(mask.sum()), -1, -1, -1))
    # Concatenation order matches y_obs/theta_star's order exactly: build_observations already
    # keeps each acquisition's observations contiguous and in tilt order.
    x_recon_pool = torch.cat(x_recon_chunks, dim=0)

    if recon_calibration == "affine_clamp":
        a_coef = x_gt.std() / x_recon_pool.std().clamp_min(1e-8)
        b_coef = x_gt.mean() - a_coef * x_recon_pool.mean()
        x_recon_pool = (a_coef * x_recon_pool + b_coef).clamp(-1.0, 1.0)
    elif recon_calibration != "none":
        raise ValueError(f"Unknown recon_calibration: {recon_calibration!r}")

    return x_recon_pool


def classical_target_for_display(
    x_recon_pool: torch.Tensor | None,
    image_idx: torch.Tensor | None,
    n_images: int,
) -> torch.Tensor | None:
    """
    One representative calibrated reconstruction per source object, aligned with x_gt's own
    object order -- for wandb_logging.log_pretrain_reconstruction's optional classical_target
    row. Robust to multiple acquisitions per object (picks that object's FIRST acquisition's
    reconstruction, not a stride-based index, since image_idx's grouping is the only thing this
    function should rely on). Returns None when there's no reconstruction pool to display.
    """
    if x_recon_pool is None:
        return None
    return torch.stack([x_recon_pool[image_idx == i][0] for i in range(n_images)])


def run_classical_recon_warmup(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    x_gt: torch.Tensor,        # (n_images, 1, H, W)
    y_obs: torch.Tensor,       # (N_obs, 1, W)         -- main.py's own D_Y, REUSED, not resampled
    theta_star: torch.Tensor,  # (N_obs,)              -- TRUE angle; oracle target, synthetic data
    image_idx: torch.Tensor,   # (N_obs,)
    acq_idx: torch.Tensor,     # (N_obs,)
    args: argparse.Namespace,
    device: torch.device,
    use_wandb: bool,
    ckpt_dir: Path,
) -> int:
    """
    Supervised stochastic-interpolant warm-start on main.py's own observation pool, run in-process
    right before the EM loop: x1 = a classical (filtered-backprojection) reconstruction of each
    acquisition in (y_obs, theta_star, acq_idx), y = that SAME acquisition's real observations
    (never resynthesized -- see backproject_pool's docstring and
    pretrain.py::build_classical_recon_pool's, which documents the same choice). Unlike
    pretrain.py, there is no separate pretraining pool here: D_Y is main.py's own EM-loop
    observation set, built with the SAME --corruptions_per_object/--n_tilts/--tilt_increment_deg/
    --noise_std flags that size the EM loop's pool.

    Returns the number of warmup steps taken (== args.warmup_steps), so main.py can pass it as
    em.run_em_loop's `global_step_start` and keep one continuous, non-decreasing wandb step axis
    across warmup -> EM (wandb requires non-decreasing steps per run -- see run_em_loop's
    docstring).
    """
    image_size = x_gt.size(-1)
    x_recon_pool = backproject_pool(
        y_obs, theta_star, acq_idx, x_gt,
        recon_filter_type=args.warmup_recon_filter_type,
        recon_calibration=args.warmup_recon_calibration,
        device=device,
    )
    print(f"Warmup classical-recon pool: {x_recon_pool.size(0)} (x_hat, y) pairs "
          f"(filter={args.warmup_recon_filter_type}, calibration={args.warmup_recon_calibration})  "
          f"x_hat stats: min={x_recon_pool.min():.3f} max={x_recon_pool.max():.3f} "
          f"mean={x_recon_pool.mean():.4f} std={x_recon_pool.std():.4f}")

    N = y_obs.size(0)
    model.train()
    running_img, running_pose = 0.0, 0.0

    def _log_panel(step: int):
        log_pretrain_reconstruction(
            model, x_gt.to(device), noise_std=args.noise_std,
            sample_steps=args.sample_steps, step=step, use_wandb=use_wandb,
            classical_target=classical_target_for_display(
                x_recon_pool, image_idx, x_gt.size(0)).to(device),
        )

    for step in tqdm(range(args.warmup_steps), desc="warmup"):
        idx = torch.randint(0, N, (args.batch_size,))
        x_batch = x_recon_pool[idx].to(device)   # classical FBP reconstruction, calibrated
        theta_batch = theta_star[idx].to(device)  # TRUE angle for that observation
        y_batch = y_obs[idx].to(device)           # the REAL observation -- not resynthesized

        loss, loss_img, loss_pose = loss_func_joint(
            model, x_batch, theta_batch, y_batch,
            style=args.interpolant_style, pose_loss_weight=args.pose_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_img += loss_img.item()
        running_pose += loss_pose.item()
        log_warmup_step(loss_img, loss_pose, grad_norm, step, use_wandb)

        if (step + 1) % args.warmup_checkpoint_every == 0:
            torch.save(model.state_dict(), ckpt_dir / "model_warmup.pt")

        if args.warmup_plot_every > 0 and (step + 1) % args.warmup_plot_every == 0:
            _log_panel(step)

    if args.warmup_steps > 0:
        if args.warmup_plot_every > 0 and args.warmup_steps % args.warmup_plot_every != 0:
            _log_panel(args.warmup_steps - 1)
        torch.save(model.state_dict(), ckpt_dir / "model_warmup.pt")
        print(f"warmup steps={args.warmup_steps}  "
              f"loss_image={running_img / args.warmup_steps:.5f}  "
              f"loss_pose={running_pose / args.warmup_steps:.5f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return args.warmup_steps
