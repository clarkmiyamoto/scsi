import argparse
from pathlib import Path

import torch
import torch.nn as nn

from scsi import (
    sample_pool_indices, propose_estep, train_mstep,
    propose_estep_3d, train_mstep_3d,
)
from wandb_logging import (
    log_em_pool_diagnostics, log_reconstruction_grid,
    log_em_pool_diagnostics_3d, log_reconstruction_grid_3d,
)

########################################################
# The outer EM loop: for k = 1..K, run one E-step then one M-step (scsi.py), so that
# Phi^(k-1) (the model used to propose the pool) and Theta^(k) (the model trained on it)
# line up exactly with the algorithm's indexing.
########################################################


def run_em_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    y_obs: torch.Tensor,       # (N_obs, 1, W)
    x_gt: torch.Tensor,        # (n_images, 1, H, W)  diagnostic only, for wandb panels
    theta_star: torch.Tensor,  # (N_obs,)             diagnostic only
    image_idx: torch.Tensor,   # (N_obs,)             diagnostic only
    acq_idx: torch.Tensor,     # (N_obs,)             diagnostic only, groups tilt series
    args: argparse.Namespace,
    device: torch.device,
    use_wandb: bool,
    ckpt_dir: Path,
) -> None:
    N_obs = y_obs.size(0)
    n_acq = int(acq_idx.max().item()) + 1
    fixed_acq_id = 0  # tracked for the whole run so its reconstruction is comparable over time
    steps_first_em = args.steps_first_em if args.steps_first_em is not None else args.steps_per_em
    global_step = [0]

    for k in range(args.n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k} / {args.n_em_steps}")
        print("=" * 60)

        steps = steps_first_em if k == 0 else args.steps_per_em
        pool_size = min(N_obs, steps * args.batch_size)
        idx = sample_pool_indices(N_obs, pool_size)   # fresh "y ~ mu" draw

        print(f"  E-step: pool_size={pool_size}  (Phi^({k-1}))")
        x_pool, theta_pool = propose_estep(
            model, y_obs[idx],
            n_steps=args.sample_steps, batch_size=args.batch_size * 2,
            method="euler", device=device,
        )

        # Visualization only, using this same Phi^(k-1) snapshot (before train_mstep updates
        # the weights) so every diagnostic for iteration k reflects one consistent model.
        log_reconstruction_grid(
            model, x_gt, y_obs, theta_star, image_idx, acq_idx,
            fixed_acq_id=fixed_acq_id, n_acq=n_acq, sample_steps=args.sample_steps,
            em_step=k, wandb_step=global_step[0], use_wandb=use_wandb, device=device,
        )

        train_mstep(
            model, opt, x_pool, theta_pool,
            noise_std=args.noise_std, style=args.interpolant_style,
            pose_loss_weight=args.pose_loss_weight,
            steps_per_em=steps, batch_size=args.batch_size,
            global_step=global_step, device=device, use_wandb=use_wandb,
        )

        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:04d}.pt")
        print(f"  checkpoint saved  ->  {ckpt_dir}/model_em{k:04d}.pt")

        log_em_pool_diagnostics(
            theta_pool, theta_star, idx,
            wandb_step=global_step[0], use_wandb=use_wandb,
        )


########################################################
# 3D / CryoET outer EM loop. Mirrors run_em_loop line-for-line — only the E/M-step and
# diagnostic calls change; the k-loop / steps_first_em / pool-size logic is copied verbatim.
########################################################

def run_em_loop_3d(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    y_obs: torch.Tensor,       # (N_obs, 1, H, W)
    x_gt: torch.Tensor,        # (n_images, 1, D, H, W)  diagnostic only, for wandb panels
    R_star: torch.Tensor,      # (N_obs, 3, 3)           diagnostic only
    image_idx: torch.Tensor,   # (N_obs,)                diagnostic only
    acq_idx: torch.Tensor,     # (N_obs,)                diagnostic only, groups tilt series
    args: argparse.Namespace,
    device: torch.device,
    use_wandb: bool,
    ckpt_dir: Path,
) -> None:
    N_obs = y_obs.size(0)
    n_acq = int(acq_idx.max().item()) + 1
    fixed_acq_id = 0  # tracked for the whole run so its reconstruction is comparable over time
    steps_first_em = args.steps_first_em if args.steps_first_em is not None else args.steps_per_em
    global_step = [0]

    for k in range(args.n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k} / {args.n_em_steps}")
        print("=" * 60)

        steps = steps_first_em if k == 0 else args.steps_per_em
        pool_size = min(N_obs, steps * args.batch_size)
        idx = sample_pool_indices(N_obs, pool_size)   # fresh "y ~ mu" draw

        print(f"  E-step: pool_size={pool_size}  (Phi^({k-1}))")
        x_pool, pose_pool = propose_estep_3d(
            model, y_obs[idx],
            n_steps=args.sample_steps, batch_size=args.batch_size * 2,
            method="euler", device=device, vol_size=args.vol_size,
        )

        # Visualization only, using this same Phi^(k-1) snapshot (before train_mstep_3d updates
        # the weights) so every diagnostic for iteration k reflects one consistent model.
        log_reconstruction_grid_3d(
            model, x_gt, y_obs, R_star, image_idx, acq_idx,
            fixed_acq_id=fixed_acq_id, n_acq=n_acq, sample_steps=args.sample_steps,
            em_step=k, wandb_step=global_step[0], use_wandb=use_wandb, device=device,
        )

        train_mstep_3d(
            model, opt, x_pool, pose_pool,
            noise_std=args.noise_std, style=args.interpolant_style,
            pose_loss_weight=args.pose_loss_weight,
            steps_per_em=steps, batch_size=args.batch_size,
            global_step=global_step, device=device, use_wandb=use_wandb,
        )

        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:04d}.pt")
        print(f"  checkpoint saved  ->  {ckpt_dir}/model_em{k:04d}.pt")

        log_em_pool_diagnostics_3d(
            pose_pool, R_star, idx,
            wandb_step=global_step[0], use_wandb=use_wandb,
        )
