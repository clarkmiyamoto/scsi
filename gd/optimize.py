"""Direct gradient-descent reconstruction loop: no EM, no neural network -- Adam
directly optimizes X's own parameters (point positions, or voxel occupancy logits,
plus per-particle poses for `joint_pose`) against a fixed observed CryoET dataset.
"""
from __future__ import annotations

import math
import os

import torch

from .corruption import render_projection
from .dataset import build_dataset
from .eval import evaluate_reconstruction, voxel_to_pointcloud
from .parameterize import (
    PointCloudParam,
    VoxelParam,
    init_from_backprojection_pointcloud,
    init_from_backprojection_voxel,
)
from .plot import save_scatter, save_tilt_montage
from .recon import get_recon_method
from .tracking import Tracker


def _build_param(args, dataset, device):
    if args.param == "pointcloud":
        init_positions = None
        if args.init == "backproject":
            init_positions = init_from_backprojection_pointcloud(
                dataset.y_obs[0], args.n_points, args.tilt_step, args.tilt_axis,
                args.extent, vol_size=args.tomo_vol, seed=args.seed, device=device,
            )
        return PointCloudParam(args.n_points, init_positions=init_positions, device=device)

    init_logits = None
    if args.init == "backproject":
        init_logits = init_from_backprojection_voxel(
            dataset.y_obs[0], args.vol_size, args.tilt_step, args.tilt_axis,
            args.extent, device=device,
        )
    return VoxelParam(args.vol_size, args.extent, init_logits=init_logits, device=device)


def _lr_factor(step: int, n_steps: int, schedule: str, min_frac: float) -> float:
    """Multiplicative factor on each param group's initial LR, decaying 1.0 -> `min_frac`
    over `n_steps`. Relative (not absolute), so it applies uniformly to `lr_shape` and
    `lr_pose` groups regardless of their differing base LRs."""
    t = min(step / max(n_steps - 1, 1), 1.0)
    if schedule == "cosine":
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "linear":
        return 1.0 + (min_frac - 1.0) * t
    if schedule == "exponential":
        return min_frac ** t
    return 1.0


def _build_scheduler(optimizer, args) -> torch.optim.lr_scheduler.LambdaLR | None:
    if args.lr_schedule == "none":
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_factor(step, args.n_steps, args.lr_schedule, args.lr_min_frac),
    )


def _ground_truth_cloud(dataset, args) -> torch.Tensor:
    if args.param == "voxel":
        return voxel_to_pointcloud(
            dataset.gt_positions, dataset.gt_weights, args.eval_points, threshold=0.5
        )
    return dataset.gt_positions


def _log_viz(tracker, X, args, y_batch, step, render_kwargs) -> None:
    positions, weights = X.render_args()
    if args.param == "pointcloud":
        cloud, img_key = positions, "recon_cloud"
    else:
        # Reduce the voxel grid to a point cloud (same reduction used for the
        # Chamfer eval) so it can be viewed the same way as the point-cloud case.
        cloud = voxel_to_pointcloud(positions, weights, args.eval_points, threshold=0.5)
        img_key = "recon_voxel"

    cloud_path = os.path.join(args.out_dir, f"{img_key}_{step:06d}.png")
    save_scatter(cloud, cloud_path, lim=args.extent)
    tracker.log_image(img_key, cloud_path, step=step)
    # Interactive rotate/zoom viewer in the W&B UI (Object3D), alongside the static PNG.
    tracker.log_cloud("recon_cloud_3d", cloud, step=step)

    with torch.no_grad():
        y_sim = render_projection(positions, weights, batch=1, **render_kwargs)[0]
    montage_path = os.path.join(args.out_dir, f"tilts_{step:06d}.png")
    save_tilt_montage(y_batch[0], y_sim, montage_path)
    tracker.log_image("tilt_montage", montage_path, step=step)


def run(args, device: torch.device):
    render_kwargs = dict(
        n_tilts=args.n_tilts, tilt_step=args.tilt_step, tilt_axis=args.tilt_axis,
        image_size=args.image_size, extent=args.extent,
        splat_kernel=args.splat_kernel, splat_size=args.splat_size,
        noise_std=args.noise_std,
    )

    dataset = build_dataset(
        shape=args.shape, param=args.param, n_objects=args.n_objects,
        n_gt_points_or_vol=args.gt_resolution, seed=args.seed, device=device,
        **render_kwargs,
    )
    gt_cloud = _ground_truth_cloud(dataset, args)

    X = _build_param(args, dataset, device)

    method_kwargs = {"dist_metric": args.dist_metric} if args.method == "distributional" else {}
    method = get_recon_method(args.method, **method_kwargs)
    state = method.init_state(args.n_objects, device)

    param_groups = [{"params": list(X.parameters()), "lr": args.lr_shape}]
    extra_params = method.parameters(state)
    if extra_params:
        param_groups.append({"params": extra_params, "lr": args.lr_pose})
    optimizer = torch.optim.Adam(param_groups)
    scheduler = _build_scheduler(optimizer, args)

    os.makedirs(args.out_dir, exist_ok=True)

    with Tracker(project=args.wandb_project, name=args.wandb_name, config=vars(args)) as tracker:
        tracker.log_cloud("ground_truth", gt_cloud, step=0)
        for step in range(args.n_steps):
            idx_batch = torch.randint(0, args.n_objects, (args.batch,), device=device)
            y_batch = dataset.y_obs[idx_batch]

            optimizer.zero_grad()
            positions, weights = X.render_args()
            loss = method.loss((positions, weights), render_kwargs, y_batch, idx_batch, state)
            if args.param == "voxel" and args.l1_weight > 0:
                loss = loss + args.l1_weight * weights.mean()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if step % args.log_every == 0 or step == args.n_steps - 1:
                positions, weights = X.render_args()
                metrics = evaluate_reconstruction(
                    positions, weights, args.param, gt_cloud, n_points=args.eval_points
                )
                lr = optimizer.param_groups[0]["lr"]
                tracker.log({"loss": loss.item(), "chamfer": metrics["chamfer"], "lr": lr}, step=step)
                print(f"[step {step:5d}] loss={loss.item():.5f}  chamfer={metrics['chamfer']:.5f}  lr={lr:.2e}")
                _log_viz(tracker, X, args, y_batch, step, render_kwargs)

    return X
