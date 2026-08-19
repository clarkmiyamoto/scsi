"""Build the fixed observed CryoET tilt-series dataset D = {y_i} for one chosen
ground-truth shape (the user's step 1-2): N independent noisy re-renders of a single
template under fresh random unknown poses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .corruption import render_projection
from .shapes import make_ground_truth


@dataclass
class Dataset:
    y_obs: torch.Tensor          # (N, T, P, P) fixed observed tilt series
    gt_positions: torch.Tensor   # ground-truth positions (matches `param`)
    gt_weights: torch.Tensor     # matching weights


def build_dataset(
    shape: str,
    param: str,                 # "pointcloud" | "voxel"
    n_objects: int,              # N
    n_gt_points_or_vol: int,     # ground-truth resolution (dense point count or voxel grid side)
    seed: int,
    device: torch.device,
    **render_kwargs,             # n_tilts, tilt_step, tilt_axis, image_size, extent,
                                  # splat_kernel, splat_size, noise_std
) -> Dataset:
    torch.manual_seed(seed)
    np.random.seed(seed)  # scipy's Rotation.random draws from numpy's global RNG

    gt_positions, gt_weights = make_ground_truth(
        shape, param, n_gt_points_or_vol, render_kwargs["extent"], device=device
    )
    with torch.no_grad():
        y_obs = render_projection(gt_positions, gt_weights, batch=n_objects, **render_kwargs)
    return Dataset(y_obs=y_obs, gt_positions=gt_positions, gt_weights=gt_weights)
