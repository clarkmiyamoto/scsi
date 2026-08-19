"""Evaluate a reconstructed `X` against the known ground truth.

`X` is only ever recovered up to the unknown global SO(3) rotation the direct
optimization loop never observes (the corruption channel bakes in a random pose per
particle), so comparison must go through blind ICP alignment rather than a raw
Chamfer distance.
"""
from __future__ import annotations

import torch

from .canonicalize import chamfer, icp_align


def voxel_to_pointcloud(
    coords: torch.Tensor,     # (M, 3) fixed voxel-center grid
    weights: torch.Tensor,    # (M,) occupancy in [0, 1]
    n_points: int,
    threshold: float | None = None,
) -> torch.Tensor:
    """Reduce a voxel `X` to a point cloud: either every center above `threshold`,
    or (default) a weighted sample of `n_points` centers, for Chamfer comparison."""
    w = weights.detach()
    if threshold is not None:
        idx = (w >= threshold).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            return coords[idx]
        idx = torch.topk(w, min(n_points, w.numel())).indices
        return coords[idx]
    probs = (w + 1e-6) / (w + 1e-6).sum()
    idx = torch.multinomial(probs, n_points, replacement=True)
    return coords[idx]


def evaluate_reconstruction(
    positions: torch.Tensor,
    weights: torch.Tensor,
    param: str,
    ground_truth: torch.Tensor,   # (M_gt, 3) point cloud, already reduced if needed
    n_points: int = 1024,
    n_iters: int = 6,
    n_restarts: int = 8,
) -> dict:
    if param == "voxel":
        recon_cloud = voxel_to_pointcloud(positions, weights, n_points)
    else:
        recon_cloud = positions.detach()

    mu_r = recon_cloud.mean(dim=0, keepdim=True)
    mu_g = ground_truth.mean(dim=0, keepdim=True)
    ref_centered = ground_truth - mu_g
    _, aligned = icp_align(
        (recon_cloud - mu_r).unsqueeze(0), ref_centered, n_iters=n_iters, n_restarts=n_restarts
    )
    err = chamfer(aligned, ref_centered.unsqueeze(0)).item()
    return {"chamfer": err, "aligned_cloud": aligned[0] + mu_g}
