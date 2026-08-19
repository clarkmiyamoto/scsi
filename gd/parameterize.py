"""The optimizable reconstruction `X`, as either a point cloud or a voxel grid.

Both reduce to the same (positions, weights) interface `render_projection` expects
(see `corruption.py`):

* `PointCloudParam`: learnable positions, fixed unit weights.
* `VoxelParam`: fixed grid-coordinate positions, learnable per-voxel weights (a
  `sigmoid` of a free logit -- a hard binary grid has no useful gradient).

Both can be initialized either randomly or from `corruption.carve_occupancy`'s soft
space-carving back-projection of a single observed tilt series (a "smart init" that
does not require training).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .corruption import carve_occupancy, sample_points_from_occupancy
from .shapes import voxel_grid_coords


class PointCloudParam(nn.Module):
    """Optimizable point cloud: `positions` are the only learnable parameters."""

    def __init__(
        self, n_points: int, init_positions: torch.Tensor | None = None, device="cpu"
    ):
        super().__init__()
        if init_positions is None:
            init_positions = torch.randn(n_points, 3, device=device) * 0.3
        self.positions = nn.Parameter(init_positions.to(device).float())
        self.register_buffer("_weights", torch.ones(self.positions.shape[0], device=device))

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    def render_args(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions, self.weights


class VoxelParam(nn.Module):
    """Optimizable dense voxel grid: fixed coordinates, learnable occupancy logits."""

    def __init__(
        self,
        vol_size: int,
        extent: float,
        init_logits: torch.Tensor | None = None,
        device="cpu",
    ):
        super().__init__()
        self.vol_size = vol_size
        self.extent = extent
        self.register_buffer("coords", voxel_grid_coords(vol_size, extent, device=device))
        if init_logits is None:
            init_logits = torch.randn(vol_size ** 3, device=device) * 0.1 - 2.0
        self.logits = nn.Parameter(init_logits.to(device).float())

    @property
    def weights(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def render_args(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coords, self.weights


def _inverse_sigmoid(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p / (1.0 - p))


def init_from_backprojection_pointcloud(
    y_obs_single: torch.Tensor,
    n_points: int,
    tilt_step: float,
    tilt_axis: str,
    extent: float,
    vol_size: int = 32,
    carve_quantile: float = 0.15,
    seed: int = 0,
    device="cpu",
) -> torch.Tensor:
    occ, vox = carve_occupancy(
        y_obs_single, tilt_step=tilt_step, tilt_axis=tilt_axis,
        extent=extent, vol_size=vol_size, carve_quantile=carve_quantile,
    )
    pts = sample_points_from_occupancy(occ, vox, vol_size, extent, n_points, seed=seed)
    return pts.to(device)


def init_from_backprojection_voxel(
    y_obs_single: torch.Tensor,
    vol_size: int,
    tilt_step: float,
    tilt_axis: str,
    extent: float,
    carve_quantile: float = 0.15,
    device="cpu",
) -> torch.Tensor:
    occ, _ = carve_occupancy(
        y_obs_single, tilt_step=tilt_step, tilt_axis=tilt_axis,
        extent=extent, vol_size=vol_size, carve_quantile=carve_quantile,
    )
    occ = occ.clamp(1e-4, 1.0 - 1e-4)
    return _inverse_sigmoid(occ).to(device)
