"""Toy data distributions: point clouds sampled from primitive surfaces.

Each sampler returns a *batch of clouds*; every cloud is a fresh random sample of
points on the same surface, so the model learns the distribution "points on a
<shape>". Several shapes are registered in :data:`SHAPE_SAMPLERS`;
:func:`make_mixture_sampler` builds a sampler that draws each object's shape
uniformly from a chosen subset (the dataset / bootstrap shape mixture). Swap any
of these for a ShapeNet / .npy loader and nothing else changes.
"""
from __future__ import annotations

import math
from typing import Callable

import torch


def sample_torus(
    batch: int,
    n_points: int,
    R: float = 1.0,   # distance from tube center to torus center (major radius)
    r: float = 0.4,   # tube radius (minor radius)
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return (batch, n_points, 3) point clouds on a torus.

    Note: sampling angles uniformly is not area-uniform on the torus, but it is
    a perfectly valid (slightly non-uniform) target distribution for the demo.
    """
    theta = torch.rand(batch, n_points, device=device) * 2 * math.pi  # around the ring
    phi = torch.rand(batch, n_points, device=device) * 2 * math.pi    # around the tube
    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)


def sample_cylinder(
    batch: int,
    n_points: int,
    radius: float = 0.6,   # tube radius
    height: float = 1.4,   # full height along z (centered at origin)
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return (batch, n_points, 3) point clouds on a closed cylinder surface.

    The cylinder's axis is z and it is centered at the origin; points cover the
    lateral side and the two end caps, allocated in proportion to their areas so
    the surface is sampled (roughly) uniformly. Scaled to sit comfortably inside
    the same world extent as :func:`sample_torus`.
    """
    h = height / 2.0
    side_area = 2.0 * math.pi * radius * height
    cap_area = math.pi * radius * radius                      # one cap
    p_side = side_area / (side_area + 2.0 * cap_area)
    on_side = torch.rand(batch, n_points, device=device) < p_side

    # lateral surface: uniform angle, uniform height
    theta = torch.rand(batch, n_points, device=device) * 2 * math.pi
    z_side = (torch.rand(batch, n_points, device=device) - 0.5) * height
    x_side, y_side = radius * torch.cos(theta), radius * torch.sin(theta)

    # end caps: area-uniform radius (sqrt), random top/bottom
    rr = torch.sqrt(torch.rand(batch, n_points, device=device)) * radius
    cap_theta = torch.rand(batch, n_points, device=device) * 2 * math.pi
    x_cap, y_cap = rr * torch.cos(cap_theta), rr * torch.sin(cap_theta)
    top = torch.rand(batch, n_points, device=device) < 0.5
    z_cap = torch.where(top, torch.full_like(rr, h), torch.full_like(rr, -h))

    x = torch.where(on_side, x_side, x_cap)
    y = torch.where(on_side, y_side, y_cap)
    z = torch.where(on_side, z_side, z_cap)
    return torch.stack([x, y, z], dim=-1)


def torus_surface_residual(
    clouds: torch.Tensor, R: float = 1.0, r: float = 0.4
) -> torch.Tensor:
    """Mean squared distance-to-surface proxy; ~ r**2 means "on the torus".

    For a point on the torus: (sqrt(x^2+y^2) - R)^2 + z^2 == r^2.
    """
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    val = (rho - R) ** 2 + clouds[..., 2] ** 2
    return val.mean()


# ── Signed distance to each surface (per point; 0 == on the surface) ──────────
# Used by the mixture residual diagnostic; unlike ``torus_surface_residual``
# above, these are centered so that 0 means "exactly on the surface".


def _torus_sd(clouds: torch.Tensor, R: float = 1.0, r: float = 0.4) -> torch.Tensor:
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    return torch.sqrt((rho - R) ** 2 + clouds[..., 2] ** 2) - r        # (B, N)


def _cylinder_sd(
    clouds: torch.Tensor, radius: float = 0.6, height: float = 1.4
) -> torch.Tensor:
    # Signed distance to a capped cylinder (axis z), 0 on the surface.
    h = height / 2.0
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    dr = rho - radius
    dz = clouds[..., 2].abs() - h
    outside = torch.sqrt(dr.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.maximum(dr, dz).clamp(max=0.0)
    return inside + outside                                            # (B, N)


# ── Shape registry + mixtures ────────────────────────────────────────────────
# name -> sampler (batch, n_points, *, device) -> (batch, n_points, 3)
SHAPE_SAMPLERS: dict[str, Callable[..., torch.Tensor]] = {
    "torus": sample_torus,
    "cylinder": sample_cylinder,
}
# name -> signed-distance-to-surface (per point); 0 means "on the surface".
_SHAPE_SD: dict[str, Callable[..., torch.Tensor]] = {
    "torus": _torus_sd,
    "cylinder": _cylinder_sd,
}


def available_shapes() -> list[str]:
    """Registered shape names (valid ``--shape`` choices)."""
    return sorted(SHAPE_SAMPLERS)


def _check_shapes(names: list[str]) -> None:
    unknown = [n for n in names if n not in SHAPE_SAMPLERS]
    if unknown:
        raise ValueError(f"unknown shape(s) {unknown}; choose from {available_shapes()}")
    if not names:
        raise ValueError("at least one shape is required")


def make_mixture_sampler(names: list[str]) -> Callable[..., torch.Tensor]:
    """Build a sampler over a uniform mixture of the named shapes.

    The returned callable has the same ``(batch, n_points, *, device)`` signature
    as the individual samplers; each object's shape is drawn uniformly from
    ``names`` (a single name collapses to that shape's sampler directly).
    """
    _check_shapes(names)
    if len(names) == 1:
        return SHAPE_SAMPLERS[names[0]]
    fns = [SHAPE_SAMPLERS[n] for n in names]

    def sampler(batch: int, n_points: int, device: torch.device | str = "cpu") -> torch.Tensor:
        which = torch.randint(len(fns), (batch,))
        clouds = torch.empty(batch, n_points, 3, device=device)
        for k, fn in enumerate(fns):
            idx = (which == k).nonzero(as_tuple=True)[0]
            if idx.numel():
                clouds[idx] = fn(int(idx.numel()), n_points, device=device)
        return clouds

    return sampler


def mixture_surface_residual(clouds: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Mean squared distance from each point to its *nearest* target surface.

    0 means every point lies on one of the ``names`` surfaces. For a single shape
    this is just that shape's squared surface distance; for a mixture each point
    is scored against whichever surface it is closest to.
    """
    _check_shapes(names)
    sds = torch.stack([_SHAPE_SD[n](clouds) for n in names], dim=0)   # (S, B, N)
    return sds.pow(2).amin(dim=0).mean()
