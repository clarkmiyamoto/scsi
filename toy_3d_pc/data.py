"""Toy data distributions: point clouds sampled uniformly from primitive volumes.

Each sampler returns a *batch of clouds*; every cloud is a fresh random sample of
points inside the same solid volume, so the model learns the distribution "points
inside a <shape>". :func:`make_mixture_sampler` draws each object's shape uniformly
from a chosen subset; :func:`sample_perturbed_dataset` is the cryo-ET / subtomogram
view (many noisy copies of fixed template(s)). Swap any sampler for a ShapeNet / .npy
loader and nothing else changes -- clouds live in world coordinates (~[-1.6, 1.6]),
*not* normalized to [-1, 1].
"""
from __future__ import annotations

import math
from typing import Callable

import torch


def sample_torus(
    batch: int,
    n_points: int,
    R: float = 1.0,   # major radius (tube center to torus center)
    r: float = 0.4,   # minor radius (tube radius)
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return (batch, n_points, 3) point clouds uniform in the solid torus volume.

    Uses rejection sampling inside the bounding annular cylinder.  Acceptance rate
    = π/4 ≈ 78.5%; 3× oversampling makes shortfall probability negligible.
    """
    n_sample = n_points * 3
    # Sample rho uniform in annular disk (density ∝ rho via inverse-CDF trick)
    u = torch.rand(batch, n_sample, device=device)
    rho = torch.sqrt(4.0 * R * r * u + (R - r) ** 2)
    theta = torch.rand(batch, n_sample, device=device) * 2 * math.pi
    z = (torch.rand(batch, n_sample, device=device) - 0.5) * 2.0 * r
    accept = (rho - R).pow(2) + z.pow(2) <= r ** 2   # inside the tube cross-section

    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    clouds_full = torch.stack([x, y, z], dim=-1)   # (batch, n_sample, 3)

    out = torch.empty(batch, n_points, 3, device=device)
    for b in range(batch):
        idx = accept[b].nonzero(as_tuple=True)[0]
        out[b] = clouds_full[b, idx[:n_points]]
    return out


def sample_cylinder(
    batch: int,
    n_points: int,
    radius: float = 0.6,
    height: float = 1.4,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return (batch, n_points, 3) clouds uniform in the solid cylinder volume (axis z, centered)."""
    r_sample = torch.sqrt(torch.rand(batch, n_points, device=device)) * radius
    theta = torch.rand(batch, n_points, device=device) * 2 * math.pi
    z = (torch.rand(batch, n_points, device=device) - 0.5) * height
    x = r_sample * torch.cos(theta)
    y = r_sample * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)


# ── Signed distance to each surface (per point; 0 == on the surface) ──────────


def _torus_sd(clouds: torch.Tensor, R: float = 1.0, r: float = 0.4) -> torch.Tensor:
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    return torch.sqrt((rho - R) ** 2 + clouds[..., 2] ** 2) - r        # (B, N)


def _cylinder_sd(
    clouds: torch.Tensor, radius: float = 0.6, height: float = 1.4
) -> torch.Tensor:
    h = height / 2.0
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    dr = rho - radius
    dz = clouds[..., 2].abs() - h
    outside = torch.sqrt(dr.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.maximum(dr, dz).clamp(max=0.0)
    return inside + outside                                            # (B, N)


# ── Shape registry + mixtures ────────────────────────────────────────────────

SHAPE_SAMPLERS: dict[str, Callable[..., torch.Tensor]] = {
    "torus": sample_torus,
    "cylinder": sample_cylinder,
}
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

    Same ``(batch, n_points, device)`` signature as the individual samplers; each
    object's shape is drawn uniformly from ``names`` (single name collapses to it).
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


def sample_perturbed_dataset(
    names: list[str],
    n_objects: int,
    n_points: int,
    perturb_eps: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``n_objects`` bounded perturbations of fixed canonical template(s).

    One canonical template cloud per shape; each object is a template (shape drawn
    uniformly when ``len(names) > 1``) plus a per-point shift uniform in the
    ``perturb_eps``-ball (``||delta_n|| <= perturb_eps``). ``perturb_eps == 0`` gives
    identical copies -- the cryo-ET / subtomogram-averaging dataset. Returns
    (n_objects, n_points, 3).
    """
    _check_shapes(names)
    templates = [SHAPE_SAMPLERS[n](1, n_points, device=device)[0] for n in names]  # each (N,3)
    which = torch.randint(len(names), (n_objects,))
    out = torch.empty(n_objects, n_points, 3, device=device)
    for k, template in enumerate(templates):
        idx = (which == k).nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = template.unsqueeze(0).expand(idx.numel(), -1, -1)
    if perturb_eps > 0:
        d = torch.randn_like(out)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # unit directions
        rad = torch.rand(n_objects, n_points, 1, device=device).pow(1.0 / 3.0) * perturb_eps
        out = out + d * rad                                    # uniform in the eps-ball
    return out


def mixture_volume_residual(clouds: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Mean squared exterior distance from each point to its nearest target volume.

    0 means every point lies inside one of the ``names`` volumes.  Interior points
    (negative signed distance) score 0; exterior points are penalised by the squared
    distance to the nearest volume boundary.
    """
    _check_shapes(names)
    sds = torch.stack([_SHAPE_SD[n](clouds) for n in names], dim=0)   # (S, B, N)
    return sds.amin(dim=0).clamp(min=0).pow(2).mean()
