"""Ground-truth shape generators, for both `X` parameterizations.

Only ONE canonical template is ever built per run (the user picks a single shape);
"N datapoints" in the observed dataset means N independent noisy re-renders of that
one template under fresh random global poses (see `dataset.py`), not N distinct
shapes. Point-cloud and voxel shapes are registered separately (same 4 names) but
both are ultimately expressed as (positions, weights) -- the same representation
`render_projection` and the `X` parameterizations use (see `corruption.py`).
"""
from __future__ import annotations

import math

import torch


# ── Point-cloud ground truth (dense surface samples) ──────────────────────────


def sample_sphere_points(n_points: int, extent: float, device="cpu") -> torch.Tensor:
    radius = 0.5 * extent
    v = torch.randn(n_points, 3, device=device)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return v * radius


def sample_torus_points(n_points: int, extent: float, device="cpu") -> torch.Tensor:
    R, r = 0.5 * extent, 0.18 * extent
    theta = torch.rand(n_points, device=device) * 2 * math.pi
    phi = torch.rand(n_points, device=device) * 2 * math.pi
    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)


def sample_cylinder_points(n_points: int, extent: float, device="cpu") -> torch.Tensor:
    radius, height = 0.35 * extent, 1.0 * extent
    h = height / 2.0
    side_area = 2.0 * math.pi * radius * height
    cap_area = math.pi * radius * radius
    p_side = side_area / (side_area + 2.0 * cap_area)
    on_side = torch.rand(n_points, device=device) < p_side

    theta = torch.rand(n_points, device=device) * 2 * math.pi
    z_side = (torch.rand(n_points, device=device) - 0.5) * height
    x_side, y_side = radius * torch.cos(theta), radius * torch.sin(theta)

    rr = torch.sqrt(torch.rand(n_points, device=device)) * radius
    cap_theta = torch.rand(n_points, device=device) * 2 * math.pi
    x_cap, y_cap = rr * torch.cos(cap_theta), rr * torch.sin(cap_theta)
    top = torch.rand(n_points, device=device) < 0.5
    z_cap = torch.where(top, torch.full_like(rr, h), torch.full_like(rr, -h))

    x = torch.where(on_side, x_side, x_cap)
    y = torch.where(on_side, y_side, y_cap)
    z = torch.where(on_side, z_side, z_cap)
    return torch.stack([x, y, z], dim=-1)


def sample_helix_points(n_points: int, extent: float, device="cpu") -> torch.Tensor:
    R, pitch, n_turns, tube_radius = 0.4 * extent, 0.5 * extent, 2.5, 0.12 * extent
    t = torch.rand(n_points, device=device) * 2 * math.pi * n_turns
    z0 = pitch * n_turns / 2.0
    center = torch.stack(
        [R * torch.cos(t), R * torch.sin(t), pitch * t / (2 * math.pi) - z0], dim=-1
    )
    tangent = torch.stack(
        [-R * torch.sin(t), R * torch.cos(t), torch.full_like(t, pitch / (2 * math.pi))], dim=-1
    )
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    offset = torch.randn(n_points, 3, device=device)
    offset = offset - (offset * tangent).sum(-1, keepdim=True) * tangent  # perp to tangent
    offset = offset / offset.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r = torch.sqrt(torch.rand(n_points, device=device)) * tube_radius
    return center + offset * r.unsqueeze(-1)


def _trefoil_curve_raw(t: torch.Tensor) -> torch.Tensor:
    x = torch.sin(t) + 2 * torch.sin(2 * t)
    y = torch.cos(t) - 2 * torch.cos(2 * t)
    z = -torch.sin(3 * t)
    return torch.stack([x, y, z], dim=-1)


# Classic parametric trefoil (t in [0, 2*pi]) has a fixed, non-unit size (max
# radius 3.0) -- rescale by this constant so the knot's bounding radius matches
# the other shapes' `0.5 * extent` convention.
_TREFOIL_MAX_RADIUS = _trefoil_curve_raw(torch.linspace(0, 2 * math.pi, 10000)).norm(dim=-1).max().item()


def _trefoil_curve(t: torch.Tensor, extent: float) -> torch.Tensor:
    return _trefoil_curve_raw(t) * (0.5 * extent / _TREFOIL_MAX_RADIUS)


def sample_trefoil_points(n_points: int, extent: float, device="cpu") -> torch.Tensor:
    tube_radius = 0.12 * extent
    t = torch.rand(n_points, device=device) * 2 * math.pi
    center = _trefoil_curve(t, extent)

    dt = 1e-3
    tangent = _trefoil_curve(t + dt, extent) - _trefoil_curve(t - dt, extent)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    offset = torch.randn(n_points, 3, device=device)
    offset = offset - (offset * tangent).sum(-1, keepdim=True) * tangent  # perp to tangent
    offset = offset / offset.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r = torch.sqrt(torch.rand(n_points, device=device)) * tube_radius
    return center + offset * r.unsqueeze(-1)


def _letter_l_params(extent: float) -> tuple[float, float, float]:
    half = 0.5 * extent
    stroke = 0.28 * extent  # bar thickness in the x/z plane
    depth = 0.22 * extent  # half-thickness of the extrusion along y
    return half, stroke, depth


def _letter_l_mask(coords: torch.Tensor, extent: float) -> torch.Tensor:
    half, stroke, depth = _letter_l_params(extent)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    vertical = (x >= -half) & (x <= -half + stroke) & (z >= -half) & (z <= half)
    horizontal = (z >= -half) & (z <= -half + stroke) & (x >= -half) & (x <= half)
    return (vertical | horizontal) & (y.abs() <= depth)


def sample_letter_l_points(n_points: int, extent: float, device="cpu") -> torch.Tensor:
    half, stroke, depth = _letter_l_params(extent)
    batch = max(4 * n_points, 1024)
    kept: list[torch.Tensor] = []
    total = 0
    while total < n_points:
        cand = torch.rand(batch, 3, device=device) * 2 - 1
        cand[:, 0] *= half
        cand[:, 1] *= depth
        cand[:, 2] *= half
        hit = cand[_letter_l_mask(cand, extent)]
        kept.append(hit)
        total += hit.shape[0]
    return torch.cat(kept, dim=0)[:n_points]


POINTCLOUD_SHAPES = {
    "sphere": sample_sphere_points,
    "torus": sample_torus_points,
    "cylinder": sample_cylinder_points,
    "helix": sample_helix_points,
    "trefoil": sample_trefoil_points,
    "letter_l": sample_letter_l_points,
}


# ── Voxel ground truth (binary occupancy over a fixed grid) ───────────────────


def voxel_grid_coords(vol_size: int, extent: float, device="cpu") -> torch.Tensor:
    """Fixed (vol_size**3, 3) grid of voxel-center coordinates spanning [-extent, extent]^3."""
    lin = torch.linspace(-extent, extent, vol_size, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)


def _sphere_occ(coords: torch.Tensor, extent: float) -> torch.Tensor:
    radius = 0.5 * extent
    return (coords.norm(dim=-1) <= radius).float()


def _cylinder_occ(coords: torch.Tensor, extent: float) -> torch.Tensor:
    radius, half_height = 0.35 * extent, 0.5 * extent
    rho = coords[:, :2].norm(dim=-1)
    return ((rho <= radius) & (coords[:, 2].abs() <= half_height)).float()


def _torus_occ(coords: torch.Tensor, extent: float) -> torch.Tensor:
    R, r = 0.5 * extent, 0.18 * extent
    rho = coords[:, :2].norm(dim=-1)
    val = (rho - R) ** 2 + coords[:, 2] ** 2
    return (val <= r * r).float()


def _helix_occ(coords: torch.Tensor, extent: float) -> torch.Tensor:
    R, pitch, n_turns, tube_radius = 0.4 * extent, 0.5 * extent, 2.5, 0.12 * extent
    n_samples = 2000
    t = torch.linspace(0, 2 * math.pi * n_turns, n_samples, device=coords.device)
    z0 = pitch * n_turns / 2.0
    curve = torch.stack(
        [R * torch.cos(t), R * torch.sin(t), pitch * t / (2 * math.pi) - z0], dim=-1
    )  # (S, 3)
    d2 = torch.cdist(coords, curve).pow(2).min(dim=-1).values  # (M,)
    return (d2 <= tube_radius ** 2).float()


def _trefoil_occ(coords: torch.Tensor, extent: float) -> torch.Tensor:
    tube_radius = 0.12 * extent
    n_samples = 2000
    t = torch.linspace(0, 2 * math.pi, n_samples, device=coords.device)
    curve = _trefoil_curve(t, extent)  # (S, 3)
    d2 = torch.cdist(coords, curve).pow(2).min(dim=-1).values  # (M,)
    return (d2 <= tube_radius ** 2).float()


def _letter_l_occ(coords: torch.Tensor, extent: float) -> torch.Tensor:
    return _letter_l_mask(coords, extent).float()


VOXEL_SHAPES = {
    "sphere": _sphere_occ,
    "cylinder": _cylinder_occ,
    "torus": _torus_occ,
    "helix": _helix_occ,
    "trefoil": _trefoil_occ,
    "letter_l": _letter_l_occ,
}


def available_shapes() -> list[str]:
    """Registered shape names, valid for both parameterizations."""
    return sorted(set(POINTCLOUD_SHAPES) & set(VOXEL_SHAPES))


def make_ground_truth(
    shape: str, param: str, resolution: int, extent: float, device="cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the single ground-truth object as (positions, weights).

    ``param == "pointcloud"``: `resolution` dense surface points, all weight 1.
    ``param == "voxel"``: a fixed `resolution`-per-side grid with binary occupancy.
    """
    if param == "pointcloud":
        positions = POINTCLOUD_SHAPES[shape](resolution, extent, device=device)
        weights = torch.ones(positions.shape[0], device=device)
    else:
        positions = voxel_grid_coords(resolution, extent, device=device)
        weights = VOXEL_SHAPES[shape](positions, extent)
    return positions, weights
