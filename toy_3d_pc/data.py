"""Toy data distributions: point clouds sampled uniformly from solid volumes.

Each shape is defined as a **solid** -- a signed-distance function (SDF) over R^3, negative
inside -- plus a bounding box to rejection-sample within. :func:`_sample_solid` is the one
generic sampler every shape routes through: draw candidates uniformly in the bounding box,
keep the ones with ``sd <= 0``, repeat (with growing candidate counts) until ``n_points`` are
collected per cloud. Every sampler call returns a *batch of clouds*; every cloud is a fresh
random sample of points inside the same solid volume, so the model learns the distribution
"points inside a <shape>". Each shape's SDF doubles as the ``mixture_volume_residual``
diagnostic. :func:`make_mixture_sampler` draws each object's shape uniformly from a chosen
subset; :func:`sample_perturbed_dataset` is the cryo-ET / subtomogram view (many noisy copies
of fixed template(s)). Clouds live in world coordinates (~[-1.6, 1.6]), *not* normalized to
[-1, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


# ── Generic solid definition + rejection sampler ──────────────────────────────


@dataclass(frozen=True)
class _Solid:
    sd: Callable[[torch.Tensor], torch.Tensor]   # (..., 3) -> (...); <=0 == inside
    bbox: tuple[float, float, float]              # half-extents to sample candidates within
    oversample: int = 4                            # initial candidates requested per point


def _sample_solid(
    solid: _Solid,
    batch: int,
    n_points: int,
    device: torch.device | str = "cpu",
    max_rounds: int = 10,
) -> torch.Tensor:
    """Rejection-sample ``n_points`` points uniformly inside ``solid``, per cloud in the batch.

    Candidates are drawn uniformly in ``[-bbox, bbox]``, accepted where ``solid.sd <= 0``.
    Low fill-fraction solids (thin tubes, sparse unions) need more candidates per accepted
    point; each round doubles the candidate count (up to ``max_rounds``) instead of assuming a
    fixed oversample ratio.
    """
    bbox = torch.tensor(solid.bbox, device=device)
    out = torch.empty(batch, n_points, 3, device=device)
    for b in range(batch):
        collected: list[torch.Tensor] = []
        have = 0
        n_try = n_points * solid.oversample
        for _ in range(max_rounds):
            if have >= n_points:
                break
            cand = (torch.rand(n_try, 3, device=device) * 2.0 - 1.0) * bbox
            accept = cand[solid.sd(cand) <= 0]
            if accept.numel():
                collected.append(accept)
                have += accept.shape[0]
            n_try *= 2
        if have < n_points:
            raise RuntimeError(
                f"solid sampler only found {have}/{n_points} interior points after "
                f"{max_rounds} rounds -- shape's bbox/sd is too sparse; widen oversample."
            )
        out[b] = torch.cat(collected, dim=0)[:n_points]
    return out


# ── Signed distance primitives ─────────────────────────────────────────────────


def _sphere_sd(p: torch.Tensor, center: tuple[float, float, float], radius: float) -> torch.Tensor:
    c = torch.tensor(center, device=p.device, dtype=p.dtype)
    return (p - c).norm(dim=-1) - radius


def _capsule_sd(
    p: torch.Tensor,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    radius: float,
) -> torch.Tensor:
    """Distance to a solid capsule (rounded cylinder) around segment a->b."""
    pa_a = torch.tensor(a, device=p.device, dtype=p.dtype)
    pa_b = torch.tensor(b, device=p.device, dtype=p.dtype)
    ba = pa_b - pa_a
    pa = p - pa_a
    h = (pa @ ba / ba.dot(ba)).clamp(0.0, 1.0)
    return (pa - ba * h[..., None]).norm(dim=-1) - radius


def _box_sd(
    p: torch.Tensor, center: tuple[float, float, float], half_extents: tuple[float, float, float]
) -> torch.Tensor:
    c = torch.tensor(center, device=p.device, dtype=p.dtype)
    he = torch.tensor(half_extents, device=p.device, dtype=p.dtype)
    q = (p - c).abs() - he
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


# ── Signed distance to each shape's surface (per point; 0 == on the surface) ──


def _torus_sd(clouds: torch.Tensor, R: float = 1.0, r: float = 0.4) -> torch.Tensor:
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    return torch.sqrt((rho - R) ** 2 + clouds[..., 2] ** 2) - r        # (B, N)


_DUMBBELL_H = 0.6          # half-distance between ball centers
_DUMBBELL_R_BALL = 0.35    # ball radius
_DUMBBELL_R_ROD = 0.15     # connecting rod radius


def _dumbbell_sd(clouds: torch.Tensor) -> torch.Tensor:
    s1 = _sphere_sd(clouds, (0.0, 0.0, -_DUMBBELL_H), _DUMBBELL_R_BALL)
    s2 = _sphere_sd(clouds, (0.0, 0.0, _DUMBBELL_H), _DUMBBELL_R_BALL)
    rod = _capsule_sd(
        clouds, (0.0, 0.0, -_DUMBBELL_H), (0.0, 0.0, _DUMBBELL_H), _DUMBBELL_R_ROD
    )
    return torch.minimum(torch.minimum(s1, s2), rod)


_TREFOIL_TUBE_R = 0.18
_TREFOIL_SCALE = 0.45
_TREFOIL_N_POLY = 300


def _trefoil_polyline(device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    s = torch.linspace(0.0, 2.0 * math.pi, _TREFOIL_N_POLY + 1, device=device, dtype=dtype)[:-1]
    x = torch.sin(s) + 2.0 * torch.sin(2.0 * s)
    y = torch.cos(s) - 2.0 * torch.cos(2.0 * s)
    z = -torch.sin(3.0 * s)
    return torch.stack([x, y, z], dim=-1) * _TREFOIL_SCALE   # (n_poly, 3)


_trefoil_poly_cache: dict[tuple, torch.Tensor] = {}


def _trefoil_sd(clouds: torch.Tensor) -> torch.Tensor:
    key = (str(clouds.device), clouds.dtype)
    if key not in _trefoil_poly_cache:
        _trefoil_poly_cache[key] = _trefoil_polyline(clouds.device, clouds.dtype)
    poly = _trefoil_poly_cache[key]                                    # (M, 3)
    lead = clouds.shape[:-1]
    flat = clouds.reshape(-1, 3)                                       # (L, 3)
    d = torch.cdist(flat, poly)                                        # (L, M)
    return (d.amin(dim=-1) - _TREFOIL_TUBE_R).reshape(lead)


def _l_shape_sd(clouds: torch.Tensor) -> torch.Tensor:
    vert = _box_sd(clouds, (0.0, 0.0, 0.0), (0.25, 0.25, 0.8))
    horiz = _box_sd(clouds, (0.4, 0.0, -0.55), (0.65, 0.25, 0.25))
    return torch.minimum(vert, horiz)


def _t_shape_sd(clouds: torch.Tensor) -> torch.Tensor:
    top = _box_sd(clouds, (0.0, 0.0, 0.55), (0.8, 0.25, 0.25))
    stem = _box_sd(clouds, (0.0, 0.0, -0.15), (0.25, 0.25, 0.65))
    return torch.minimum(top, stem)


# ── Shape registry ──────────────────────────────────────────────────────────

_SHAPE_SOLIDS: dict[str, _Solid] = {
    "torus": _Solid(_torus_sd, bbox=(1.4, 1.4, 0.4), oversample=4),
    "dumbbell": _Solid(_dumbbell_sd, bbox=(0.35, 0.35, 0.95), oversample=8),
    "trefoil": _Solid(_trefoil_sd, bbox=(1.6, 1.6, 0.6), oversample=32),
    "l_shape": _Solid(_l_shape_sd, bbox=(1.05, 0.25, 0.8), oversample=4),
    "t_shape": _Solid(_t_shape_sd, bbox=(0.8, 0.25, 0.8), oversample=4),
}


def _make_sampler(name: str) -> Callable[..., torch.Tensor]:
    solid = _SHAPE_SOLIDS[name]

    def sampler(batch: int, n_points: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return _sample_solid(solid, batch, n_points, device=device)

    return sampler


sample_torus = _make_sampler("torus")
sample_dumbbell = _make_sampler("dumbbell")
sample_trefoil = _make_sampler("trefoil")
sample_l_shape = _make_sampler("l_shape")
sample_t_shape = _make_sampler("t_shape")

SHAPE_SAMPLERS: dict[str, Callable[..., torch.Tensor]] = {
    name: _make_sampler(name) for name in _SHAPE_SOLIDS
}
_SHAPE_SD: dict[str, Callable[..., torch.Tensor]] = {
    name: solid.sd for name, solid in _SHAPE_SOLIDS.items()
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
