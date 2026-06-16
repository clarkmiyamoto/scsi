"""Bootstrap priors pi(0) for lifted SCSI.

A *bootstrap* turns the problem inputs into the initial pool of clean-cloud
candidates ``x_pool`` of shape ``(Nobj, n_points, 3)`` (CPU) that seeds the very
first E-step in :func:`scsi.scsi_train`. The choice of warmstart matters a lot
for whether EM converges to the data distribution, so this file collects them in
one place and makes adding a new one a single decorated function.

Add a bootstrap::

    @register("my_bootstrap")
    def _my_bootstrap(ctx: BootstrapContext) -> torch.Tensor:
        return ...  # (ctx.n_objects, ctx.n_points, 3) on CPU

The CLI ``--bootstrap`` choices are generated from :func:`available_bootstraps`,
so registering here is all that's needed to expose it.

The ``perturbed`` bootstrap starts from the dataset object (a torus) and pushes
it off-surface by Gaussian noise. The object it perturbs away from is one entry
in :data:`SHAPES` -- add an entry (or pass ``--perturb-shape``) to perturb from a
different object without touching the bootstrap itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .data import sample_torus

# ── shapes the "perturbed" bootstrap can start from ──────────────────────────
# name -> sampler with signature (batch, n_points) -> (batch, n_points, 3) on CPU.
# Add an entry here to perturb away from a different dataset object.
ShapeFn = Callable[..., torch.Tensor]
SHAPES: dict[str, ShapeFn] = {
    "torus": sample_torus,
}


@dataclass
class BootstrapContext:
    """Everything a bootstrap might need to build pi(0).

    Not every field is used by every bootstrap (e.g. ``noise`` ignores ``y_obs``);
    bundling them keeps the registry signature uniform and future-proof.
    """

    y_obs: torch.Tensor          # (Nobj, 1, P, P) frozen observations
    n_objects: int               # Nobj
    n_points: int                # points per cloud (N)
    extent: float                # world half-extent mapped to the image
    seed: int
    perturb_std: float = 0.3     # noise scale for the "perturbed" bootstrap
    perturb_shape: str = "torus"  # which SHAPES entry "perturbed" starts from


BootstrapFn = Callable[[BootstrapContext], torch.Tensor]
BOOTSTRAPS: dict[str, BootstrapFn] = {}


def register(name: str) -> Callable[[BootstrapFn], BootstrapFn]:
    """Decorator that adds a bootstrap to the registry under ``name``."""

    def deco(fn: BootstrapFn) -> BootstrapFn:
        BOOTSTRAPS[name] = fn
        return fn

    return deco


@register("backproject")
def _backproject(ctx: BootstrapContext) -> torch.Tensor:
    """Lift each 2D observation into a "puffed silhouette" point cloud."""
    # Lazy import: corruption pulls in scipy, and we want importing this module
    # (e.g. for the CLI's --bootstrap choices) to stay cheap.
    from .corruption import backproject_bootstrap

    return backproject_bootstrap(
        ctx.y_obs, ctx.n_points, extent=ctx.extent, seed=ctx.seed
    )


@register("noise")
def _noise(ctx: BootstrapContext) -> torch.Tensor:
    """Pure Gaussian pi(0) ~ N(0, I)."""
    g = torch.Generator().manual_seed(ctx.seed)
    return torch.randn(ctx.n_objects, ctx.n_points, 3, generator=g)


@register("perturbed")
def _perturbed(ctx: BootstrapContext) -> torch.Tensor:
    """pi(0) = (fresh samples of the dataset object) + Gaussian noise.

    Starts from ``perturb_shape`` (default the torus the data lives on) and
    pushes it off-surface by ``perturb_std`` -- a warmstart that already knows the
    rough geometry. Swap the object via :data:`SHAPES` / ``--perturb-shape``.
    """
    if ctx.perturb_shape not in SHAPES:
        raise ValueError(
            f"unknown perturb_shape {ctx.perturb_shape!r}; "
            f"choose from {available_shapes()}"
        )
    shape_fn = SHAPES[ctx.perturb_shape]
    # sample_torus draws from the global RNG; fork so the bootstrap is reproducible
    # from ctx.seed alone without disturbing the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ctx.seed)
        clean = shape_fn(ctx.n_objects, ctx.n_points)    # (Nobj, N, 3) on CPU
    g = torch.Generator().manual_seed(ctx.seed)
    noise = torch.randn(clean.shape, generator=g) * ctx.perturb_std
    return clean + noise


def available_bootstraps() -> list[str]:
    """Registered bootstrap names (the valid ``--bootstrap`` choices)."""
    return sorted(BOOTSTRAPS)


def available_shapes() -> list[str]:
    """Registered shape names the ``perturbed`` bootstrap can start from."""
    return sorted(SHAPES)


def make_bootstrap(name: str, ctx: BootstrapContext) -> torch.Tensor:
    """Build pi(0) using the registered bootstrap ``name``.

    Returns an ``(Nobj, n_points, 3)`` tensor on CPU.
    """
    if name not in BOOTSTRAPS:
        raise ValueError(
            f"unknown bootstrap {name!r}; choose from {available_bootstraps()}"
        )
    return BOOTSTRAPS[name](ctx)
