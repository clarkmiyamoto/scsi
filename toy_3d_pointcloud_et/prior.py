"""Bootstrap prior pi(0) for lifted SCSI (CryoET).

Turns the frozen tilt-series observations into an initial pool of clean-cloud
candidates ``x_pool`` of shape ``(Nobj, n_points, 3)`` (CPU) that seeds the
supervised pretraining phase and then the first E-step in
:func:`scsi.scsi_train`.

The only bootstrap is ``"tomo"``: 3D space-carving back-projection of the K-tilt
series using the known tilt geometry, giving a visual-hull reconstruction whose
residual global orientation the EM loop resolves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BootstrapContext:
    """Fields needed to build the tomo pi(0)."""

    y_obs: torch.Tensor    # (Nobj, K, P, P) frozen CryoET tilt series
    n_objects: int         # Nobj
    n_points: int          # N points per cloud
    extent: float
    seed: int
    tilt_step: float = 12.0
    tilt_axis: str = "y"
    tomo_vol: int = 48
    tomo_quantile: float = 0.15


BootstrapFn = Callable[[BootstrapContext], torch.Tensor]
BOOTSTRAPS: dict[str, BootstrapFn] = {}


def register(name: str) -> Callable[[BootstrapFn], BootstrapFn]:
    """Decorator that adds a bootstrap to the registry under ``name``."""

    def deco(fn: BootstrapFn) -> BootstrapFn:
        BOOTSTRAPS[name] = fn
        return fn

    return deco


@register("tomo")
def _tomo(ctx: BootstrapContext) -> torch.Tensor:
    """Tomographic back-projection of a CryoET tilt series into a 3D point cloud.

    Lifts the K known-tilt projections in ``ctx.y_obs`` (shape (Nobj, K, P, P)) into a
    point cloud via 3D back-projection using only the known tilt geometry; the
    residual global orientation is left for EM to resolve.
    """
    from .corruption import backproject_tomo

    return backproject_tomo(
        ctx.y_obs, ctx.n_points, ctx.tilt_step, ctx.tilt_axis,
        extent=ctx.extent, vol_size=ctx.tomo_vol,
        carve_quantile=ctx.tomo_quantile, seed=ctx.seed,
    )


def available_bootstraps() -> list[str]:
    """Registered bootstrap names."""
    return sorted(BOOTSTRAPS)


def make_bootstrap(name: str, ctx: BootstrapContext) -> torch.Tensor:
    """Build pi(0) using the registered bootstrap ``name``.

    Returns an ``(Nobj, n_points, 3)`` tensor on CPU.
    """
    if name not in BOOTSTRAPS:
        raise ValueError(
            f"unknown bootstrap {name!r}; choose from {available_bootstraps()}"
        )
    return BOOTSTRAPS[name](ctx)
