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

The ``perturbed`` bootstrap pre-trains the conditional velocity field to generate
random SO(3) rotations of the dataset object(s), conditioned on their projections
``F(x)``, then samples pi(0) from it (see :func:`scsi.pretrain_rotation_prior`).
The objects it rotates are the ``--shape`` mixture (see :data:`data.SHAPE_SAMPLERS`)
-- the same shapes the dataset is drawn from -- so the warmup covers every shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from .data import available_shapes, make_mixture_sampler

if TYPE_CHECKING:  # annotations only; avoids importing heavy / cyclic deps at runtime
    import torch.nn as nn

    from .tracking import Tracker


@dataclass
class BootstrapContext:
    """Everything a bootstrap might need to build pi(0).

    Not every field is used by every bootstrap (e.g. ``noise`` ignores ``y_obs``,
    and only ``perturbed`` uses the ``model`` / pre-training fields); bundling them
    keeps the registry signature uniform and future-proof.
    """

    y_obs: torch.Tensor          # (Nobj, 1, P, P) frozen observations
    n_objects: int               # Nobj
    n_points: int                # points per cloud (N)
    extent: float                # world half-extent mapped to the image
    seed: int
    shapes: list[str] = field(default_factory=lambda: ["torus"])  # dataset/warmup shape mixture
    perturb_std: float = 0.0     # optional 3D jitter on "perturbed" targets (0 = clean)

    # ── fields used only by the "perturbed" pre-trained generator ────────────
    model: "nn.Module | None" = None        # velocity net to pre-train (weights kept)
    device: "torch.device | None" = None
    radius: float = 0.08         # ball radius for the channel F
    noise_std: float = 0.1       # AWGN std on the projections F(x)
    image_size: int = 32         # projection size P
    channel: str = "so3"         # forward channel: "so3" | "so2" | "awgn_proj"
    so2_axis: str = "z"          # axis for the SO(2) pose ("x"/"y"/"z")
    pretrain_steps: int = 2000   # flow-matching steps before EM
    batch: int = 64
    lr: float = 2e-4
    sample_steps: int = 50       # Euler ODE steps used to draw pi(0)
    use_amp: bool = True
    tracker: "Tracker | None" = None
    global_step: "list | None" = None  # shared wandb step counter (kept monotonic)


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
    """pi(0) from a generator pre-trained on random rotations of the dataset object(s).

    Trains ``ctx.model`` (flow matching) to generate random SO(3) rotations of the
    ``ctx.shapes`` mixture coupled to their projections ``F(x)`` -- F uses its own
    fresh random pose -- then samples pi(0) from it on the real observations. The
    pre-trained weights are kept and warm-start the EM model. The warmup covers the
    same shape mixture as the dataset; ``perturb_std`` adds optional 3D jitter.
    """
    if ctx.model is None or ctx.device is None:
        raise ValueError(
            "the 'perturbed' bootstrap pre-trains a model; ctx.model and ctx.device "
            "must be set"
        )
    # Lazy import: the training routine lives with the rest of the training stack in
    # scsi.py; importing it at call time avoids an import cycle.
    from .scsi import pretrain_rotation_prior

    return pretrain_rotation_prior(
        ctx.model, ctx.y_obs,
        shape_fn=make_mixture_sampler(ctx.shapes),
        n_points=ctx.n_points, radius=ctx.radius, noise_std=ctx.noise_std,
        image_size=ctx.image_size, extent=ctx.extent,
        steps=ctx.pretrain_steps, batch=ctx.batch, lr=ctx.lr,
        sample_steps=ctx.sample_steps, perturb_std=ctx.perturb_std,
        shapes=ctx.shapes,
        device=ctx.device, use_amp=ctx.use_amp, seed=ctx.seed,
        tracker=ctx.tracker, global_step=ctx.global_step,
        channel=ctx.channel, so2_axis=ctx.so2_axis,
    )


def available_bootstraps() -> list[str]:
    """Registered bootstrap names (the valid ``--bootstrap`` choices)."""
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
