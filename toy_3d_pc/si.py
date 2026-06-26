"""Stochastic interpolant for point clouds: schedules, interpolant, transport ODE.

Convention (standard flow matching): ``t: 0 -> 1`` runs ``noise -> data``.

    I_t   = alpha_t * z + beta_t * x        z ~ N(0, I)  (noise),  x ~ data
    dI_t  = alpha_dot_t * z + beta_dot_t * x

The velocity network ``b_t`` is trained to regress ``dI_t``; sampling integrates
``dX/dt = b_t(X, t | y)`` from ``t=0`` (z') to ``t=1`` (x-hat). The pseudocode's
``X_{t=1} = z'`` is the time-reversed convention -- identical dynamics, integrated
``0 -> 1`` here. ``linear`` is the optimal-transport interpolant; ``gvp`` is the
geometric (variance-preserving) one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _schedule(t: torch.Tensor, style: str):
    """Return (alpha, beta, alpha_dot, beta_dot) at times ``t`` (any broadcast shape)."""
    if style == "linear":
        return (1.0 - t), t, -torch.ones_like(t), torch.ones_like(t)
    if style == "gvp":
        half_pi = math.pi / 2.0
        a = torch.cos(half_pi * t)
        b = torch.sin(half_pi * t)
        return a, b, -half_pi * b, half_pi * a
    raise ValueError(f"unknown interpolant style {style!r}; choose 'linear' or 'gvp'")


def interpolant(
    z: torch.Tensor,    # (B, N, 3) noise
    x: torch.Tensor,    # (B, N, 3) data / target clean cloud
    t: torch.Tensor,    # (B,) times in [0, 1]
    style: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(I_t, dI_t/dt)`` for the chosen schedule."""
    tt = t.view(-1, *([1] * (x.dim() - 1)))           # (B, 1, 1) broadcast over (N, 3)
    a, b, a_dot, b_dot = _schedule(tt, style)
    I_t = a * z + b * x
    I_dot = a_dot * z + b_dot * x
    return I_t, I_dot


@torch.no_grad()
def transport_sample(
    model: nn.Module,
    z0: torch.Tensor,        # (B, N, 3) initial noise z'
    y: torch.Tensor,         # (B, K, P, P) conditioning observation
    n_steps: int = 50,
) -> torch.Tensor:
    """Euler-integrate the conditional flow ODE from t=0 (z') to t=1 -> x-hat (B, N, 3).

    This is ``Phi(z' | y)``. The network uses only LayerNorm (mode-independent), so no
    train/eval toggle is needed for correctness. Runs under ``no_grad``.

    ``y`` is fixed across all ODE steps, so we encode it once and pass the cached
    context tokens on every step instead of re-running the image encoder 64 times.
    """
    x = z0
    B = z0.size(0)
    dt = 1.0 / n_steps
    # Encode observation once; fall back to passing y if the model predates this API.
    ctx = model.encode_obs(y) if hasattr(model, "encode_obs") else None
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=z0.device, dtype=z0.dtype)
        if ctx is not None:
            x = x + model(x, t, ctx=ctx) * dt
        else:
            x = x + model(x, t, y) * dt
    return x
