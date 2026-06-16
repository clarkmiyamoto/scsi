"""Toy data distribution: point clouds sampled from a torus surface.

Each call returns a *batch of clouds*; every cloud is a fresh random sample of
points on the same torus, so the model learns the distribution "points on a
torus". Swap this function for a ShapeNet / .npy loader and nothing else changes.
"""
from __future__ import annotations

import math

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


def torus_surface_residual(
    clouds: torch.Tensor, R: float = 1.0, r: float = 0.4
) -> torch.Tensor:
    """Mean squared distance-to-surface proxy; ~ r**2 means "on the torus".

    For a point on the torus: (sqrt(x^2+y^2) - R)^2 + z^2 == r^2.
    """
    rho = torch.sqrt(clouds[..., 0] ** 2 + clouds[..., 1] ** 2)
    val = (rho - R) ** 2 + clouds[..., 2] ** 2
    return val.mean()
