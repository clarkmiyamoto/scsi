"""Render a 2D point cloud into a ``(P, P)`` density image (the ``G`` in the CryoEM channel).

Three interchangeable representations of a point cloud ``X in R^{N x 2}`` as an image, all
implemented so any can be selected via ``--splat``:

  * ``gaussian`` -- an isotropic Gaussian blob at every point (separable, default-cost).
  * ``disk``     -- a filled 2D disk at every point (chord-length profile of a solid disk;
                    the 2D analogue of ``toy_3d_pc``'s filled-*ball* splat).
  * ``histogram``-- hard binning of point positions onto the ``(P,P)`` lattice (counts). The
                    literal "discretize on a lattice" representation.

This module renders the object in its own canonical (unrotated) frame; :mod:`corruption`
applies pose + tilt rotation and projection to the rendered image to form the CryoEM channel.
Rendering is never backpropped through (the SCSI loss regresses the velocity field, not
through ``F``), so the non-differentiable hard-binning ``histogram`` kernel is fine.
"""
from __future__ import annotations

import torch


def gaussian_splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float, radius: float
) -> torch.Tensor:
    """Separable isotropic-Gaussian splat. px/py: (..., N) -> (..., P, P).

    ``g(dx, dy) = gx(dx) * gy(dy)`` so the image is an einsum over per-axis factors --
    O(prod(...) * N * P) memory. Leading dims are arbitrary ((B,) typically).
    """
    device = px.device
    grid = torch.linspace(-extent, extent, image_size, device=device)  # (P,)
    sigma = max(float(radius), 1e-4)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    dx = grid - px[..., None]   # (..., N, P)
    dy = grid - py[..., None]   # (..., N, P)
    gx = torch.exp(-dx * dx * inv_2s2)
    gy = torch.exp(-dy * dy * inv_2s2)
    return torch.einsum("...ni,...nj->...ij", gy, gx)  # (..., P, P) [row=y, col=x]


def disk_splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float, radius: float,
    chunk: int = 64,
) -> torch.Tensor:
    """Filled-disk splat. px/py: (..., N) -> (..., P, P).

    A filled 2D disk of radius ``r`` has chord-length profile ``sqrt(r^2 - d^2)`` for
    in-plane distance ``d <= r`` from center (peak-normalized to 1 at the center, 0 at the
    rim) -- the 2D analogue of ``toy_3d_pc``'s filled-*ball* splat (there, the projection of
    a solid ball onto 2D; here, the disk itself is already 2D so no projection is needed).
    Not separable, so the ``(..., N, P, P)`` work tensor is built in chunks over N.
    """
    device = px.device
    grid = torch.linspace(-extent, extent, image_size, device=device)  # (P,)
    r = max(float(radius), 1e-4)
    inv_r2 = 1.0 / (r * r)

    lead = px.shape[:-1]
    n = px.shape[-1]
    pxf = px.reshape(-1, n)   # (L, N)
    pyf = py.reshape(-1, n)
    img = torch.zeros(pxf.shape[0], image_size, image_size, device=device)  # (L, P, P)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dx = grid - pxf[:, s:e, None]          # (L, c, P) along x (cols)
        dy = grid - pyf[:, s:e, None]          # (L, c, P) along y (rows)
        d2 = dy[:, :, :, None] ** 2 + dx[:, :, None, :] ** 2     # (L, c, P, P) [row=y, col=x]
        val = torch.sqrt(torch.clamp(1.0 - d2 * inv_r2, min=0.0))
        img = img + val.sum(dim=1)
    return img.reshape(*lead, image_size, image_size)


def histogram_splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float,
) -> torch.Tensor:
    """Hard 2D binning of point positions onto the (P, P) lattice. px/py: (..., N) -> (..., P, P).

    Each point is assigned to the bin its coordinate falls into (uniform bins spanning
    ``[-extent, extent]``); the image holds per-bin point counts. Non-differentiable (no
    gradient needs to flow through the forward channel).
    """
    device = px.device
    P = image_size
    scale = P / (2.0 * extent)
    col = ((px + extent) * scale).floor().long().clamp(0, P - 1)   # (..., N)
    row = ((py + extent) * scale).floor().long().clamp(0, P - 1)

    lead = px.shape[:-1]
    n = px.shape[-1]
    flat_row = row.reshape(-1, n)
    flat_col = col.reshape(-1, n)
    flat_idx = flat_row * P + flat_col                              # (L, N)
    img = torch.zeros(flat_idx.shape[0], P * P, device=device, dtype=torch.float32)
    img.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    return img.reshape(*lead, P, P)


def render(
    points: torch.Tensor,           # (B, N, 2)
    image_size: int,
    extent: float,
    radius: float,
    kind: str,
) -> torch.Tensor:
    """Dispatch to the chosen splat kernel. points (B, N, 2) -> (B, P, P) image.

    ``kind`` is required (no default) -- callers must choose {"gaussian", "disk", "histogram"}.
    """
    px, py = points[..., 0], points[..., 1]
    if kind == "gaussian":
        return gaussian_splat(px, py, image_size, extent, radius)
    if kind == "disk":
        return disk_splat(px, py, image_size, extent, radius)
    if kind == "histogram":
        return histogram_splat(px, py, image_size, extent)
    raise ValueError(f"unknown splat kind {kind!r}; choose 'gaussian', 'disk', or 'histogram'")
