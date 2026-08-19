"""CryoET forward channel: one shared weighted-point renderer for both `X`
parameterizations (point cloud and voxel).

Models a CryoET tilt series: one *unknown* global SO(3) orientation ``R(theta)`` (the
nuisance), composed under ``K`` *known* tilt rotations ``R_tilt(n * Delta-theta)``
about a fixed lab axis, giving K projections

    F(X) = { splat_k( R_tilt(n*Delta-theta) R(theta) positions, weights ) + Z }_{n=1..K}

A point cloud ``X`` and a voxel grid ``X`` are both just **weighted 3D points** here:

* point cloud: ``positions`` are the learnable point coordinates, ``weights`` are
  fixed at 1 (every point contributes equally).
* voxel grid: ``positions`` are a *fixed* grid of voxel centers, ``weights`` are
  ``sigmoid(logits)`` where ``logits`` is the learnable per-voxel parameter.

Both feed into :func:`render_projection`, so the splat-kernel choice and size apply
identically regardless of parameterization -- there is exactly one forward channel.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

SplatKernel = Literal["gaussian", "ball", "cube"]


def tilt_rotations(
    n_tilts: int,
    tilt_step_deg: float,
    axis: str = "y",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rotation matrices for a single-axis CryoET tilt series, shape (K, 3, 3).

    The K *known* tilt increments are symmetric about 0,
    ``((arange(K) - (K-1)/2) * tilt_step_deg)`` degrees about ``axis`` ("x"/"y").
    """
    n = np.arange(n_tilts) - (n_tilts - 1) / 2.0
    angles = (n * tilt_step_deg)[:, None]
    mats = Rotation.from_euler(axis, angles, degrees=True).as_matrix().astype(np.float32)
    return torch.from_numpy(mats).to(device)


# ── Splat kernels ──────────────────────────────────────────────────────────────
# Each kernel is the orthographic projection of a solid 3D primitive of the given
# `size` centered at the point: a soft Gaussian bump, a solid sphere's radially
# varying shadow, or a solid cube's constant-height square shadow.


def _separable_splat(
    px: torch.Tensor, py: torch.Tensor, weights: torch.Tensor,
    image_size: int, extent: float, size: float, kernel: SplatKernel,
) -> torch.Tensor:
    """Fast O(N*P) splat for kernels that factor as f(dx) * f(dy): ``gaussian`` and
    ``cube`` (a solid cube's shadow is a separable box: ``box(dx) * box(dy)``).

    px/py/weights: (..., N) -> (..., P, P) [row=y, col=x].
    """
    device = px.device
    grid = torch.linspace(-extent, extent, image_size, device=device)  # (P,)
    size = max(float(size), 1e-4)

    dx = grid - px[..., None]   # (..., N, P)
    dy = grid - py[..., None]   # (..., N, P)
    if kernel == "gaussian":
        inv_2s2 = 1.0 / (2.0 * size * size)
        gx = torch.exp(-dx * dx * inv_2s2)
        gy = torch.exp(-dy * dy * inv_2s2)
    else:  # cube: constant-height (2*size) box footprint of a solid cube
        gx = (dx.abs() <= size).to(dx.dtype) * (2.0 * size)
        gy = (dy.abs() <= size).to(dy.dtype)
    gx = gx * weights[..., None]
    return torch.einsum("...ni,...nj->...ij", gy, gx)  # (..., P, P)


def _ball_splat(
    px: torch.Tensor, py: torch.Tensor, weights: torch.Tensor,
    image_size: int, extent: float, size: float,
) -> torch.Tensor:
    """Direct O(N*P^2) splat: the projected shadow of a solid sphere of radius
    ``size``, height(r) = 2*sqrt(size^2 - r^2) for r <= size, else 0.

    Not separable in (dx, dy) (r depends on both), so unlike the other kernels this
    materializes the full (..., N, P, P) distance field before reducing over N --
    notably more expensive for large N (e.g. a full voxel grid: keep `vol_size`
    modest when using this kernel with the voxel parameterization).
    """
    device = px.device
    grid = torch.linspace(-extent, extent, image_size, device=device)  # (P,)
    size = max(float(size), 1e-4)

    dx = grid - px[..., None]   # (..., N, P)  P indexes columns (x)
    dy = grid - py[..., None]   # (..., N, P)  P indexes rows (y)
    r2 = dy[..., :, :, None] ** 2 + dx[..., :, None, :] ** 2  # (..., N, P_row, P_col)
    s2 = size * size
    height = torch.where(
        r2 <= s2, 2.0 * torch.sqrt((s2 - r2).clamp(min=0.0)), torch.zeros_like(r2)
    )
    height = height * weights[..., :, None, None]
    return height.sum(dim=-3)  # (..., P, P)


def _splat(
    px: torch.Tensor, py: torch.Tensor, weights: torch.Tensor,
    image_size: int, extent: float, size: float, kernel: SplatKernel,
) -> torch.Tensor:
    if kernel in ("gaussian", "cube"):
        return _separable_splat(px, py, weights, image_size, extent, size, kernel)
    if kernel == "ball":
        return _ball_splat(px, py, weights, image_size, extent, size)
    raise ValueError(f"unknown splat kernel {kernel!r}; choose gaussian/ball/cube")


def render_projection(
    positions: torch.Tensor,        # (N, 3) shared, or (B, N, 3) per-object
    weights: torch.Tensor,          # (N,) shared, or (B, N) per-object
    batch: int | None = None,       # number of fresh random poses to draw (R is None)
    n_tilts: int = 11,
    tilt_step: float = 12.0,
    tilt_axis: str = "y",
    image_size: int = 32,
    extent: float = 2.0,
    splat_kernel: SplatKernel = "gaussian",
    splat_size: float = 0.08,
    noise_std: float = 0.1,
    R: torch.Tensor | None = None,  # (B, 3, 3) known global poses, or None for fresh random
) -> torch.Tensor:
    """Render weighted 3D points to a CryoET tilt series, shape (B, K, P, P).

    ``positions``/``weights`` may be a single shared shape (2D/1D, e.g. the current
    guess ``X``) broadcast across ``B`` objects, or already batched per-object (e.g.
    a batch of distinct ground-truth objects). Exactly one of ``batch`` (draw ``B``
    fresh Haar-uniform global poses) or ``R`` (use these ``B`` known/learned poses)
    determines ``B``.
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)   # (1, N, 3)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)       # (1, N)
    device = positions.device

    if R is None:
        B = batch if batch is not None else positions.shape[0]
        mats = Rotation.random(B).as_matrix().astype(np.float32)
        R_global = torch.from_numpy(mats).to(device)
    else:
        R_global = R
        B = R_global.shape[0]

    x_glob = torch.matmul(positions, R_global.transpose(-1, -2))  # (B, N, 3)
    w = weights.expand(B, -1)                                     # (B, N)

    R_tilt = tilt_rotations(n_tilts, tilt_step, tilt_axis, device)  # (K, 3, 3)
    x_rot = torch.einsum("bnd,ked->bkne", x_glob, R_tilt)           # (B, K, N, 3)
    w_kt = w.unsqueeze(1).expand(-1, n_tilts, -1)                   # (B, K, N)

    img = _splat(
        x_rot[..., 0], x_rot[..., 1], w_kt, image_size, extent, splat_size, splat_kernel
    )  # (B, K, P, P)
    if noise_std > 0:
        img = img + noise_std * torch.randn_like(img)
    return img


# ── Soft space-carving back-projection: a shared "smart init" for both X params ──


def carve_occupancy(
    y_obs_single: torch.Tensor,  # (K, P, P) a single object's CryoET tilt series
    tilt_step: float,
    tilt_axis: str = "y",
    extent: float = 2.0,
    vol_size: int = 32,
    carve_quantile: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft space-carving lift of one K-tilt series into a per-voxel occupancy.

    Builds a ``vol_size^3`` occupancy grid in ``[-extent, extent]^3``: for each known
    tilt, rotate the voxel centers by ``R_tilt[n]``, orthographically project (drop
    z), and bilinearly sample that tilt's image. A voxel of the object must land
    *inside* the bright region in (nearly) every view, so the occupancy is a **soft
    space carve** -- each tilt is normalized and the occupancy is a low quantile over
    tilts (0.0 = strict intersection/min, 0.5 = median) -- a visual-hull
    reconstruction, not a blurry sum-back-projection.

    Returns ``(occ (M,), vox (M, 3))`` with ``M = vol_size**3``. Only the *known*
    tilt geometry is used (not the unknown global pose), so this lives in whatever
    frame the observed particle happens to be in -- a valid initial guess, since the
    optimization loop never observes the true frame either (handled at eval time via
    ICP alignment, see ``eval.py``).
    """
    y = y_obs_single.detach().cpu().float()
    K, P, _ = y.shape
    R_tilt = tilt_rotations(K, tilt_step, tilt_axis, "cpu")  # (K, 3, 3)

    lin = torch.linspace(-extent, extent, vol_size)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    vox = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (M, 3)

    vox_rot = torch.einsum("md,ked->kme", vox, R_tilt)  # (K, M, 3)
    uv = (vox_rot[..., :2] / extent).unsqueeze(1)        # (K, 1, M, 2)

    imgs = y.unsqueeze(1)  # (K, 1, P, P)
    sampled = F.grid_sample(
        imgs, uv, mode="bilinear", align_corners=True, padding_mode="zeros"
    )  # (K, 1, 1, M)
    s = sampled.view(K, -1).clamp(min=0)                     # (K, M)
    s = s / s.amax(dim=1, keepdim=True).clamp_min(1e-6)       # per-tilt normalize
    occ = torch.quantile(s, carve_quantile, dim=0)            # (M,)
    return occ, vox


def sample_points_from_occupancy(
    occ: torch.Tensor,      # (M,)
    vox: torch.Tensor,      # (M, 3)
    vol_size: int,
    extent: float,
    n_points: int,
    seed: int = 0,
) -> torch.Tensor:
    """Weighted-multinomial sample of `n_points` voxel centers, with sub-voxel jitter."""
    g = torch.Generator().manual_seed(seed)
    w = occ + 1e-6
    w = w / w.sum()
    idx = torch.multinomial(w, n_points, replacement=True, generator=g)
    spacing = (2.0 * extent) / max(vol_size - 1, 1)
    jitter = (torch.rand(n_points, 3, generator=g) - 0.5) * spacing
    return vox[idx] + jitter
