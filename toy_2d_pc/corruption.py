"""CryoEM forward channel ``F`` (2D image -> 1D signal) and its pseudo-inverse ``F``-dagger.

A 2D object is a point cloud ``X in R^{N x 2}``; :mod:`renderer` turns it into a ``(P, P)``
density image in its own canonical frame. The CryoEM channel renders a **tilt series** of 1D
projections of that image:

    F(X) = { sum_rows( Rot(theta + n . d-theta) . G(X) ) + Z }_{n=1..K}   ->  (B, K, P)

  * ``G``           renders the point cloud to a ``(P, P)`` image (:mod:`renderer`; gaussian /
                    disk / histogram).
  * ``Rot(theta)``  rotates the *image* by angle ``theta`` (bilinear, via
                    ``affine_grid``/``grid_sample``) -- one *unknown* Haar-uniform global SO(2)
                    pose per cloud (the nuisance).
  * ``d-theta``     the ``K`` *known* tilt increments (centered at 0, step ``tilt_step``).
  * ``sum_rows``    collapses the rotated image along the row (y) axis, leaving a length-P
                    projection along the column (x, "detector") axis -- a rotate-then-sum Radon
                    transform.
  * ``Z``           white Gaussian signal noise, drawn independently per projection.

SO(2) is **abelian** (unlike 3D's SO(3)), so the unknown global pose and each known tilt
compose into a *single* combined rotation angle per tilt -- one image rotation per tilt,
not two composed rotation stages as in ``toy_3d_pc``.

``F`` only ever runs forward in SCSI (no gradient needed). ``pseudo_inverse`` (==
:func:`backproject_tomo`) is ``F``-dagger: a space-carving back-projection of the K tilts into
a point cloud, used for the warm-start (Algorithm 1) and pi(0). Back-projection only uses the
*known* tilt geometry -- the residual global pose is left for EM to resolve.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .renderer import render as _render

# Module-level caches for the (fixed) tilt-angle rotation matrices / angles, keyed by
# (n_tilts, tilt_step_deg, device_str). Avoids repeated recomputation in the training hot-loop.
_tilt_angle_cache: dict[tuple, torch.Tensor] = {}


# ── Rotations (SO(2)) ───────────────────────────────────────────────────────────


def _angles_to_matrices(theta: torch.Tensor) -> torch.Tensor:
    """theta: (...,) radians -> (..., 2, 2) SO(2) rotation matrices."""
    c, s = torch.cos(theta), torch.sin(theta)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def random_rotations(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``n`` Haar-uniform SO(2) rotation matrices, shape (n, 2, 2)."""
    theta = torch.rand(n, device=device) * (2.0 * math.pi)
    return _angles_to_matrices(theta)


def rotate_clouds(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply per-cloud rotations. points (B, N, 2), R (B, 2, 2) -> (B, N, 2).

    Point-level rotation utility -- not used by :func:`forward_channel` itself (which rotates
    the *rendered image*), but used by ``warmstart``'s ``g . F-dagger(y)`` pose augmentation.
    """
    return torch.matmul(points, R.transpose(-1, -2))


def tilt_angles(
    n_tilts: int, tilt_step_deg: float, device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Known tilt-series rotation angles, shape (K,), radians.

    The K tilt increments are symmetric about 0, ``(arange(K) - (K-1)/2) * tilt_step`` degrees.
    Unlike ``toy_3d_pc``'s ``tilt_rotations`` there is no axis parameter -- 2D has only one
    rotation plane.
    """
    n = torch.arange(n_tilts, device=device, dtype=torch.float32) - (n_tilts - 1) / 2.0
    return torch.deg2rad(n * tilt_step_deg)  # (K,)


# ── Image rotation (batched, for the Radon-style projection) ───────────────────


def _rotate_images(img: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate each image by each of K angles. img (B, P, P), angles (B, K) -> (B, K, P, P).

    Bilinear resample via ``affine_grid``/``grid_sample``, zero-padded outside the frame.
    """
    B, P, _ = img.shape
    K = angles.shape[1]
    imgs_exp = img.unsqueeze(1).expand(B, K, P, P).reshape(B * K, 1, P, P)
    theta = angles.reshape(-1)                                     # (B*K,)
    c, s = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(c)
    affine = torch.stack([
        torch.stack([c, -s, zeros], dim=-1),
        torch.stack([s, c, zeros], dim=-1),
    ], dim=-2)                                                     # (B*K, 2, 3)
    grid = F.affine_grid(affine, imgs_exp.shape, align_corners=True)
    rotated = F.grid_sample(
        imgs_exp, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    return rotated.reshape(B, K, P, P)


# ── F: CryoEM forward channel ────────────────────────────────────────────────────


def forward_channel(
    points: torch.Tensor,           # (B, N, 2)
    radius: float = 0.08,
    noise_std: float = 0.1,         # Z: white Gaussian signal noise
    image_size: int = 64,
    extent: float = 2.0,
    theta: torch.Tensor | None = None,  # (B,) fixed global pose angle, or None for fresh random
    coord_noise_std: float = 0.0,   # W: AWGN on point coords before rendering
    n_tilts: int = 11,
    tilt_step: float = 12.0,
    splat: str = "gaussian",        # blob kernel G: "gaussian", "disk", or "histogram"
) -> torch.Tensor:
    """Render point clouds to a CryoEM tilt-series sinogram, shape (B, K, P)."""
    B, N, _ = points.shape
    device = points.device

    # W: AWGN on coordinates (iid per coord per particle), before rendering.
    if coord_noise_std > 0:
        points = points + coord_noise_std * torch.randn_like(points)

    # G: render the object in its own canonical (unrotated) frame.
    img = _render(points, image_size, extent, radius, kind=splat)  # (B, P, P)

    # One global SO(2) pose per cloud (fresh unless a fixed theta is supplied).
    theta_global = (
        torch.rand(B, device=device) * (2.0 * math.pi) if theta is None else theta
    )

    # K known tilts: cached per (n_tilts, tilt_step, device).
    cache_key = (n_tilts, tilt_step, str(device))
    if cache_key not in _tilt_angle_cache:
        _tilt_angle_cache[cache_key] = tilt_angles(n_tilts, tilt_step, device)
    dtheta = _tilt_angle_cache[cache_key]                          # (K,)

    # SO(2) is abelian: global pose and tilt compose into one combined angle per tilt.
    combined = theta_global[:, None] + dtheta[None, :]             # (B, K)

    rotated = _rotate_images(img, combined)                        # (B, K, P, P)
    sino = rotated.sum(dim=-2)                                     # collapse rows (y) -> (B, K, P)

    if noise_std > 0:
        sino = sino + noise_std * torch.randn_like(sino)           # Z, indep per tilt
    return sino


# ── F-dagger: pseudo-inverse (space-carving back-projection) ────────────────────


def backproject_tomo(
    y_obs: torch.Tensor,    # (Nobj, K, P)  fixed CryoEM sinograms
    n_points: int,
    tilt_step: float,
    extent: float = 2.0,
    vol_size: int = 64,
    carve_quantile: float = 0.15,
    seed: int = 0,
) -> torch.Tensor:
    """Space-carving lift of a K-tilt sinogram into a point cloud (the pseudo-inverse F-dagger).

    Builds a ``vol_size^2`` occupancy grid in ``[-extent, extent]^2``: for each known tilt,
    rotate the grid points and bilinearly sample that tilt's 1D signal at the projected
    coordinate. A point of the object must land inside the bright region in (nearly) every
    view, so the occupancy is a **soft space carve** -- each tilt is normalized and the
    occupancy is a low quantile over tilts (``carve_quantile``; 0.0 = strict intersection/min,
    0.5 = median). Then sample ``n_points`` grid cells with probability proportional to
    occupancy, with sub-cell jitter. Only the *known* tilt geometry is used, so the
    reconstruction lives in the lab frame; the residual global pose is left for EM to resolve.
    Returns (Nobj, n_points, 2) on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    y = y_obs.detach().cpu().float()
    Nobj, K, P = y.shape
    dtheta = tilt_angles(K, tilt_step, "cpu")                       # (K,)
    R_tilt = _angles_to_matrices(dtheta)                            # (K, 2, 2)

    lin = torch.linspace(-extent, extent, vol_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    vox = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)     # (M, 2)
    M = vox.shape[0]
    spacing = (2.0 * extent) / max(vol_size - 1, 1)

    vox_rot = torch.einsum("md,ked->kme", vox, R_tilt)              # (K, M, 2)
    u = (vox_rot[..., 0] / extent).unsqueeze(1)                     # (K, 1, M)
    v = torch.zeros_like(u)
    uv = torch.stack([u, v], dim=-1)                                # (K, 1, M, 2) [x, y]

    clouds = []
    for i in range(Nobj):
        imgs = y[i].unsqueeze(1).unsqueeze(1)                       # (K, 1, 1, P) 1-row image
        sampled = F.grid_sample(
            imgs, uv, mode="bilinear", align_corners=True, padding_mode="zeros"
        )                                                          # (K, 1, 1, M)
        s = sampled.view(K, M).clamp(min=0)                        # (K, M)
        s = s / s.amax(dim=1, keepdim=True).clamp_min(1e-6)        # per-tilt normalize
        occ = torch.quantile(s, carve_quantile, dim=0)             # (M,) soft space carve
        w = occ + 1e-6
        w = w / w.sum()
        idx = torch.multinomial(w, n_points, replacement=True, generator=g)
        jitter = (torch.rand(n_points, 2, generator=g) - 0.5) * spacing
        clouds.append(vox[idx] + jitter)
    return torch.stack(clouds, dim=0)                               # (Nobj, n_points, 2)


# F-dagger public alias.
pseudo_inverse = backproject_tomo
