"""CryoET forward channel ``F`` (point-cloud representation) and its pseudo-inverse ``F``-dagger.

A 3D object is a set of ``N`` points ``X in R^{N x 3}``. The CryoET channel renders a
**tilt series**:

    F(X) = { P . R_tilt(n . d-theta) . R(theta) . (G o (X + W)) + Z }_{n=1..K}   ->  (B, K, P, P)

  * ``G``       places a 3D blob at every point: an isotropic **Gaussian** (default) or a
                **solid/filled ball** (the analytic projection of a uniform ball -- a filled
                disk, NOT a shell). Projection ``P`` (drop z) is baked into the 2D splat.
  * ``R(theta)``    one *unknown* Haar-uniform global SO(3) pose per cloud (the nuisance).
  * ``R_tilt``  the ``K`` *known* single-axis tilt increments (centered at 0, step ``tilt_step``).
  * ``W``       AWGN on point coordinates, applied once per cloud before rotation.
  * ``Z``       white Gaussian image noise, drawn independently per projection.

``F`` only ever runs forward in SCSI (no gradient needed) but is torch-native/differentiable.
``pseudo_inverse`` (== :func:`backproject_tomo`) is ``F``-dagger: a space-carving back-projection
of the K tilts into a point cloud, used for the warm-start (Algorithm 1) and pi(0).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

# Module-level cache for the (fixed) tilt-series rotation matrices, keyed by
# (n_tilts, tilt_step_deg, axis, device_str).  Avoids repeated scipy + numpy
# round-trips inside the training hot-loop.
_R_tilt_cache: dict[tuple, torch.Tensor] = {}


# ── Rotations ─────────────────────────────────────────────────────────────────


def random_rotations(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``n`` Haar-uniform SO(3) rotation matrices, shape (n, 3, 3).

    On CUDA, generates natively on the device (avoids CPU↔GPU transfer).
    On CPU/MPS, scipy is faster for the batch sizes used in training.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        # GPU-native via unit quaternions -- no CPU round-trip.
        q = torch.randn(n, 4, device=dev)
        q = q / q.norm(dim=1, keepdim=True)
        w, x, y, z = q.unbind(1)
        return torch.stack([
            1 - 2 * (y * y + z * z),   2 * (x * y - z * w),   2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z),   2 * (y * z - x * w),
                2 * (x * z - y * w),   2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], dim=1).reshape(n, 3, 3)
    mats = Rotation.random(n).as_matrix().astype(np.float32)
    return torch.from_numpy(mats).to(dev)


def rotate_clouds(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply per-cloud rotations. points (B, N, 3), R (B, 3, 3) -> (B, N, 3)."""
    return torch.matmul(points, R.transpose(-1, -2))


def tilt_rotations(
    n_tilts: int,
    tilt_step_deg: float,
    axis: str = "y",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Known single-axis CryoET tilt-series rotations, shape (K, 3, 3).

    The K tilt increments are symmetric about 0, ``(arange(K) - (K-1)/2) * tilt_step``
    degrees about ``axis`` ("x"/"y"; "z" is in-plane and degenerate under drop-z
    projection). With a full SO(3) global pose a constant offset is absorbed, so
    centering the series is equivalent to the literal ``n . d-theta`` schedule.
    """
    n = np.arange(n_tilts) - (n_tilts - 1) / 2.0
    angles = (n * tilt_step_deg)[:, None]  # (K, 1) degrees
    mats = Rotation.from_euler(axis, angles, degrees=True).as_matrix().astype(np.float32)
    return torch.from_numpy(mats).to(device)  # (K, 3, 3)


# ── G: render points to a 2D image (projection P baked in) ────────────────────


def _gaussian_splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float, radius: float
) -> torch.Tensor:
    """Separable isotropic-Gaussian splat. px/py: (..., N) -> (..., P, P).

    ``g(dx, dy) = gx(dx) * gy(dy)`` so the image is an einsum over per-axis factors --
    O(prod(...) * N * P) memory. Leading dims are arbitrary ((B,) or (B, K)).
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


def _ball_splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float, radius: float,
    chunk: int = 64,
) -> torch.Tensor:
    """Solid/filled-ball splat. px/py: (..., N) -> (..., P, P).

    The orthographic projection of a *uniform solid ball* of radius ``r`` is a filled
    disk whose intensity is the chord length through the ball, ``~ sqrt(r^2 - d^2)`` for
    in-plane distance ``d <= r`` (peak-normalized here to 1 at the center, 0 at the rim).
    This is a *full* ball, not a shell. Unlike the Gaussian it is not separable, so the
    (..., N, P, P) work tensor is built in chunks over N to bound memory.
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


def _splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float, radius: float,
    kind: str = "gaussian",
) -> torch.Tensor:
    if kind == "gaussian":
        return _gaussian_splat(px, py, image_size, extent, radius)
    if kind == "ball":
        return _ball_splat(px, py, image_size, extent, radius)
    raise ValueError(f"unknown splat kind {kind!r}; choose 'gaussian' or 'ball'")


# ── F: CryoET forward channel ─────────────────────────────────────────────────


def forward_channel(
    points: torch.Tensor,           # (B, N, 3)
    radius: float = 0.08,
    noise_std: float = 0.1,         # Z: white Gaussian image noise
    image_size: int = 32,
    extent: float = 2.0,
    R: torch.Tensor | None = None,  # (B, 3, 3) fixed global pose, or None for fresh random
    coord_noise_std: float = 0.0,   # W: AWGN on point coords before rotation
    n_tilts: int = 11,              # number of projections K in the tilt series
    tilt_step: float = 12.0,        # degrees between consecutive tilts
    tilt_axis: str = "y",           # tilt axis ("x"/"y")
    splat: str = "gaussian",        # blob kernel G: "gaussian" or "ball" (solid/filled)
) -> torch.Tensor:
    """Render point clouds to a CryoET tilt series, shape (B, K, P, P)."""
    B, N, _ = points.shape
    device = points.device

    # W: AWGN on coordinates (iid per coord per particle), before the rotation.
    if coord_noise_std > 0:
        points = points + coord_noise_std * torch.randn_like(points)

    # One global SO(3) pose per cloud (fresh unless a fixed R is supplied).
    R_global = random_rotations(B, device) if R is None else R
    x_glob = rotate_clouds(points, R_global)                       # (B, N, 3)

    # K known tilts: cached per (n_tilts, tilt_step, axis, device).
    cache_key = (n_tilts, tilt_step, tilt_axis, str(device))
    if cache_key not in _R_tilt_cache:
        _R_tilt_cache[cache_key] = tilt_rotations(n_tilts, tilt_step, tilt_axis, device)
    R_tilt = _R_tilt_cache[cache_key]
    x_rot = torch.einsum("bnd,ked->bkne", x_glob, R_tilt)         # (B, K, N, 3)
    img = _splat(x_rot[..., 0], x_rot[..., 1], image_size, extent, radius, kind=splat)  # (B,K,P,P)

    if noise_std > 0:
        img = img + noise_std * torch.randn_like(img)             # Z, indep per tilt
    return img


# ── F-dagger: pseudo-inverse (space-carving tomographic back-projection) ───────


def backproject_tomo(
    y_obs: torch.Tensor,    # (Nobj, K, P, P)  fixed CryoET tilt series
    n_points: int,
    tilt_step: float,
    tilt_axis: str = "y",
    extent: float = 2.0,
    vol_size: int = 48,
    carve_quantile: float = 0.15,
    seed: int = 0,
) -> torch.Tensor:
    """Space-carving lift of a K-tilt series into a point cloud (the pseudo-inverse F-dagger).

    Builds a ``vol_size^3`` occupancy grid in ``[-extent, extent]^3``: for each known
    tilt, rotate the voxel centers, orthographically project (drop z) and bilinearly
    sample that tilt's image. A voxel of the object must land *inside* the bright region
    in (nearly) every view, so the occupancy is a **soft space carve** -- each tilt is
    normalized and the occupancy is a low quantile over tilts (``carve_quantile``;
    0.0 = strict intersection/min, 0.5 = median). This is a visual-hull reconstruction,
    NOT a sum (an unfiltered sum-back-projection would smear into a blurry blob). Then
    sample ``n_points`` voxels with probability proportional to occupancy, with sub-voxel
    jitter. Only the *known* tilt geometry is used, so the reconstruction lives in the lab
    frame (= R(theta) . x_canonical); the residual global pose is left for EM to resolve.
    Returns (Nobj, n_points, 3) on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    y = y_obs.detach().cpu().float()
    Nobj, K, P, _ = y.shape
    R_tilt = tilt_rotations(K, tilt_step, tilt_axis, "cpu")          # (K, 3, 3)

    lin = torch.linspace(-extent, extent, vol_size)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    vox = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (M, 3)
    M = vox.shape[0]
    spacing = (2.0 * extent) / max(vol_size - 1, 1)

    vox_rot = torch.einsum("md,ked->kme", vox, R_tilt)              # (K, M, 3)
    uv = (vox_rot[..., :2] / extent).unsqueeze(1)                   # (K, 1, M, 2) [x, y]

    clouds = []
    for i in range(Nobj):
        imgs = y[i].unsqueeze(1)                                    # (K, 1, P, P)
        sampled = F.grid_sample(
            imgs, uv, mode="bilinear", align_corners=True, padding_mode="zeros"
        )                                                          # (K, 1, 1, M)
        s = sampled.view(K, M).clamp(min=0)                        # (K, M)
        s = s / s.amax(dim=1, keepdim=True).clamp_min(1e-6)        # per-tilt normalize
        occ = torch.quantile(s, carve_quantile, dim=0)            # (M,) soft space carve
        w = occ + 1e-6
        w = w / w.sum()
        idx = torch.multinomial(w, n_points, replacement=True, generator=g)
        jitter = (torch.rand(n_points, 3, generator=g) - 0.5) * spacing
        clouds.append(vox[idx] + jitter)
    return torch.stack(clouds, dim=0)                               # (Nobj, n_points, 3)


# F-dagger public alias.
pseudo_inverse = backproject_tomo
