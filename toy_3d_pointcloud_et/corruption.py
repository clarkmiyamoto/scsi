"""CryoET forward channel F for the lifted-SCSI inverse problem (point-cloud space).

Models a CryoET tilt series: one *unknown* global SO(3) orientation ``R(theta)``
(the nuisance), composed under ``K`` *known* tilt rotations
``R_tilt(n * Delta-theta)`` about a fixed lab axis, giving K projections

    F(X) = { P R_tilt(n*Delta-theta) R(theta) (X + W) + Z }_{n=1..K}  ->  (B, K, P, P)

with W applied once per cloud (same particle across tilts) and Z drawn independently
per projection.

Because SCSI is trained EM-style (the M-step samples a pool under no_grad, the
E-step trains the velocity field on it), F only ever runs forward; it does not
need to be differentiable. It is nonetheless torch-native and differentiable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def tilt_rotations(
    n_tilts: int,
    tilt_step_deg: float,
    axis: str = "y",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rotation matrices for a single-axis CryoET tilt series, shape (K, 3, 3).

    The K *known* tilt increments are symmetric about 0,
    ``((arange(K) - (K-1)/2) * tilt_step_deg)`` degrees about ``axis`` ("x"/"y";
    "z" is in-plane and adds no information under the drop-z projection). With a full
    SO(3) global pose, a constant tilt offset is absorbed into it, so centring the
    series is equivalent to the literal ``n * Delta-theta`` schedule.
    """
    n = np.arange(n_tilts) - (n_tilts - 1) / 2.0
    angles = (n * tilt_step_deg)[:, None]  # (K, 1) degrees
    mats = Rotation.from_euler(axis, angles, degrees=True).as_matrix().astype(np.float32)
    return torch.from_numpy(mats).to(device)  # (K, 3, 3)


def _gaussian_splat(
    px: torch.Tensor, py: torch.Tensor, image_size: int, extent: float, radius: float
) -> torch.Tensor:
    """Separable isotropic-Gaussian ball splat. ``px``/``py``: (..., N) -> (..., P, P).

    ``g(dx, dy) = gx(dx) * gy(dy)`` so the image is an einsum over the per-axis
    factors -- O(prod(...) * N * P) memory instead of O(... * N * P^2). The leading
    dims are arbitrary (e.g. (B,) for one projection or (B, K) for a tilt series).
    """
    device = px.device
    grid = torch.linspace(-extent, extent, image_size, device=device)  # (P,)
    sigma = max(float(radius), 1e-4)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    dx = grid - px[..., None]   # (..., N, P)
    dy = grid - py[..., None]   # (..., N, P)
    gx = torch.exp(-dx * dx * inv_2s2)         # (..., N, P)  along x (columns)
    gy = torch.exp(-dy * dy * inv_2s2)         # (..., N, P)  along y (rows)
    return torch.einsum("...ni,...nj->...ij", gy, gx)  # (..., P, P)  [row=y, col=x]


def forward_channel(
    points: torch.Tensor,           # (B, N, 3)
    radius: float = 0.08,
    noise_std: float = 0.1,         # Z: white Gaussian image noise
    image_size: int = 32,
    extent: float = 2.0,
    R: torch.Tensor | None = None,  # (B, 3, 3) fixed global pose, or None for fresh random
    coord_noise_std: float = 0.0,   # W: AWGN on point coordinates before rotation
    n_tilts: int = 11,              # number of projections K in the tilt series
    tilt_step: float = 12.0,        # degrees between consecutive tilts
    tilt_axis: str = "y",           # tilt axis ("x"/"y")
) -> torch.Tensor:
    """Render point clouds to a CryoET tilt series, shape (B, K, P, P).

    One global SO(3) pose ``R(theta)`` per cloud (fresh Haar-uniform unless ``R`` is
    given) composed under the K known tilts (see :func:`tilt_rotations`). Image noise
    ``Z`` is independent per tilt. Coordinate noise ``W`` is applied once per cloud
    before the global rotation.
    """
    B, N, _ = points.shape
    device = points.device

    # W: AWGN on coordinates (iid per coordinate per particle), before the rotation.
    if coord_noise_std > 0:
        points = points + coord_noise_std * torch.randn_like(points)

    # One global SO(3) pose per cloud (fresh unless a fixed R is supplied).
    if R is None:
        mats = Rotation.random(B).as_matrix().astype(np.float32)
        R_global = torch.from_numpy(mats).to(device)
    else:
        R_global = R
    x_glob = torch.matmul(points, R_global.transpose(-1, -2))     # (B, N, 3)

    # K known tilts: x_rot[b,k,n] = R_tilt[k] @ x_glob[b,n].
    R_tilt = tilt_rotations(n_tilts, tilt_step, tilt_axis, device)  # (K, 3, 3)
    x_rot = torch.einsum("bnd,ked->bkne", x_glob, R_tilt)         # (B, K, N, 3)
    img = _gaussian_splat(
        x_rot[..., 0], x_rot[..., 1], image_size, extent, radius
    )                                                            # (B, K, P, P)
    if noise_std > 0:
        img = img + noise_std * torch.randn_like(img)            # Z, indep per tilt
    return img


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
    """Space-carving lift of a K-tilt series into a point cloud for pi(0).

    Builds a ``vol_size^3`` occupancy grid in ``[-extent, extent]^3``: for each known
    tilt, rotate the voxel centres by ``R_tilt[n]``, orthographically project (drop
    z), and bilinearly sample that tilt's image. A voxel of the object must land
    *inside* the bright region in (nearly) every view, so the occupancy is a **soft
    space carve** -- each tilt is normalised and the occupancy is a low quantile over
    tilts (``carve_quantile``; 0.0 = strict intersection / min, 0.5 = median). This is
    a visual-hull reconstruction, NOT a sum: an unfiltered sum-back-projection would
    smear credit along every ray and yield a blurry blob. Then sample ``n_points``
    voxels with probability proportional to the occupancy, with sub-voxel jitter.

    Only the *known* tilt geometry is used (not the unknown global ``theta``), so the
    reconstruction lives in the lab frame (``= R(theta) . x_canonical``) -- a valid
    clean sample whose residual orientation the EM loop resolves. The number of tilts
    K is read from ``y_obs``. Returns (Nobj, n_points, 3) on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    y = y_obs.detach().cpu().float()
    Nobj, K, P, _ = y.shape
    R_tilt = tilt_rotations(K, tilt_step, tilt_axis, "cpu")          # (K, 3, 3)

    # Voxel centres of the reconstruction cube, as (x, y, z).
    lin = torch.linspace(-extent, extent, vol_size)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    vox = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (M, 3)
    M = vox.shape[0]
    spacing = (2.0 * extent) / max(vol_size - 1, 1)

    # Project every voxel under every tilt -> grid_sample coords in [-1, 1].
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
        occ = torch.quantile(s, carve_quantile, dim=0)            # (M,) soft space carving
        w = occ + 1e-6
        w = w / w.sum()
        idx = torch.multinomial(w, n_points, replacement=True, generator=g)
        jitter = (torch.rand(n_points, 3, generator=g) - 0.5) * spacing
        clouds.append(vox[idx] + jitter)
    return torch.stack(clouds, dim=0)                               # (Nobj, n_points, 3)
