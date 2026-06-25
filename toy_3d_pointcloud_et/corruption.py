"""Corruption channel F for the lifted-SCSI inverse problem (point-cloud space).

The forward model mirrors the cryoEM channel used for voxels in
``toy_3d/scsi_train.py`` (random SO(3) pose -> parallel-beam projection -> AWGN),
but operates directly on a point cloud. In operator form it is

    F(X) = P G (X + W) + Z

applied per cloud:

    0. add coordinate noise W ~ N(0, coord_noise_std^2) to every point (iid per
       coordinate per particle),
    1. rotate the cloud by a random rotation G,
    2. orthographically project P to the xy-plane (drop z),
    3. "place a ball of radius r at every point" and accumulate it into a P x P
       image -- a sum over points of 2D Gaussians of std sigma = radius,
    4. add white Gaussian image noise Z ~ N(0, noise_std^2).

The rotation G is selectable via ``channel``: ``"so3"`` (full random pose),
``"so2"`` (single-axis pose, see ``so2_axis``), or ``"cryoet"`` (a tilt series, see
below). ``coord_noise_std`` (W) defaults to 0, so the rotation-only channels reduce
to ``P G X + Z``.

The ``"cryoet"`` channel models a CryoET tilt series: one *unknown* global SO(3)
orientation ``R(theta)`` (the nuisance), composed under ``K`` *known* tilt rotations
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


def random_so3(batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Haar-uniform SO(3) rotation matrices, shape (batch, 3, 3)."""
    mats = Rotation.random(batch).as_matrix().astype(np.float32)  # (batch, 3, 3)
    return torch.from_numpy(mats).to(device)

def random_so2(
    batch: int, device: torch.device | str = "cpu", axis: str = "z"
) -> torch.Tensor:
    """Haar-uniform SO(2) rotation matrices about a single ``axis`` ("x"/"y"/"z"),
    shape (batch, 3, 3).

    Because ``forward_channel`` projects by dropping z, ``axis="z"`` is an in-plane
    rotation of the 2D image while "x"/"y" tilt the object out of plane.
    """
    # Sample uniform angles between 0 and 2*pi. Shape (batch, 1) so SciPy reads each
    # row as one single-axis rotation rather than one rotation of `batch` angles.
    angles = np.random.uniform(0, 2 * np.pi, (batch, 1))

    # Generate rotations around the chosen axis
    mats = Rotation.from_euler(axis, angles).as_matrix().astype(np.float32)  # (batch, 3, 3)

    return torch.from_numpy(mats).to(device)


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
    channel: str = "so3",           # rotation G: "so3" | "so2" | "cryoet"
    so2_axis: str = "z",            # axis for the SO(2) pose ("x"/"y"/"z")
    coord_noise_std: float = 0.0,   # W: AWGN on point coordinates before rotation
    n_tilts: int = 11,              # [cryoet] number of projections K in the tilt series
    tilt_step: float = 12.0,        # [cryoet] degrees between consecutive tilts
    tilt_axis: str = "y",           # [cryoet] tilt axis ("x"/"y")
) -> torch.Tensor:
    """Render point clouds to noisy 2D projections.

    For ``channel in {"so3", "so2"}`` this is ``F(X) = P G (X + W) + Z`` and returns
    (B, 1, P, P): a fresh random pose ``G`` is drawn per cloud when ``R is None`` (the
    marginalize-over-pose E-step setting); pass a fixed ``R`` to freeze it.

    For ``channel == "cryoet"`` it is a tilt series
    ``F(X) = { P R_tilt(n*Delta-theta) R(theta) (X + W) + Z }_{n=1..K}`` and returns
    (B, K, P, P): one global SO(3) pose ``R(theta)`` per cloud (fresh unless ``R`` is
    given) composed under the K known tilts (see :func:`tilt_rotations`). The image
    noise ``Z`` is independent per projection.

    The coordinate noise ``W`` (std ``coord_noise_std``, iid per coordinate per
    particle) is added once *before* the rotation; ``Z`` (std ``noise_std``) is added
    to the rendered image(s).
    """
    B, N, _ = points.shape
    device = points.device

    # W: AWGN on coordinates (iid per coordinate per particle), before the rotation.
    if coord_noise_std > 0:
        points = points + coord_noise_std * torch.randn_like(points)

    if channel == "cryoet":
        # One global SO(3) pose per cloud (fresh unless a fixed R is supplied) ...
        R_global = random_so3(B, device) if R is None else R
        x_glob = torch.matmul(points, R_global.transpose(-1, -2))     # (B, N, 3)
        # ... then the K known tilts: x_rot[b,k,n] = R_tilt[k] @ x_glob[b,n].
        R_tilt = tilt_rotations(n_tilts, tilt_step, tilt_axis, device)  # (K, 3, 3)
        x_rot = torch.einsum("bnd,ked->bkne", x_glob, R_tilt)         # (B, K, N, 3)
        img = _gaussian_splat(
            x_rot[..., 0], x_rot[..., 1], image_size, extent, radius
        )                                                            # (B, K, P, P)
        if noise_std > 0:
            img = img + noise_std * torch.randn_like(img)            # Z, indep per tilt
        return img

    # ── single-projection so3 / so2 ────────────────────────────────────────────
    # G: random rotation (fresh per cloud unless a fixed R is supplied).
    if R is None:
        R = (
            random_so2(B, device, axis=so2_axis)
            if channel == "so2"
            else random_so3(B, device)
        )
    x_rot = torch.matmul(points, R.transpose(-1, -2))   # (B, N, 3)
    img = _gaussian_splat(
        x_rot[..., 0], x_rot[..., 1], image_size, extent, radius
    ).unsqueeze(1)                                       # (B, 1, P, P)

    # Z: white Gaussian image noise.
    if noise_std > 0:
        img = img + noise_std * torch.randn_like(img)
    return img


def backproject_bootstrap(
    y_obs: torch.Tensor,    # (Nobj, C, P, P)  fixed observations
    n_points: int,
    extent: float = 2.0,
    z_extent: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Lift each 2D observation into a "puffed silhouette" point cloud for pi(0).

    For each observation, sample ``n_points`` pixels with probability
    proportional to (clamped) intensity, map them to world (x, y), and assign a
    random depth ``z in [-z_extent, z_extent]``. Analogous to the voxel
    revolve/tile bootstrap. For a K-tilt CryoET series the central tilt is used as
    the silhouette. Returns (Nobj, n_points, 3) on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    y = y_obs.detach().cpu()
    Nobj, C, P, _ = y.shape
    ch = C // 2  # central tilt for a K-tilt series; 0 for a single projection
    grid = torch.linspace(-extent, extent, P)          # (P,)
    spacing = (2.0 * extent) / max(P - 1, 1)
    clouds = []
    for i in range(Nobj):
        w = y[i, ch].clamp(min=0).flatten() + 1e-6     # (P*P,)
        w = w / w.sum()
        idx = torch.multinomial(w, n_points, replacement=True, generator=g)  # (n_points,)
        row = torch.div(idx, P, rounding_mode="floor")
        col = idx % P
        x = grid[col] + (torch.rand(n_points, generator=g) - 0.5) * spacing
        yy = grid[row] + (torch.rand(n_points, generator=g) - 0.5) * spacing
        z = (torch.rand(n_points, generator=g) - 0.5) * 2.0 * z_extent
        clouds.append(torch.stack([x, yy, z], dim=-1))
    return torch.stack(clouds, dim=0)                   # (Nobj, n_points, 3)


def backproject_tomo(
    y_obs: torch.Tensor,    # (Nobj, K, P, P)  fixed CryoET tilt series
    n_points: int,
    tilt_step: float,
    tilt_axis: str = "y",
    extent: float = 2.0,
    vol_size: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Tomographic back-projection lift of a K-tilt series into a point cloud for pi(0).

    Builds a ``vol_size^3`` occupancy grid in ``[-extent, extent]^3``: for each known
    tilt, rotate the voxel centres by ``R_tilt[n]``, orthographically project (drop
    z), and bilinearly sample that tilt's image; sum the back-projected intensity over
    tilts. Then sample ``n_points`` voxels with probability proportional to the
    (clamped) occupancy, with sub-voxel jitter.

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
        occ = sampled.view(K, M).clamp(min=0).sum(dim=0)           # (M,)
        w = occ + 1e-6
        w = w / w.sum()
        idx = torch.multinomial(w, n_points, replacement=True, generator=g)
        jitter = (torch.rand(n_points, 3, generator=g) - 0.5) * spacing
        clouds.append(vox[idx] + jitter)
    return torch.stack(clouds, dim=0)                               # (Nobj, n_points, 3)
