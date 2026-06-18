"""Corruption channel F for the lifted-SCSI inverse problem (point-cloud space).

The forward model mirrors the cryoEM channel used for voxels in
``toy_3d/scsi_train.py`` (random SO(3) pose -> parallel-beam projection -> AWGN),
but operates directly on a point cloud:

    1. rotate the cloud by a random SO(3) matrix,
    2. orthographically project to the xy-plane (drop z),
    3. "place a ball of radius r at every point" and accumulate it into a P x P
       image -- a sum over points of 2D Gaussians of std sigma = radius,
    4. add white Gaussian noise.

Because SCSI is trained EM-style (the M-step samples a pool under no_grad, the
E-step trains the velocity field on it), F only ever runs forward; it does not
need to be differentiable. It is nonetheless torch-native and differentiable.
"""
from __future__ import annotations

import numpy as np
import torch
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


def forward_channel(
    points: torch.Tensor,           # (B, N, 3)
    radius: float = 0.08,
    noise_std: float = 0.1,
    image_size: int = 32,
    extent: float = 2.0,
    R: torch.Tensor | None = None,  # (B, 3, 3) fixed pose, or None for fresh random
    so2: bool = False,
    so2_axis: str = "z",            # axis for the SO(2) pose ("x"/"y"/"z")
) -> torch.Tensor:
    """Render point clouds to noisy 2D projections. Returns (B, 1, P, P).

    A fresh random pose is drawn per cloud when ``R is None`` (the
    marginalize-over-pose setting used in the E-step); pass a fixed ``R`` to
    freeze the observations.

    The ball splat uses the separability of an isotropic 2D Gaussian:
    ``g(dx, dy) = gx(dx) * gy(dy)``, so the image is an einsum over the per-axis
    factors -- O(B*N*P) memory instead of O(B*N*P^2).
    """
    B, N, _ = points.shape
    device = points.device

    if R is None:
        if so2:
            R = random_so2(B, device, axis=so2_axis)
        else:
            R = random_so3(B, device)
    x_rot = torch.matmul(points, R.transpose(-1, -2))   # (B, N, 3)
    px = x_rot[..., 0]                                   # (B, N) -> image columns (x)
    py = x_rot[..., 1]                                   # (B, N) -> image rows (y)

    grid = torch.linspace(-extent, extent, image_size, device=device)  # (P,)
    sigma = max(float(radius), 1e-4)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    dx = grid[None, None, :] - px[..., None]   # (B, N, P)
    dy = grid[None, None, :] - py[..., None]   # (B, N, P)
    gx = torch.exp(-dx * dx * inv_2s2)         # (B, N, P)  along x (columns)
    gy = torch.exp(-dy * dy * inv_2s2)         # (B, N, P)  along y (rows)

    img = torch.einsum("bni,bnj->bij", gy, gx)  # (B, P, P)  [row=y, col=x]
    img = img.unsqueeze(1)                       # (B, 1, P, P)

    if noise_std > 0:
        img = img + noise_std * torch.randn_like(img)
    return img


def backproject_bootstrap(
    y_obs: torch.Tensor,    # (Nobj, 1, P, P)  fixed observations
    n_points: int,
    extent: float = 2.0,
    z_extent: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Lift each 2D observation into a "puffed silhouette" point cloud for pi(0).

    For each observation, sample ``n_points`` pixels with probability
    proportional to (clamped) intensity, map them to world (x, y), and assign a
    random depth ``z in [-z_extent, z_extent]``. Analogous to the voxel
    revolve/tile bootstrap. Returns (Nobj, n_points, 3) on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    y = y_obs.detach().cpu()
    Nobj, _, P, _ = y.shape
    grid = torch.linspace(-extent, extent, P)          # (P,)
    spacing = (2.0 * extent) / max(P - 1, 1)
    clouds = []
    for i in range(Nobj):
        w = y[i, 0].clamp(min=0).flatten() + 1e-6      # (P*P,)
        w = w / w.sum()
        idx = torch.multinomial(w, n_points, replacement=True, generator=g)  # (n_points,)
        row = torch.div(idx, P, rounding_mode="floor")
        col = idx % P
        x = grid[col] + (torch.rand(n_points, generator=g) - 0.5) * spacing
        yy = grid[row] + (torch.rand(n_points, generator=g) - 0.5) * spacing
        z = (torch.rand(n_points, generator=g) - 0.5) * 2.0 * z_extent
        clouds.append(torch.stack([x, yy, z], dim=-1))
    return torch.stack(clouds, dim=0)                   # (Nobj, n_points, 3)
