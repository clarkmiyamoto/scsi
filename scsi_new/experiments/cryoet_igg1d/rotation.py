"""
SO(3) sampling and volume posing for the IgG-1D channel.

Posing follows CryoBench's convention (cryobench_data/igg1d.py::rotate_volume), so volumes produced
here live in the same frame as IgG1DDataset(volume_frame="image") targets: out(p) = x(R^T (p - s)),
with p the (x, y, z) voxel coordinate relative to index D//2, trilinear interpolation and zero
padding. Summing `out` over z (dim -3) is then the projection of x at pose R, shifted in-plane by s.
Unlike cryoet_mnist3d/rotation.py there is no [-1, 1] shift trick: density background is 0, which
is exactly grid_sample's zero padding.
"""

import torch
import torch.nn.functional as F


def sample_uniform_rotation_so3(B: int, device: torch.device | None = None,
                                generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Haar-uniform rotation on SO(3), as a (B, 3, 3) matrix: a unit quaternion from a normalized
    4D isotropic Gaussian is uniform on S^3, which pushes forward to the Haar measure on SO(3).
    Copied from cryoet_mnist3d/rotation.py.
    """
    q = torch.randn(B, 4, device=device, generator=generator)
    return quaternion_to_matrix(q)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    (..., 4) quaternion (w, x, y, z), normalized internally -> (..., 3, 3) rotation matrix.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],     dim=-1),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],     dim=-1),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def _voxel_coords(D: int, device, dtype=torch.float32) -> torch.Tensor:
    # (D^3, 3) voxel coordinates (x, y, z) relative to index D//2, in (z, y, x) memory order.
    idx = torch.arange(D, dtype=dtype, device=device) - D // 2
    z, y, x = torch.meshgrid(idx, idx, idx, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def pose_volumes(x: torch.Tensor, R: torch.Tensor | None = None, shift: torch.Tensor | None = None,
                 center: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-sample posing: out_b(p) = x_b(R_b^T (p - s_b) + c_b).

    With c = 0 this is CryoBench's rotate_volume, batched over one rotation and shift per volume.
    `center` c (the (x, y, z) offset of the object from D//2) first moves c_b to the box centre,
    so recentring and posing cost one grid_sample.

    Args:
        x: (B, 1, D, D, D) volumes in (z, y, x) order
        R: (B, 3, 3) rotations (cryoDRGN convention), default identity
        shift: (B, 2) in-plane (x, y) shifts in pixels, default 0
        center: (B, 3) (x, y, z) offsets in voxels, default 0

    Returns:
        (B, 1, D, D, D)
    """
    B, _, D = x.shape[:3]
    p = _voxel_coords(D, x.device).expand(B, -1, -1)                   # (B, D^3, 3)
    if shift is not None:
        s = torch.zeros(B, 1, 3, device=x.device)
        s[:, 0, :2] = shift.to(x.device, torch.float32)
        p = p - s
    q = p @ R.to(x.device, torch.float32) if R is not None else p       # row vectors: R^T p
    if center is not None:
        q = q + center.to(x.device, torch.float32)[:, None, :]
    grid = ((q + D // 2) * (2 / (D - 1)) - 1).reshape(B, D, D, D, 3)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def center_of_mass(x: torch.Tensor) -> torch.Tensor:
    """
    (B, 1, D, D, D) -> (B, 3) (x, y, z) centre of mass of max(x, 0), relative to index D//2.
    Volumes with no positive mass get 0.
    """
    B, D = x.size(0), x.size(-1)
    w = x.float().clamp_min(0).reshape(B, -1)                          # (B, D^3)
    mass = w.sum(dim=1, keepdim=True)
    com = (w @ _voxel_coords(D, x.device)) / mass.clamp_min(1e-12)     # (B, 3)
    return torch.where(mass > 0, com, torch.zeros_like(com))


def mirror_z(x: torch.Tensor) -> torch.Tensor:
    """
    Reflect volumes through the z = D//2 plane (index k -> D - k mod D). A reflection along the
    viewing axis leaves the z-projection unchanged, so a single image cannot tell x from
    mirror_z(x): this is the handedness ambiguity eval_metrics scores around.
    """
    return torch.roll(torch.flip(x, dims=[-3]), shifts=1, dims=-3)
