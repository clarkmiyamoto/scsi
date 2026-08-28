import torch
import torch.nn.functional as F


def sample_uniform_rotation_so3(B: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Haar-uniform rotation on SO(3), as a (B, 3, 3) matrix -- the 3D analogue of
    rotation.sample_uniform_angle, drawn the same "literal z ~ N(0,1)" way: a unit quaternion
    from a normalized 4D isotropic Gaussian is exactly uniform on S^3, which pushes forward to
    the Haar measure on SO(3).

    Returns:
        R: (B, 3, 3)
    """
    q = torch.randn(B, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],     dim=-1),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],     dim=-1),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula: rotation by each `angle` radians about a single fixed `axis`.

    Args:
        axis: (3,) in grid-sample coordinate order -- component 0 is the W (last spatial) axis,
            1 is H, 2 is D (see rotate_3d). Need not be unit length; normalized internally.
        angle: (N,) radians.

    Returns:
        R: (N, 3, 3)
    """
    axis = axis / axis.norm().clamp_min(1e-8)
    ax, ay, az = axis.unbind()
    zero = torch.zeros((), device=angle.device, dtype=angle.dtype)
    K = torch.stack([
        torch.stack([zero, -az,  ay]),
        torch.stack([az,   zero, -ax]),
        torch.stack([-ay,  ax,   zero]),
    ])  # (3, 3) skew-symmetric cross-product matrix of the unit axis
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    s = torch.sin(angle)[:, None, None]
    c = torch.cos(angle)[:, None, None]
    return eye + s * K + (1 - c) * (K @ K)


def sample_tilt_series_rotations_so3(
    n_acquisitions: int, n_tilts: int, tilt_increment: float,
    tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    3D analogue of rotation.sample_tilt_series_angles: one Haar-uniform "mount" rotation per
    acquisition (unknown specimen orientation, applied FIRST / inner) composed with T
    evenly-spaced rotations by `tilt_increment` about a FIXED lab-frame `tilt_axis` (applied
    SECOND / outer), starting from an independent uniform random offset per acquisition. In 2D
    the "mount" and the "tilt axis" collapse into the same single SO(2) freedom; in 3D they are
    distinct, which is the whole reason this is not just sample_tilt_series_angles.

    R_total = R_tilt(angle_i) @ R_mount -- tilt outer (lab frame), mount inner (specimen frame:
    the specimen is mounted at a random unknown orientation FIRST, the holder then tilts about
    its own fixed physical axis SECOND). Swapping the order would tilt each acquisition about a
    different physical axis -- no longer a real tilt series.

    Args:
        n_acquisitions: how many independent tilt series to draw.
        n_tilts: T, number of tilts within one series.
        tilt_increment: angular step between consecutive tilts, in radians.
        tilt_axis: the FIXED physical tilt axis, in grid-sample coordinate order (W, H, D) --
            see axis_angle_to_matrix / rotate_3d. The default (0, 1, 0) is the H axis, which is
            perpendicular to the projection (D) axis, so the tilt series spans genuine views.

    Returns:
        R: (n_acquisitions, n_tilts, 3, 3)
    """
    R_mount = sample_uniform_rotation_so3(n_acquisitions, device)                     # (n_acq, 3, 3)
    start = torch.rand(n_acquisitions, device=device) * 2 * torch.pi                  # (n_acq,)
    steps = torch.arange(n_tilts, device=device, dtype=start.dtype) * tilt_increment  # (T,)
    angles = (start[:, None] + steps[None, :]).reshape(-1)                            # (n_acq*T,)

    axis = torch.tensor(tilt_axis, device=device, dtype=start.dtype)
    R_tilt = axis_angle_to_matrix(axis, angles).reshape(n_acquisitions, n_tilts, 3, 3)
    return R_tilt @ R_mount[:, None, :, :]                                            # (n_acq, T, 3, 3)


def rotate_3d(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate each (1, D, H, W) volume in the batch by its own rotation matrix, filling the
    background with true black (-1), not grid_sample's raw zero-padding (which would be mid-gray
    in [-1, 1] space). Same shifted-frame (x + 1) trick as rotate_2d: rotate where zero-padding
    IS the background, then shift back.

    F.affine_grid / F.grid_sample index the sampling grid's 3-vector as (W, H, D) -- so `R` acts
    on coordinates in that order, which is why tilt_axis is specified the same way.

    Args:
        x: (B, 1, D, H, W) in [-1, 1]
        R: (B, 3, 3)

    Returns:
        (B, 1, D, H, W) in [-1, 1]
    """
    zeros = torch.zeros(R.size(0), 3, 1, device=x.device, dtype=R.dtype)
    theta = torch.cat([R, zeros], dim=2)  # (B, 3, 4)

    grid = F.affine_grid(theta, x.shape, align_corners=True)
    x_shifted = x + 1.0  # background -1 -> 0, matches zeros padding
    x_rot = F.grid_sample(x_shifted, grid, align_corners=True,
                          mode="bilinear", padding_mode="zeros")
    return x_rot - 1.0
