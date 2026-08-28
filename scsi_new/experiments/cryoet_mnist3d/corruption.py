import torch
from rotation import sample_tilt_series_rotations_so3, rotate_3d


def project_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel-beam projection: integrate along a fixed axis (depth). Rotation supplies the view
    diversity, same fixed-projection-axis convention as the 2D project_1d.

    Args:
        x: (B, 1, D, H, W)

    Returns:
        (B, 1, H, W)
    """
    return x.sum(dim=-3)


def cryoet_channel(xs: torch.Tensor, rotations: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Core CryoET channel, given explicit tilt-series rotations: rotate -> project -> AWGN, one
    projection per (volume, tilt) pair, batched.

    Args:
        xs: (B, 1, D, H, W) in [-1, 1]
        rotations: (B, T, 3, 3) -- a tilt series (T SO(3) matrices) per volume
        noise_std: float

    Returns:
        y: (B, T, 1, H, W)
    """
    B, _, D, H, W = xs.shape
    T = rotations.size(1)
    xs_rep = xs.unsqueeze(1).expand(-1, T, -1, -1, -1, -1).reshape(B * T, 1, D, H, W)
    R_flat = rotations.reshape(B * T, 3, 3)
    x_rot = rotate_3d(xs_rep, R_flat)                     # (B*T, 1, D, H, W)
    y = project_2d(x_rot)                                 # (B*T, 1, H, W)
    y = y + noise_std * torch.randn_like(y)
    return y.reshape(B, T, 1, H, W)


def corruption_channel(x: torch.Tensor,
                       num_tilts: int = 16,
                       tilt_increment_deg: float = 7.5,
                       noise_std: float = 3.0,
                       tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
                       rotations: torch.Tensor | None = None) -> torch.Tensor:
    """
    Black-box forward model: draws a fresh random tilt series (one Haar-uniform "mount" rotation
    + evenly spaced tilts about a fixed lab-frame axis).

    Args:
        x: (B, 1, D, H, W) in [-1, 1]
        num_tilts: T, tilt-series length. Unused if `rotations` is given.
        tilt_increment_deg: angular step between consecutive tilts, in degrees. Unused if
            `rotations` is given.
        noise_std: float
        tilt_axis: fixed physical tilt axis, in grid-sample coordinate order (W, H, D); see
            rotation.rotate_3d. Unused if `rotations` is given.
        rotations: (B, T, 3, 3), optional explicit tilt series (overrides num_tilts/
            tilt_increment_deg/tilt_axis).

    Returns:
        y: (B, num_tilts, 1, H, W) (or (B, rotations.size(1), 1, H, W) if rotations is given)
    """
    if rotations is None:
        tilt_increment_rad = tilt_increment_deg * torch.pi / 180.0
        rotations = sample_tilt_series_rotations_so3(x.size(0), num_tilts, tilt_increment_rad,
                                                    tilt_axis=tilt_axis, device=x.device)
    return cryoet_channel(x, rotations, noise_std=noise_std)
