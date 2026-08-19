"""
Minimal 2D -> 1D CryoET forward channel: in-plane SO(2) rotation -> 1D projection -> AWGN,
applied once per tilt in a whole tilt series per image. Rotation math lives in rotation.py;
this file only adds the projection operator and the noise/tilt-series assembly on top of it.
"""

import torch
from rotation import sample_tilt_series_angles, rotate_2d


def project_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel-beam projection: integrate along a fixed axis (height).

    Args:
        x: (B, 1, H, W)

    Returns:
        (B, 1, W)
    """
    return x.sum(dim=-2)


def cryoet_channel(xs: torch.Tensor, thetas: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Core CryoET channel, given explicit tilt-series angles: rotate -> project -> AWGN, one
    projection per (image, tilt) pair, batched.

    Args:
        xs: (B, 1, H, W) in [-1, 1]
        thetas: (B, T) radians -- a tilt series (T angles) per image
        noise_std: float

    Returns:
        y: (B, T, 1, W)
    """
    B, _, H, W = xs.shape
    T = thetas.size(1)
    xs_rep = xs.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, 1, H, W)
    theta_flat = thetas.reshape(B * T)
    x_rot = rotate_2d(xs_rep, theta_flat)               # (B*T, 1, H, W)
    y = project_1d(x_rot)                                 # (B*T, 1, W)
    y = y + noise_std * torch.randn_like(y)
    return y.reshape(B, T, 1, W)


def corruption_channel(x: torch.Tensor,
                       num_tilts: int = 16,
                       tilt_increment_deg: float = 7.5,
                       noise_std: float = 3.0,
                       thetas: torch.Tensor | None = None) -> torch.Tensor:
    """
    Black-box forward model: draws a fresh random tilt series (one random start angle + evenly
    spaced tilts) per image unless `thetas` is supplied explicitly -- the latter is how a pool's
    already-recovered angles get reused to compute ŷ = F(x̂; θ̂) instead of drawing fresh ones.
    This is the function main.py's E-step (scsi.py::estep, called with just `x`) and
    data.py::build_observations (called with num_tilts/tilt_increment_deg/noise_std as keyword
    args) both use.

    Args:
        x: (B, 1, H, W) in [-1, 1]
        num_tilts: T, tilt-series length. Unused if `thetas` is given.
        tilt_increment_deg: angular step between consecutive tilts, in degrees. Unused if
            `thetas` is given.
        noise_std: float
        thetas: (B, T) radians, optional explicit tilt series (overrides num_tilts/
            tilt_increment_deg).

    Returns:
        y: (B, num_tilts, 1, W) (or (B, thetas.size(1), 1, W) if thetas is given)
    """
    if thetas is None:
        tilt_increment_rad = tilt_increment_deg * torch.pi / 180.0
        thetas = sample_tilt_series_angles(x.size(0), num_tilts, tilt_increment_rad,
                                           device=x.device)
    return cryoet_channel(x, thetas, noise_std=noise_std)
