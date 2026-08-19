import torch
import torch.nn.functional as F


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians into (-pi, pi]."""
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def sample_uniform_angle(B: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Haar-uniform angle on SO(2), drawn as the literal `z ~ N(0,1)` noise the pseudocode calls
    for: a 2-D isotropic Gaussian direction is exactly uniform on the unit circle.

    Returns:
        theta: (B,) in (-pi, pi]
    """
    z = torch.randn(B, 2, device=device)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.atan2(z[:, 1], z[:, 0])


def sample_tilt_series_angles(
    n_acquisitions: int, n_tilts: int, tilt_increment: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    One CryoET-style tilt series per acquisition: T = n_tilts angles, evenly spaced by
    `tilt_increment` radians, starting from an independent Haar-uniform random offset per
    acquisition (so the dataset doesn't share one fixed set of absolute angles).

    Args:
        n_acquisitions: how many independent tilt series to draw (e.g. one GT image imaged as
            one batch element gets one tilt series).
        n_tilts: T, number of tilts within one series.
        tilt_increment: angular step between consecutive tilts, in radians.

    Returns:
        theta: (n_acquisitions, n_tilts) in (-pi, pi]
    """
    theta0 = sample_uniform_angle(n_acquisitions, device)                       # (n_acq,)
    steps = torch.arange(n_tilts, device=device, dtype=theta0.dtype) * tilt_increment  # (T,)
    return wrap_to_pi(theta0[:, None] + steps[None, :])                        # (n_acq, T)


def rotate_2d(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate each (1, H, W) image in the batch by its own angle, filling the background with
    true black (-1), not grid_sample's raw zero-padding (which would be mid-gray in [-1,1]
    space). Done by rotating in a shifted (x+1) frame where zero-padding IS the background,
    then shifting back.

    Args:
        x: (B, 1, H, W) in [-1, 1]
        theta: (B,) radians

    Returns:
        (B, 1, H, W) in [-1, 1]
    """
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(theta)
    rot = torch.stack([
        torch.stack([cos, -sin, zeros], dim=-1),
        torch.stack([sin,  cos, zeros], dim=-1),
    ], dim=-2)  # (B, 2, 3)

    grid = F.affine_grid(rot, x.shape, align_corners=True)
    x_shifted = x + 1.0  # background -1 -> 0, matches zeros padding
    x_rot = F.grid_sample(x_shifted, grid, align_corners=True,
                          mode="bilinear", padding_mode="zeros")
    return x_rot - 1.0
