import torch
import torch.nn.functional as F


def corruption_channel(x: torch.Tensor,
                       noise_std: float) -> torch.tensor:
    '''MRA Channel'''
    batch = x.size(0)
    device = x.device

    thetas = sample_uniform_angle(batch, device)
    ys = rotate_2d(x, thetas)
    ys = ys + noise_std * torch.randn_like(ys)

    return ys


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