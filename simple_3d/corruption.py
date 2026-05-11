import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation


def forward_channel(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    CryoEM forward channel: random SO(3) rotation -> Radon projection -> AWGN.

    Args:
        x: (B, 1, D, H, W) float32
        noise_std: float

    Returns:
        y: (B, 1, H, W) — 2D projection
    """
    x_rot = random_so3_rotate(x)
    proj = radon_projection(x_rot)
    return proj + noise_std * torch.randn_like(proj)


def radon_projection(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel-beam Radon transform: integrate along the beam (depth) axis.
    P(x,y) = integral V_rot(x,y,z) dz  ->  sum over dim 2.

    Args:
        x: (B, 1, D, H, W)

    Returns:
        (B, 1, H, W)
    """
    return x.sum(dim=2)


def random_so3_rotate(x: torch.Tensor) -> torch.Tensor:
    """
    Apply independent Haar-uniform SO(3) rotations to each volume in the batch.

    Args:
        x: (B, 1, D, H, W)

    Returns:
        rotated volumes of the same shape
    """
    B = x.size(0)
    R_np = Rotation.random(B).as_matrix().astype(np.float32)   # (B, 3, 3)
    R = torch.from_numpy(R_np).to(x.device)                    # (B, 3, 3)

    zeros = torch.zeros(B, 3, 1, device=x.device)
    theta = torch.cat([R, zeros], dim=2)                        # (B, 3, 4)

    grid = F.affine_grid(theta, x.shape, align_corners=True)   # (B, D, H, W, 3)
    return F.grid_sample(x, grid, align_corners=True,
                         mode='bilinear', padding_mode='zeros')
