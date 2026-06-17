"""CryoET tilt-series forward model (the corruption).

Minimal MNIST port of ``src/forward_maps.py::radon_tilt_series``. A clean image
is observed by ``K`` 1-D Radon projections at a *known* relative tilt schedule
``{Delta_theta_k}``, after rotating the image by a *single unknown* global angle
``theta0 ~ U(0, 2*pi)`` drawn per image. Relative angles within a stack are known;
the absolute orientation of the stack is not (this is the cryo-ET ambiguity).

Sign convention (must match ``backwards.fbp``): the forward path rotates by
``-theta`` then averages along the width axis. The pseudo-inverse rotates by
``+theta``. Getting these signs consistent is what makes the FBP round-trip work.
"""
import math

import torch
import torch.nn.functional as F


def rotate_image(image: torch.Tensor, angles: torch.Tensor, bg: float = 0.0) -> torch.Tensor:
    """Per-sample rotation of ``[N, C, H, W]`` images by ``angles[i]`` (radians).

    Uses ``affine_grid`` + ``grid_sample`` with a shift-trick that fills
    outside-source samples with ``bg`` instead of zero. For pm1 MNIST pass
    ``bg=-1.0`` so the rotated halo matches the background; otherwise the default
    zero-fill creates a +1 step at the rotation boundary that shows up as a
    spurious bright ring in the projections / back-projection. A positive angle
    rotates counter-clockwise (standard math convention).
    """
    cos, sin = torch.cos(angles), torch.sin(angles)
    zeros = torch.zeros_like(cos)
    theta = torch.stack([
        torch.stack([cos, -sin, zeros], dim=1),
        torch.stack([sin,  cos, zeros], dim=1),
    ], dim=1)                                                  # [N, 2, 3]
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    rotated = F.grid_sample(image - bg, grid, align_corners=False, padding_mode='zeros')
    return rotated + bg


def tilt_angles(K: int, tilt_span_deg: float) -> torch.Tensor:
    """Known relative tilt schedule ``linspace(-span, +span, K)`` in radians."""
    return torch.linspace(-tilt_span_deg, tilt_span_deg, K) * (math.pi / 180.0)


def forward_tilt_series(image: torch.Tensor, K: int = 16, tilt_span_deg: float = 60.0,
                        epsilon: float = 0.0, generator=None) -> torch.Tensor:
    """Clean image ``[N, C, H, W]`` (pm1) -> projections ``[N, K, H]``.

    Draws one unknown global rotation ``theta0`` per image, adds the known
    relative schedule, rotates by ``-angle``, projects (mean over width), and
    adds i.i.d. Gaussian noise of std ``epsilon`` to each projection.
    """
    N, C, H, W = image.shape
    device = image.device
    delta = tilt_angles(K, tilt_span_deg).to(device)
    theta0 = (torch.rand(N, generator=generator) * (2.0 * math.pi)).to(device)
    angles = theta0.unsqueeze(1) + delta.unsqueeze(0)          # [N, K]
    img_rep = image.unsqueeze(1).expand(N, K, C, H, W).reshape(N * K, C, H, W)
    rotated = rotate_image(img_rep, -angles.reshape(N * K), bg=-1.0)
    proj = rotated.mean(dim=(1, 3))                            # [N*K, H]
    if epsilon > 0:
        proj = proj + epsilon * torch.randn(proj.shape, generator=generator).to(device)
    return proj.reshape(N, K, H)


def radon_tilt_series(epsilon: float = 0.0, K: int = 16, tilt_span_deg: float = 60.0):
    """Factory returning ``fwd(x) -> (z_out, cond)``, the SCSI ``push_fwd`` contract.

    ``z_out`` is the image-shaped standard-Gaussian forward-model prior (the
    ``t=1`` endpoint the interpolant transports *from*). ``cond`` is the ``K``
    projections tiled along the width axis to ``[N, K, H, W]`` — the conditioning
    the drift network consumes via its image-latent path. The true projection
    vector is the first column of each ``[H, W]`` tile.
    """
    K = int(K)
    if K < 1:
        raise ValueError(f"radon_tilt_series: K must be >= 1, got {K}")

    def fwd(image: torch.Tensor, return_latents: bool = False, generator=None):
        squeeze = (image.dim() == 3)
        if squeeze:
            image = image.unsqueeze(0)
        N, C, H, W = image.shape
        proj = forward_tilt_series(image, K, tilt_span_deg, epsilon, generator)  # [N,K,H]
        cond = proj.unsqueeze(-1).expand(N, K, H, W).contiguous()                # tile -> [N,K,H,W]
        z_out = torch.randn((N, C, H, W), generator=generator).to(image.device)
        if squeeze:
            z_out, cond = z_out.squeeze(0), cond.squeeze(0)
        return (z_out, cond) if return_latents else z_out

    return fwd
