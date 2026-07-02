"""MNIST digit pool -> point-cloud data distribution.

Unlike ``toy_3d_pc``'s synthetic SDF-solid shapes, the "clean signal" distribution here is a
finite pool of real MNIST images restricted to one digit class: :func:`load_mnist_digit_pool`
takes the *first* ``n_images`` examples of a chosen digit (in dataset order, not shuffled).
Each object is then a point cloud sampled from one pool image's pixel-intensity density --
:func:`image_to_pointcloud` treats pixel intensity as an unnormalized density and draws
``n_points`` pixel locations via ``torch.multinomial`` (inverse-CDF sampling), with uniform
sub-pixel jitter so coordinates are continuous rather than snapped to a lattice.
:func:`make_mnist_sampler` wraps the pool into the same ``(batch, n_points, device) ->
(batch, n_points, 2)`` signature as ``toy_3d_pc.data``'s shape samplers, so it drops into
``warmstart``/``scsi``/``supervised`` unchanged; repeated calls draw fresh images from the pool
and fresh point realizations of them, mirroring the stochasticity of the 3D rejection sampler.
Clouds live in world coordinates ``[-extent, extent]``, not normalized to ``[-1, 1]``.
"""
from __future__ import annotations

from typing import Callable

import torch
import torchvision.transforms as transforms
from torchvision import datasets


def load_mnist_digit_pool(
    digit: int, n_images: int, image_size: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """First ``n_images`` MNIST training examples of ``digit`` -> (n_images, P, P) in [0, 1]."""
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    imgs: list[torch.Tensor] = []
    for img, label in dataset:
        if label == digit:
            imgs.append(img.squeeze(0))
            if len(imgs) >= n_images:
                break
    if len(imgs) < n_images:
        raise RuntimeError(
            f"only found {len(imgs)}/{n_images} MNIST training images of digit {digit}"
        )
    return torch.stack(imgs, dim=0).to(device)  # (n_images, P, P)


def _images_to_pointclouds(
    imgs: torch.Tensor, n_points: int, extent: float
) -> torch.Tensor:
    """imgs: (B, P, P) intensities in [0, 1] -> (B, n_points, 2) in [-extent, extent]^2.

    Vectorized inverse-CDF sampling: ``torch.multinomial`` draws ``n_points`` pixel indices
    per image weighted by intensity, then indices are unraveled to (row, col) and mapped to
    world coordinates on a ``P``-point lattice, plus uniform sub-pixel jitter for continuity.
    """
    B, P, _ = imgs.shape
    device = imgs.device
    weights = imgs.reshape(B, P * P).clamp(min=0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    idx = torch.multinomial(weights, n_points, replacement=True)  # (B, n_points)
    row = torch.div(idx, P, rounding_mode="floor")
    col = idx % P
    grid = torch.linspace(-extent, extent, P, device=device)
    px = grid[col]                                                # (B, n_points)
    py = grid[row]
    spacing = (2.0 * extent) / max(P - 1, 1)
    jitter = (torch.rand(B, n_points, 2, device=device) - 0.5) * spacing
    return torch.stack([px, py], dim=-1) + jitter


def image_to_pointcloud(image: torch.Tensor, n_points: int, extent: float) -> torch.Tensor:
    """Single-image convenience wrapper: (P, P) -> (n_points, 2)."""
    return _images_to_pointclouds(image.unsqueeze(0), n_points, extent)[0]


def make_mnist_sampler(
    pool: torch.Tensor, extent: float
) -> Callable[..., torch.Tensor]:
    """Wrap an image pool (n_images, P, P) into a (batch, n_points, device) -> (B,N,2) sampler.

    Each call draws ``batch`` images uniformly (with replacement) from the pool and re-samples
    ``n_points`` points from each -- same signature as ``toy_3d_pc.data.SHAPE_SAMPLERS``.
    """
    n_images = pool.shape[0]

    def sampler(batch: int, n_points: int, device: torch.device | str = "cpu") -> torch.Tensor:
        pool_d = pool.to(device)
        img_idx = torch.randint(n_images, (batch,), device=device)
        imgs = pool_d[img_idx]                                    # (batch, P, P)
        return _images_to_pointclouds(imgs, n_points, extent)

    return sampler
