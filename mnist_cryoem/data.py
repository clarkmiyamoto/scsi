import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from model import IMAGE_SIZE
from corruption import forward_channel


def load_mnist_subset(n_images: int, image_size: int = IMAGE_SIZE,
                      seed: int | None = None) -> torch.Tensor:
    """Load a random subset of n_images MNIST training digits, normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(dataset, batch_size=n_images, shuffle=True, generator=generator)
    x_gt, _ = next(iter(loader))  # (n_images, 1, image_size, image_size)
    return x_gt


def build_observations(
    x_gt: torch.Tensor,
    corruptions_per_image: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the CryoEM channel `corruptions_per_image` times to each GT digit — each repetition
    gets an independent random rotation and independent noise, so a single digit produces
    multiple distinct observations (multiple "shots" of the same particle).

    Args:
        x_gt: (n_images, 1, H, W)
        corruptions_per_image: M
        noise_std: float

    Returns:
        y_obs:      (n_images * M, 1, W)  — the observation set (this is mu; used for training)
        theta_star: (n_images * M,)       — the true rotation used per observation.
            Diagnostic ONLY — never fed to the model, since the algorithm must recover the
            prior without knowing the true pose. Kept for eval plots / circular-error metrics.
        image_idx:  (n_images * M,)       — which source digit each observation came from,
            for grouped visualization.
    """
    n_images = x_gt.size(0)
    x_expanded = x_gt.repeat_interleave(corruptions_per_image, dim=0)
    y_obs, theta_star = forward_channel(x_expanded, noise_std=noise_std)
    image_idx = torch.arange(n_images).repeat_interleave(corruptions_per_image)
    return y_obs, theta_star, image_idx
