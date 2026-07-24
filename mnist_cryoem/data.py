import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from model import IMAGE_SIZE
from corruption import forward_channel, sample_tilt_series_angles


def load_mnist_subset(n_images_per_class: int, image_size: int = IMAGE_SIZE,
                      digit_classes: list[int] | None = None,
                      seed: int | None = None) -> torch.Tensor:
    """
    Load n_images_per_class random MNIST training digits from EACH class in digit_classes,
    normalized to [-1, 1], concatenated in digit_classes order.

    Args:
        n_images_per_class: how many images to draw per class (not a total count).
        digit_classes: which MNIST classes (0-9) to draw from. None = all 10 classes.
    """
    if digit_classes is None:
        digit_classes = list(range(10))
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    chunks = []
    for digit_class in digit_classes:
        class_idx = (dataset.targets == digit_class).nonzero(as_tuple=True)[0]
        loader = DataLoader(Subset(dataset, class_idx), batch_size=n_images_per_class,
                            shuffle=True, generator=generator)
        x_c, _ = next(iter(loader))  # (n_images_per_class, 1, image_size, image_size)
        chunks.append(x_c)
    return torch.cat(chunks, dim=0)


def build_observations(
    x_gt: torch.Tensor,
    corruptions_per_object: int,
    n_tilts: int,
    tilt_increment_deg: float,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the CryoEM channel F_CryoET to each GT object `corruptions_per_object` times —
    each repetition is an independent ACQUISITION: a full tilt series of `n_tilts` projections
    at evenly-spaced angles (step `tilt_increment_deg` degrees) from an independent random
    per-acquisition start offset, each tilt with its own independent AWGN. So a single object
    produces `corruptions_per_object * n_tilts` distinct observations (multiple tilt-series
    "shots" of the same particle).

    Args:
        x_gt: (n_images, 1, H, W)
        corruptions_per_object: number of independent tilt-series acquisitions per GT object
        n_tilts: T, number of tilts (projections) within one acquisition's series
        tilt_increment_deg: angular step between consecutive tilts, in degrees
        noise_std: float

    Returns:
        y_obs:      (n_images * corruptions_per_object * n_tilts, 1, W) — the observation set
            (this is mu; used for training)
        theta_star: (n_images * corruptions_per_object * n_tilts,)      — the true rotation
            used per observation. Diagnostic ONLY — never fed to the model, since the
            algorithm must recover the prior without knowing the true pose. Kept for eval
            plots / circular-error metrics.
        image_idx:  (n_images * corruptions_per_object * n_tilts,)      — which source object
            each observation came from, for grouped visualization.
        acq_idx:    (n_images * corruptions_per_object * n_tilts,)      — which acquisition
            (0..n_acq-1) each observation belongs to. Observations from the same acquisition
            are contiguous and already in tilt order (see `x_expanded`/`theta_flat` below),
            so `y_obs[acq_idx == a]` is one full tilt series in order — used to visualize the
            tilt-series structure (wandb_logging.py::log_reconstruction_grid) instead of an
            independent random draw over the flattened pool.
    """
    n_images = x_gt.size(0)
    n_acq = n_images * corruptions_per_object
    tilt_increment = tilt_increment_deg * torch.pi / 180.0

    x_per_acq = x_gt.repeat_interleave(corruptions_per_object, dim=0)         # (n_acq, 1, H, W)
    theta = sample_tilt_series_angles(n_acq, n_tilts, tilt_increment, x_gt.device)  # (n_acq, T)

    x_expanded = x_per_acq.repeat_interleave(n_tilts, dim=0)                  # (n_acq*T, 1, H, W)
    theta_flat = theta.reshape(-1)                                           # (n_acq*T,)
    y_obs, theta_star = forward_channel(x_expanded, noise_std=noise_std, theta=theta_flat)
    image_idx = torch.arange(n_images).repeat_interleave(corruptions_per_object * n_tilts)
    acq_idx = torch.arange(n_acq).repeat_interleave(n_tilts)
    return y_obs, theta_star, image_idx, acq_idx
