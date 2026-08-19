from pathlib import Path

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, ConcatDataset
from dataclasses import dataclass

from corruption import corruption_channel

@dataclass
class Config_Dataset_MNIST:
    # Dataset
    n_images_per_class: int = 5_000
    image_size: int = 28
    digit_classes: list[int] | None = None # e.g. [3, 7] will load digits 3's and 5's.

    # Corruption channel
    noise_std: float = 3.0

    # Warmup
    filtered: bool = True
    filter_type: str = "hann"  # "hann" or "ramp"

    # Misc
    seed: int = 42
    train: bool = True

def build_observations(config: Config_Dataset_MNIST) -> Dataset:
    '''
    Loads in MNIST and applies the forward model to generate observations.
    '''
    dataset_gt = load_mnist_subset(config)
    dataloader_gt = DataLoader(dataset_gt, batch_size=516, shuffle=False)

    observations = []
    for (x_gt,) in dataloader_gt:  # dataloader_gt yields (image,) tuples -- unpack the tensor
        y_obs = corruption_channel(x_gt, noise_std=config.noise_std)
        observations.append(y_obs)
    observations = torch.cat(observations, dim=0)

    return TensorDataset(observations)

def build_warmup(observations: Dataset, config: Config_Dataset_MNIST) -> Dataset:
    '''
    Warm is constructed by (F(x), F(F(x))).
    '''
    batch_size = 516

    dataloader_observations = DataLoader(observations, batch_size=batch_size, shuffle=False)
    device = observations[0][0].device

    xhat_batch = []
    y_batch = []

    for (y,) in dataloader_observations:
        x_hat = y
        y_hat = corruption_channel(x_hat, noise_std=config.noise_std)

        xhat_batch.append(x_hat)
        y_batch.append(y_hat)

    x_warmup = torch.cat(xhat_batch, dim=0)
    y_warmup = torch.cat(y_batch, dim=0)

    return TensorDataset(x_warmup, y_warmup)

def load_mnist_subset(config: Config_Dataset_MNIST) -> Dataset:
    """
    Load n_images_per_class random MNIST digits from EACH class in digit_classes,
    normalized to [-1, 1], concatenated in digit_classes order.
    """
    digit_classes = config.digit_classes if config.digit_classes is not None else list(range(10))

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(
        "./data", train=config.train, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(config.seed) if config.seed is not None else None

    chunks = []
    for digit_class in digit_classes:
        class_idx = (dataset.targets == digit_class).nonzero(as_tuple=True)[0]

        loader = DataLoader(
            Subset(dataset, class_idx),
            batch_size=config.n_images_per_class,
            shuffle=True,
            generator=generator,
        )

        x_c, _ = next(iter(loader))
        chunks.append(TensorDataset(x_c))

    return ConcatDataset(chunks)

def build_viz_pool(config: Config_Dataset_MNIST, n_pool: int, viz_seed: int) -> dict:
    '''
    Small diagnostic pool for wandb viz: n_pool images at deterministic positions in the SAME
    training subset build_observations() draws from (so they're guaranteed in-distribution), with
    a fixed x0/y seeded independently by viz_seed (comparable across hyperparameter sweeps).
    Runs entirely on CPU with the CPU RNG state saved/restored, so it has no effect on the rest
    of the run's RNG stream.
    '''
    dataset_gt = load_mnist_subset(config)
    idx = torch.linspace(0, len(dataset_gt) - 1, n_pool).round().long().tolist()
    x_gt = torch.stack([dataset_gt[i][0] for i in idx], dim=0)

    rng_state = torch.get_rng_state()
    torch.manual_seed(viz_seed)
    x0 = torch.randn(x_gt.size(0), 1, config.image_size, config.image_size)
    y = corruption_channel(x_gt, noise_std=config.noise_std)
    torch.set_rng_state(rng_state)

    return {"x_gt": x_gt, "x0": x0, "y": y}

if __name__ == "__main__":
    # Visualization CLI: gallery of GT digit / observed (rotated+noisy) image / warmup x_hat.
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Visualize build_observations()/build_warmup() on MNIST")
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None)
    parser.add_argument("--n_images_per_class", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--noise_std", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", action="store_true")
    parser.add_argument("--n_display", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="data_viz")
    args = parser.parse_args()

    config = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class, image_size=args.image_size,
        digit_classes=args.digit_classes, noise_std=args.noise_std,
        seed=args.seed, train=not args.test_split,
    )
    digit_classes = config.digit_classes or list(range(10))

    dataset_gt = load_mnist_subset(config)
    observations = build_observations(config)
    warmup = build_warmup(observations, config)

    ### Visualize warmup
    n_display = min(args.n_display, len(dataset_gt))
    idx = torch.linspace(0, len(dataset_gt) - 1, n_display).round().long().tolist()

    # Normalize contrast across all pannels.
    y_vmin, y_vmax = observations.tensors[0].aminmax()  # shared scale across all sinogram panels
    y_vmin_obs, y_vmax_obs = y_vmin.item(), y_vmax.item()

    v_min, v_max = warmup.tensors[0].aminmax()  # shared scale across all pseudoinverse panels
    v_min_warmup, v_max_warmup = v_min.item(), v_max.item()

    fig, axes = plt.subplots(3, n_display, figsize=(2 * n_display, 8.5), squeeze=False)
    for r, label in enumerate(["GT", "Observed\n(rotated + noise)", "Warmup x_hat\n(= y)"]):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for col, i in enumerate(idx):
        (x_gt,), (y_obs,) = dataset_gt[i], observations[i]
        x_hat, _ = warmup[i]

        axes[0, col].imshow(x_gt[0].numpy(), cmap="gray", vmin=v_min_warmup, vmax=v_max_warmup)
        axes[1, col].imshow(y_obs[0].numpy(), cmap="gray", vmin=y_vmin_obs, vmax=y_vmax_obs)
        axes[2, col].imshow(x_hat[0].numpy(), cmap="gray", vmin=v_min_warmup, vmax=v_max_warmup)
        for r in range(3):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
        axes[0, col].set_title(f"idx={i}", fontsize=8)

    fig.suptitle(f"build_observations() / build_warmup()  |  digits={digit_classes}  "
                f"n_per_class={config.n_images_per_class}  noise_std={config.noise_std}",
                fontsize=10)
    plt.tight_layout()

    # Save figure
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_gallery.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Gallery -> {out_path}")
