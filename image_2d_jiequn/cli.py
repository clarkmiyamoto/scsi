"""Command-line interface -> typed ``Config`` for the minimal CryoET-SCSI
MNIST trainer."""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # forward model (CryoET tilt series)
    K: int = 16                     # number of tilt projections
    tilt_span_deg: float = 60.0     # half-span of the tilt schedule (degrees)
    epsilon: float = 0.0            # per-projection Gaussian noise std

    # SCSI / training
    train_steps: int = 2000
    warmup_steps: int = 800         # FBP-seeded warmup before bootstrap
    transport_steps: int = 200      # frozen transport-model refresh interval
    batch_size: int = 32
    lr: float = 2e-4
    ode_steps: int = 80             # Euler steps for transport
    alpha: float = 1.0              # prob of using freshly re-corrupted data
    ema_decay: float = 0.999
    network: str = "unet"           # 'unet' or 'dit' (dit = best FID, needs diffusers)
    model_channels: int = 32        # UNet base channels (ignored by dit)

    # data / io
    data_root: str = "./data"
    results_folder: str = "./results_clean"
    max_images: int = -1            # subset MNIST for fast smoke tests (-1 = all)
    num_workers: int = 0
    seed: int = 42


def parse_args(argv=None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(description="Minimal CryoET-SCSI on MNIST")
    p.add_argument("--K", type=int, default=d.K)
    p.add_argument("--tilt_span_deg", type=float, default=d.tilt_span_deg)
    p.add_argument("--epsilon", type=float, default=d.epsilon)
    p.add_argument("--train_steps", type=int, default=d.train_steps)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--transport_steps", type=int, default=d.transport_steps)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--ode_steps", type=int, default=d.ode_steps)
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--ema_decay", type=float, default=d.ema_decay)
    p.add_argument("--network", type=str, default=d.network, choices=["unet", "dit"])
    p.add_argument("--model_channels", type=int, default=d.model_channels)
    p.add_argument("--data_root", type=str, default=d.data_root)
    p.add_argument("--results_folder", type=str, default=d.results_folder)
    p.add_argument("--max_images", type=int, default=d.max_images)
    p.add_argument("--num_workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    return Config(**vars(p.parse_args(argv)))