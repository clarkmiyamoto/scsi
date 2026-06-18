"""Entry point: wire dataset -> model -> interpolant -> trainer and run.

    python main.py --train_steps 2000 --warmup_steps 800 --max_images 500

All knobs live in ``cli.py``. The pieces:
    data.CorruptedTiltDataset  : MNIST + on-the-fly tilt corruption
    forward.radon_tilt_series  : the CryoET forward map (push_fwd)
    backwards.warmup_target    : FBP reconstruction used to seed warmup
    model.build_model          : conditional drift UNet
    scsi.SCSInterpolant        : loss + transport
    bootstrap.Trainer          : the self-consistency loop
"""
import json
import math
import os
from dataclasses import asdict

import torch

from cli import parse_args
from data import load_mnist_pm1, CorruptedTiltDataset
from forward import radon_tilt_series, rotate_image
from backwards import warmup_target
from model import build_model
from scsi import SCSInterpolant
from bootstrap import Trainer


def make_orbit_random_fn():
    """Spread the warmup target over the rotation orbit: apply an independent
    random global rotation per image (the cryo-ET ``theta0`` is unknown, so the
    warmup x_0 should not bake in FBP's canonical pose)."""
    def fn(x0):
        ang = torch.rand(x0.shape[0], device=x0.device) * (2.0 * math.pi)
        return rotate_image(x0, ang, bg=-1.0)
    return fn


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = load_mnist_pm1(cfg.data_root)
    if cfg.max_images > 0:
        base = torch.utils.data.Subset(base, range(cfg.max_images))

    fwd = radon_tilt_series(cfg.epsilon, cfg.K, cfg.tilt_span_deg)
    dataset = CorruptedTiltDataset(base, fwd, base_seed=cfg.seed)

    model = build_model(D=32, nc=1, K=cfg.K, model_channels=cfg.model_channels,
                        network=cfg.network)
    interpolant = SCSInterpolant(fwd, n_steps=cfg.ode_steps, alpha=cfg.alpha)

    trainer = Trainer(
        model, interpolant, dataset,
        lr=cfg.lr, batch_size=cfg.batch_size, train_steps=cfg.train_steps,
        warmup_steps=cfg.warmup_steps, transport_steps=cfg.transport_steps,
        ema_decay=cfg.ema_decay, results_folder=cfg.results_folder,
        warmup_target_fn=lambda c: warmup_target(c, cfg.K, cfg.tilt_span_deg),
        warmup_orbit_random_fn=make_orbit_random_fn(),
        num_workers=cfg.num_workers, device=device,
    )

    with open(os.path.join(cfg.results_folder, "args.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    trainer.train()


if __name__ == "__main__":
    main()