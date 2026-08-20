import argparse
from dataclasses import dataclass

from data import Config_Dataset_MNIST
from scsi import Config_SCSI, Config_SCSI_MStep
from scsi_args import Config_Viz, add_scsi_args, scsi_configs_from_args


@dataclass
class Config:
    dataset: Config_Dataset_MNIST
    warmup: Config_SCSI_MStep   # mstep_lifted config for the (y, F(y)) warm start
    scsi: Config_SCSI           # nests .estep / .mstep for the EM loop proper
    viz: Config_Viz
    patch_size: int = 4         # DiT patch size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCSI on MNIST under an MRA-style channel: an unknown SO(2) in-plane "
                    "rotation followed by AWGN, drawn fresh per call (no tilt series, no "
                    "explicit angle recovery)."
    )

    # --- Dataset ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--n_images_per_class", type=int, default=5_000)
    dataset.add_argument("--image_size", type=int, default=28,
                         help="Must match model.IMAGE_SIZE (28) -- the DiT backbone assumes "
                              "this grid size.")
    dataset.add_argument("--digit_classes", type=int, nargs="+", default=None,
                         help="e.g. --digit_classes 3 7. Default: all 10 digits.")
    dataset.add_argument("--seed", type=int, default=42)
    dataset.add_argument("--train_split", dest="train", action="store_true", default=True)
    dataset.add_argument("--test_split", dest="train", action="store_false")

    # --- Corruption channel ---
    channel = parser.add_argument_group("corruption channel")
    channel.add_argument("--noise_std", type=float, default=1.5)

    # --- Model ---
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--patch_size", type=int, default=4)

    # --- Warmup training / SCSI e-step / SCSI m-step / SCSI outer loop / viz (shared) ---
    add_scsi_args(parser, default_wandb_project="scsi-mra-mnist")

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    dataset = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class,
        image_size=args.image_size,
        digit_classes=args.digit_classes,
        noise_std=args.noise_std,
        seed=args.seed,
        train=args.train,
    )

    warmup, scsi_config, viz = scsi_configs_from_args(args)

    return Config(dataset=dataset, warmup=warmup, scsi=scsi_config, viz=viz,
                  patch_size=args.patch_size)
