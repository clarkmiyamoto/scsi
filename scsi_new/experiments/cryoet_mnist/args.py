import argparse
from dataclasses import dataclass

from data import Config_Dataset_MNIST
from scsi import Config_SCSI, Config_SCSI_MStep
from scsi_args import Config_Viz, add_scsi_args, scsi_configs_from_args


@dataclass
class Config:
    dataset: Config_Dataset_MNIST
    warmup: Config_SCSI_MStep   # mstep_lifted config for the pseudoinverse warm start
    scsi: Config_SCSI           # nests .estep / .mstep for the EM loop proper
    viz: Config_Viz
    arch: str = "dit"           # image branch backbone: "dit" or "unet"
    patch_size: int = 4         # only used when arch == "dit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCSI on MNIST under a 2D CryoET-style channel: SO(2) in-plane rotation -> "
                    "1D projection -> AWGN, applied as a tilt series per image."
    )

    # --- Dataset ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--n_images_per_class", type=int, default=5_000)
    dataset.add_argument("--image_size", type=int, default=32,
                         help="Must match model.IMAGE_SIZE (32) -- both backbones and "
                              "pseudoinverse() assume this grid size.")
    dataset.add_argument("--digit_classes", type=int, nargs="+", default=None,
                         help="e.g. --digit_classes 3 7. Default: all 10 digits.")
    dataset.add_argument("--seed", type=int, default=42)
    dataset.add_argument("--train_split", dest="train", action="store_true", default=True)
    dataset.add_argument("--test_split", dest="train", action="store_false")

    # --- Corruption channel ---
    channel = parser.add_argument_group("corruption channel")
    channel.add_argument("--num_tilts", type=int, default=16)
    channel.add_argument("--tilt_increment_deg", type=float, default=7.5)
    channel.add_argument("--noise_std", type=float, default=3.0)

    # --- Warmup pseudoinverse (classical FBP warm start) ---
    warmup_recon = parser.add_argument_group("warmup pseudoinverse")
    warmup_recon.add_argument("--filtered", dest="filtered", action="store_true", default=True,
                              help="Ramp-filter the warmup pseudoinverse (sharp FBP). Default: True.")
    warmup_recon.add_argument("--no_filtered", dest="filtered", action="store_false",
                              help="Use plain (unfiltered) backprojection instead.")
    warmup_recon.add_argument("--filter_type", type=str, default="hann", choices=["hann", "ramp"])

    # --- Model ---
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--arch", type=str, default="dit", choices=["dit", "unet"])
    model_grp.add_argument("--patch_size", type=int, default=4, help="Only used when --arch dit.")

    # --- Warmup training / SCSI e-step / SCSI m-step / SCSI outer loop / viz (shared) ---
    add_scsi_args(parser, default_wandb_project="scsi-cryoet-mnist")

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    dataset = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class,
        image_size=args.image_size,
        digit_classes=args.digit_classes,
        num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
        filtered=args.filtered,
        filter_type=args.filter_type,
        seed=args.seed,
        train=args.train,
    )

    warmup, scsi_config, viz = scsi_configs_from_args(args)

    return Config(dataset=dataset, warmup=warmup, scsi=scsi_config, viz=viz,
                  arch=args.arch, patch_size=args.patch_size)
