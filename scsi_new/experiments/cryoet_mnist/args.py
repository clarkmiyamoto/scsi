import argparse
from dataclasses import dataclass

import torch

from data import Config_Dataset_MNIST
from scsi import Config_SCSI, Config_SCSI_EStep, Config_SCSI_MStep


@dataclass
class Config_Viz:
    seed: int = 0
    n_pool: int = 24
    n_display: int = 6
    n_trajectory_rows: int = 3
    n_snapshots: int = 8
    every: int = 1
    wandb_project: str = "scsi-cryoet-mnist"
    wandb_run_name: str | None = None


@dataclass
class Config:
    dataset: Config_Dataset_MNIST
    warmup: Config_SCSI_MStep   # mstep_lifted config for the pseudoinverse warm start
    scsi: Config_SCSI           # nests .estep / .mstep for the EM loop proper
    viz: Config_Viz
    arch: str = "dit"           # image branch backbone: "dit" or "unet"
    patch_size: int = 4         # only used when arch == "dit"


def _autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    # --- Warmup training (mstep_lifted on pseudoinverse (x_hat, y) pairs) ---
    warmup_train = parser.add_argument_group("warmup training")
    warmup_train.add_argument("--warmup_interpolant_style", type=str, default="gvp",
                              choices=["linear", "gvp"])
    warmup_train.add_argument("--warmup_n_steps_train", type=int, default=2_000)
    warmup_train.add_argument("--warmup_batch_size", type=int, default=258)
    warmup_train.add_argument("--warmup_lr", type=float, default=1e-4)
    warmup_train.add_argument("--warmup_weight_decay", type=float, default=0.0)
    warmup_train.add_argument("--warmup_ema", type=float, default=0.999)

    # --- SCSI E-step ---
    estep = parser.add_argument_group("scsi e-step")
    estep.add_argument("--estep_num_samples", type=int, default=10_000)
    estep.add_argument("--estep_method", type=str, default="euler", choices=["euler"])
    estep.add_argument("--estep_n_steps_sampling", type=int, default=64)
    estep.add_argument("--estep_batch_size", type=int, default=1024)

    # --- SCSI M-step ---
    mstep = parser.add_argument_group("scsi m-step")
    mstep.add_argument("--mstep_interpolant_style", type=str, default="gvp",
                       choices=["linear", "gvp"])
    mstep.add_argument("--mstep_n_steps_train", type=int, default=5_000)
    mstep.add_argument("--mstep_batch_size", type=int, default=258)
    mstep.add_argument("--mstep_lr", type=float, default=3e-4)
    mstep.add_argument("--mstep_weight_decay", type=float, default=0.0)
    mstep.add_argument("--mstep_ema", type=float, default=0.999)

    # --- SCSI outer loop ---
    scsi = parser.add_argument_group("scsi outer loop")
    scsi.add_argument("--num_scsi_steps", type=int, default=40)
    scsi.add_argument("--device", type=str, default=None,
                      choices=["cuda", "mps", "cpu"],
                      help="Default: autodetect cuda -> mps -> cpu.")
    scsi.add_argument("--eta_min", type=float, default=1e-5,
                      help="Smallest learning rate")

    # --- Visualization / wandb ---
    viz = parser.add_argument_group("visualization / wandb")
    viz.add_argument("--viz_seed", type=int, default=0,
                     help="Independent of --seed, so fixed panels match across hyperparameter sweeps.")
    viz.add_argument("--viz_n_pool", type=int, default=24)
    viz.add_argument("--viz_n_display", type=int, default=6)
    viz.add_argument("--viz_n_trajectory_rows", type=int, default=3)
    viz.add_argument("--viz_n_snapshots", type=int, default=8)
    viz.add_argument("--viz_every", type=int, default=1)
    viz.add_argument("--wandb_project", type=str, default="scsi-cryoet-mnist")
    viz.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    device = args.device or _autodetect_device()

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

    warmup = Config_SCSI_MStep(
        interpolant_style=args.warmup_interpolant_style,
        n_steps_train=args.warmup_n_steps_train,
        batch_size=args.warmup_batch_size,
        lr=args.warmup_lr,
        weight_decay=args.warmup_weight_decay,
        ema=args.warmup_ema,
    )

    estep = Config_SCSI_EStep(
        num_samples=args.estep_num_samples,
        method=args.estep_method,
        n_steps_sampling=args.estep_n_steps_sampling,
        batch_size=args.estep_batch_size,
    )

    mstep = Config_SCSI_MStep(
        interpolant_style=args.mstep_interpolant_style,
        n_steps_train=args.mstep_n_steps_train,
        batch_size=args.mstep_batch_size,
        lr=args.mstep_lr,
        weight_decay=args.mstep_weight_decay,
        ema=args.mstep_ema,
    )

    scsi_config = Config_SCSI(
        num_scsi_steps=args.num_scsi_steps,
        estep=estep,
        mstep=mstep,
        device=device,
        seed=args.seed,
        lr_eta_min=args.eta_min,
    )

    viz = Config_Viz(
        seed=args.viz_seed,
        n_pool=args.viz_n_pool,
        n_display=args.viz_n_display,
        n_trajectory_rows=args.viz_n_trajectory_rows,
        n_snapshots=args.viz_n_snapshots,
        every=args.viz_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    return Config(dataset=dataset, warmup=warmup, scsi=scsi_config, viz=viz,
                  arch=args.arch, patch_size=args.patch_size)
