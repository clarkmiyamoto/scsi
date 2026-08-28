import argparse
from dataclasses import dataclass

from data import Config_Dataset_MNIST
from scsi import Config_SCSI, Config_SCSI_MStep
from scsi_args import Config_Viz, add_scsi_args, scsi_configs_from_args


@dataclass
class Config:
    dataset: Config_Dataset_MNIST
    warmup: Config_SCSI_MStep   # mstep_lifted config for the GT-supervised warm start
    scsi: Config_SCSI           # nests .estep / .mstep for the EM loop proper
    viz: Config_Viz
    block_out_channels: tuple[int, ...] = (64, 128, 256, 256)  # UNet3DConditionModel widths
    layers_per_block: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCSI on extruded-MNIST volumes under a 3D CryoET-style channel: a "
                    "Haar-uniform SO(3) mount + evenly-spaced tilt series about a fixed lab "
                    "axis -> 2D parallel-beam projection -> AWGN. 3D->2D counterpart of "
                    "experiments/cryoet_mnist."
    )

    # --- Dataset ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--n_images_per_class", type=int, default=500,
                         help="10x smaller than the 2D default -- volumes are ~V larger.")
    dataset.add_argument("--vol_size", type=int, default=32,
                         help="Must match model.VOL_SIZE (32).")
    dataset.add_argument("--inplane_size", type=int, default=None,
                         help="Digit load resolution; default round(vol_size * 0.65).")
    dataset.add_argument("--depth_extent", type=int, default=None,
                         help="Depth band the digit is extruded across; default "
                              "round(vol_size * 0.25).")
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
    channel.add_argument("--tilt_axis", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                         metavar=("W", "H", "D"),
                         help="Fixed physical tilt axis, in grid-sample (W, H, D) order. "
                              "Default (0, 1, 0) = the H axis, perpendicular to the "
                              "projection (D) axis.")

    # --- Warmup pseudoinverse (classical weighted-backprojection warm start) ---
    warmup_recon = parser.add_argument_group("warmup pseudoinverse")
    warmup_recon.add_argument("--filtered", dest="filtered", action="store_true", default=True,
                              help="Ramp-filter the warmup pseudoinverse (sharp WBP). Default: True.")
    warmup_recon.add_argument("--no_filtered", dest="filtered", action="store_false",
                              help="Use plain (unfiltered) backprojection instead.")
    warmup_recon.add_argument("--filter_type", type=str, default="hann", choices=["hann", "ramp"])

    # --- Model (UNet3DConditionModel) ---
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--block_out_channels", type=int, nargs="+",
                           default=[64, 128, 256, 256],
                           help="Per-level channel widths; #levels sets the spatial "
                                "downsampling depth (depth axis is never downsampled).")
    model_grp.add_argument("--layers_per_block", type=int, default=2)

    # --- Warmup training / SCSI e-step / SCSI m-step / SCSI outer loop / viz (shared) ---
    add_scsi_args(parser, default_wandb_project="scsi-cryoet-mnist3d")

    # 3D overrides for the shared batch-size defaults. A volume is ~V larger than a 2D image,
    # AND the diffusers video-UNet reshapes (B, C, D, H, W) -> (B*D, C, H, W) for its spatial
    # convs, multiplying the effective 2D batch by D again -- so the shared mstep_batch_size
    # 258 / estep_batch_size 1024 would OOM immediately. Each E-step posterior sample is also
    # a full ODE integration of that 32x-larger tensor, so drop estep_num_samples too.
    parser.set_defaults(
        mstep_batch_size=8,
        estep_batch_size=16,
        estep_num_samples=2_000,
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    dataset = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class,
        vol_size=args.vol_size,
        inplane_size=args.inplane_size,
        depth_extent=args.depth_extent,
        digit_classes=args.digit_classes,
        num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
        tilt_axis=tuple(args.tilt_axis),
        filtered=args.filtered,
        filter_type=args.filter_type,
        seed=args.seed,
        train=args.train,
    )

    warmup, scsi_config, viz = scsi_configs_from_args(args)

    return Config(dataset=dataset, warmup=warmup, scsi=scsi_config, viz=viz,
                  block_out_channels=tuple(args.block_out_channels),
                  layers_per_block=args.layers_per_block)
