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
    lift: bool = False          # E-step pairs (R.x_hat, F(x_hat)), R a fresh independent SO(3)
                                # rotation (corruption.build_pair_sample). Default off: preserves
                                # in-flight SCSI runs. main_supervised.py defaults this on.
    # LR schedule / EMA / checkpointing / eval (main.py). Defaults reproduce the original run:
    # one global cosine over warmup + all EM steps, raw weights sample, no checkpoints.
    lr_schedule: str = "cosine"                # "cosine" | "constant" | "cosine_per_mstep"
    lr_horizon_scsi_steps: int | None = None   # cosine reaches eta_min after this many EM steps;
                                               # None -> num_scsi_steps
    sample_with_ema: bool = False              # E-step + panels use the EMA weights
    save_warmup_ckpt: str | None = None
    load_warmup_ckpt: str | None = None
    ckpt_dir: str | None = None                # rolling latest.pt after every EM step
    resume: bool = False                       # continue from ckpt_dir/latest.pt if it exists
    em_seed: int | None = None                 # reseed the global RNG once warmup is done/loaded
    eval_n: int = 32                           # eval_metrics pool size; 0 disables the metric
    eval_seed: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCSI on extruded EMNIST-digit volumes under a 3D CryoET-style channel: a "
                    "Haar-uniform SO(3) mount + evenly-spaced tilt series about a fixed lab "
                    "axis -> 2D parallel-beam projection -> AWGN. 3D->2D counterpart of "
                    "experiments/cryoet_mnist (which stays on plain MNIST)."
    )

    # --- Dataset ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--n_images_per_class", type=int, default=23_000,
                         help="Per-digit draw from EMNIST 'digits' (24k train / 4k test per "
                              "class). 23k/class = 230k volumes: load_mnist_volumes holds the "
                              "whole (N,1,V,V,V) pool in RAM (~30 GB at V=32) and "
                              "build_observations / build_warmup each add a comparable tensor.")
    dataset.add_argument("--vol_size", type=int, default=32,
                         help="Must match model.VOL_SIZE (32).")
    dataset.add_argument("--digit_scale", type=float, default=1.0,
                         help="Isotropic size multiplier for the extruded digit: scales both "
                              "the in-plane footprint (default round(vol_size * 0.65)) and the "
                              "depth band (default round(vol_size * 0.25)). >1 widens/thickens "
                              "the shape, <1 shrinks it. --inplane_size / --depth_extent, when "
                              "given, override the corresponding axis. wandb point-cloud panels "
                              "size their voxel budget from the GT ink count, so they track "
                              "this automatically.")
    dataset.add_argument("--inplane_size", type=int, default=None,
                         help="Digit load resolution; overrides --digit_scale in-plane. "
                              "Default round(vol_size * 0.65 * digit_scale).")
    dataset.add_argument("--depth_extent", type=int, default=None,
                         help="Depth band the digit is extruded across; overrides --digit_scale "
                              "in depth. Default round(vol_size * 0.25 * digit_scale).")
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
    channel.add_argument("--lift", dest="lift", action="store_true", default=False,
                         help="Train the E-step / M-step on (R.x, F(x)) with R a fresh "
                              "independent random SO(3) rotation, instead of the canonical "
                              "(x, F(x)). Symmetrizes the target over the rotation group -- see "
                              "corruption.build_pair_sample. Default: off (preserves in-flight runs).")
    channel.add_argument("--no_lift", dest="lift", action="store_false",
                         help="Canonical (x, F(x)) target (the default).")

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

    # --- LR schedule / EMA / checkpointing / eval (this experiment only) ---
    sched = parser.add_argument_group("lr schedule / ema / checkpointing / eval")
    sched.add_argument("--lr_schedule", type=str, default="cosine",
                       choices=["cosine", "constant", "cosine_per_mstep"],
                       help="cosine (default): one cosine from --mstep_lr down to --eta_min over "
                            "warmup + --lr_horizon_scsi_steps EM steps (the original schedule). "
                            "constant: --mstep_lr throughout. cosine_per_mstep: a fresh cosine "
                            "--mstep_lr -> --eta_min over the warmup and again over every M-step. "
                            "The warmup always trains at --mstep_lr; --warmup_lr is still unused.")
    sched.add_argument("--lr_horizon_scsi_steps", type=int, default=None,
                       help="cosine only: the EM step at which the LR reaches --eta_min. Default "
                            "--num_scsi_steps. Smaller: the LR holds at --eta_min for the "
                            "remaining steps. Larger: the run stops partway down the cosine, "
                            "replaying the first N steps of a longer run's schedule.")
    sched.add_argument("--sample_with_ema", action="store_true", default=False,
                       help="E-step and wandb panels use the EMA weights (--mstep_ema) instead "
                            "of the raw ones. Default off: the EMA is tracked but was never "
                            "used. eval/ logs both either way.")
    sched.add_argument("--save_warmup_ckpt", type=str, default=None, metavar="PATH",
                       help="After warmup, torch.save model + EMA + optimizer + global step + "
                            "RNG state to PATH. Pair with --num_scsi_steps 0 for a warmup-only job.")
    sched.add_argument("--load_warmup_ckpt", type=str, default=None, metavar="PATH",
                       help="Skip build_warmup and warmup training and start EM from a "
                            "--save_warmup_ckpt file. Refuses a checkpoint whose dataset / model "
                            "args or observations differ from this run's. The LR is reset to "
                            "this run's --mstep_lr / --lr_schedule at the checkpoint's step.")
    sched.add_argument("--ckpt_dir", type=str, default=None, metavar="DIR",
                       help="Overwrite DIR/latest.pt (model + EMA + optimizer + scheduler) "
                            "after every EM step. Default: no checkpoints.")
    sched.add_argument("--resume", action="store_true", default=False,
                       help="If --ckpt_dir/latest.pt exists, continue from it: model, EMA, "
                            "optimizer, global + EM step, RNG state and the same wandb run. "
                            "Takes precedence over --load_warmup_ckpt. With no latest.pt, starts "
                            "normally, so one SBATCH file serves the first job and every chained "
                            "(--dependency=afterany) continuation. Refuses a latest.pt with "
                            "different dataset / model / EM args; --num_scsi_steps may grow.")
    sched.add_argument("--em_seed", type=int, default=None,
                       help="Reseed the global RNG right after warmup (or after "
                            "--load_warmup_ckpt restores its RNG state). Use it to get a "
                            "seed-to-seed noise floor from one shared warmup.")
    sched.add_argument("--eval_n", type=int, default=32,
                       help="Eval pool size for eval_metrics (rotation-searched correlation vs "
                            "GT), logged every --viz_every EM steps for both raw and EMA "
                            "weights. 0 disables it.")
    sched.add_argument("--eval_seed", type=int, default=1,
                       help="Seeds the eval pool's x0 / tilt series. Keep it fixed across runs "
                            "you want to compare.")

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

    args = parser.parse_args()
    if args.resume and not args.ckpt_dir:
        parser.error("--resume needs --ckpt_dir")
    return args


def config_from_args(args: argparse.Namespace) -> Config:
    dataset = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class,
        vol_size=args.vol_size,
        digit_scale=args.digit_scale,
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
                  layers_per_block=args.layers_per_block, lift=args.lift,
                  lr_schedule=args.lr_schedule,
                  lr_horizon_scsi_steps=args.lr_horizon_scsi_steps,
                  sample_with_ema=args.sample_with_ema,
                  save_warmup_ckpt=args.save_warmup_ckpt,
                  load_warmup_ckpt=args.load_warmup_ckpt,
                  ckpt_dir=args.ckpt_dir,
                  resume=args.resume,
                  em_seed=args.em_seed,
                  eval_n=args.eval_n,
                  eval_seed=args.eval_seed)
