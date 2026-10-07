import argparse
from dataclasses import dataclass

from .corruption import PAIR_FRAMES
from .data import DEFAULT_CRYOBENCH_ROOT, DEFAULT_DATA_ROOT, Config_Dataset_IgG
from .model import ENCODER_CFG, UNET_CFG
from scsi_new.scsi import Config_SCSI, Config_SCSI_MStep
from scsi_new.scsi_args import Config_Viz, add_scsi_args, scsi_configs_from_args


@dataclass
class Config:
    dataset: Config_Dataset_IgG
    warmup: Config_SCSI_MStep   # mstep_lifted config for the pseudoinverse warm start
    scsi: Config_SCSI           # nests .estep / .mstep for the EM loop proper
    viz: Config_Viz
    # Channel
    pair_frame: str = "image"   # target of each (target, F(x_hat)) pair; corruption.build_pair_sample
    recenter: bool = True       # move x_hat's centre of mass to the box centre before posing
    vol_gain: float | None = None  # F's projection gain = model volume units; None -> calibrate_gain
    # Model (cryofm's igg1d_cond_64 network)
    block_out_channels: tuple[int, ...] = UNET_CFG["block_out_channels"]
    layers_per_block: int = UNET_CFG["layers_per_block"]
    encoder_channels: tuple[int, ...] = ENCODER_CFG["channels"]
    bf16: bool = True
    # LR schedule / EMA / checkpointing / eval: as in cryoet_mnist3d
    lr_schedule: str = "cosine"                # "cosine" | "constant" | "cosine_per_mstep"
    lr_horizon_scsi_steps: int | None = None   # cosine reaches eta_min after this many EM steps
    sample_with_ema: bool = False              # E-step + panels use the EMA weights
    save_warmup_ckpt: str | None = None
    load_warmup_ckpt: str | None = None
    ckpt_dir: str | None = None                # rolling latest.pt after every EM step
    resume: bool = False                       # continue from ckpt_dir/latest.pt if it exists
    em_seed: int | None = None                 # reseed the global RNG once warmup is done/loaded
    eval_n: int = 16                           # eval pool size; 0 disables the metrics
    eval_seed: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCSI on CryoBench IgG-1D: single particle images of a 1D conformational "
                    "motion under a uniform SO(3) pose, in-plane shift, experimental CTF and noise. "
                    "Unsupervised counterpart of the image-conditioned cryofm model "
                    "(~/cryofm/scripts/train_igg1d.py): only the images are observed."
    )

    # --- Dataset ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                         help="Unzipped CryoBench IgG-1D directory.")
    dataset.add_argument("--cryobench_root", type=str, default=DEFAULT_CRYOBENCH_ROOT,
                         help="CryoBench checkout providing cryobench_data.igg1d.")
    dataset.add_argument("--resolution", type=int, default=64,
                         help="Volumes resolution^3, images resolution^2, Fourier-cropped from "
                              "128 px (box 384 A, so 6 A/px at 64). Even, and divisible by 8 "
                              "for the UNet's 3 downsamplings.")
    dataset.add_argument("--snr", type=float, default=0.01, help="Reads images/snr{snr}/.")
    dataset.add_argument("--val_fraction", type=float, default=0.02,
                         help="Held-out images, as cryofm. Observations are the rest.")
    dataset.add_argument("--split_seed", type=int, default=0, help="As cryofm.")
    dataset.add_argument("--n_observations", type=int, default=None,
                         help="Random subset of the ~98k training images. Default: all of them "
                              "(~1.6 GB at 64 px).")
    dataset.add_argument("--seed", type=int, default=42,
                         help="Draws the --n_observations subset; seeds model init and training.")

    # --- Corruption channel ---
    channel = parser.add_argument_group("corruption channel")
    channel.add_argument("--shift_extent_A", type=float, default=30.0,
                         help="In-plane shifts are uniform in +-this (A). CryoBench's is 30 A "
                              "(project3d --t-extent 20 px at 1.5 A/px).")
    channel.add_argument("--pair_frame", type=str, default="image", choices=PAIR_FRAMES,
                         help="Target paired with y = F(x_hat). image (default): x_hat as posed "
                              "by F, the volume whose projection y is -- the supervised cryofm "
                              "frame, and the one its z-broadcast conditioning assumes. "
                              "canonical: x_hat itself. lift: x_hat under an independent random "
                              "rotation (cryoet_mnist3d --lift).")
    channel.add_argument("--no_recenter", dest="recenter", action="store_false", default=True,
                         help="Pose x_hat as is. By default its centre of mass is first moved to "
                              "the box centre, since x_hat carries its source image's shift.")
    channel.add_argument("--vol_gain", type=float, default=None,
                         help="F multiplies the projection by this, which sets the units of the "
                              "volumes the model generates. Default: calibrated so the "
                              "pseudoinverse warm-start volumes have unit std "
                              "(data.calibrate_gain). `python -m scsi_new.experiments.cryoet_igg1d.data` prints this value and a "
                              "GT-derived one.")

    # --- Warm start (pseudoinverse.py) ---
    warm = parser.add_argument_group("warm start pseudoinverse")
    warm.add_argument("--pinv_diameter_A", type=float, default=192.0,
                      help="Each image's Wiener CTF-corrected backprojection is cut to a centred "
                           "sphere of this diameter (A), like a cryo-EM mask diameter. IgG-1D "
                           "density, shift included, lies within ~100 A of the box centre. 0 "
                           "smears through the whole box (the exact pseudoinverse). "
                           "`python -m scsi_new.experiments.cryoet_igg1d.pseudoinverse` scores values against GT.")

    # --- Model (cryofm's image-conditioned UNet, vendored) ---
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--block_out_channels", type=int, nargs="+",
                           default=list(UNET_CFG["block_out_channels"]),
                           help="UNet widths; the last two levels have attention.")
    model_grp.add_argument("--layers_per_block", type=int, default=UNET_CFG["layers_per_block"])
    model_grp.add_argument("--encoder_channels", type=int, nargs="+",
                           default=list(ENCODER_CFG["channels"]),
                           help="Image encoder widths.")
    model_grp.add_argument("--no_bf16", dest="bf16", action="store_false", default=True,
                           help="Run the network in fp32 (default: bf16 autocast on CUDA, as cryofm).")

    # --- Warmup training / SCSI e-step / SCSI m-step / SCSI outer loop / viz (shared) ---
    add_scsi_args(parser, default_wandb_project="scsi-cryoet-igg1d")

    # --- LR schedule / EMA / checkpointing / eval (as cryoet_mnist3d) ---
    sched = parser.add_argument_group("lr schedule / ema / checkpointing / eval")
    sched.add_argument("--lr_schedule", type=str, default="cosine",
                       choices=["cosine", "constant", "cosine_per_mstep"],
                       help="cosine (default): one cosine from --mstep_lr down to --eta_min over "
                            "warmup + --lr_horizon_scsi_steps EM steps. constant: --mstep_lr "
                            "throughout. cosine_per_mstep: a fresh cosine over the warmup and "
                            "again over every M-step. The warmup trains at --mstep_lr; "
                            "--warmup_lr is unused.")
    sched.add_argument("--lr_horizon_scsi_steps", type=int, default=None,
                       help="cosine only: the EM step at which the LR reaches --eta_min (then "
                            "held). Default --num_scsi_steps.")
    sched.add_argument("--sample_with_ema", action="store_true", default=False,
                       help="E-step and wandb panels use the EMA weights (--mstep_ema). eval/ "
                            "logs both either way.")
    sched.add_argument("--save_warmup_ckpt", type=str, default=None, metavar="PATH",
                       help="After warmup, save model + EMA + optimizer + global step + RNG "
                            "state to PATH. Pair with --num_scsi_steps 0 for a warmup-only job.")
    sched.add_argument("--load_warmup_ckpt", type=str, default=None, metavar="PATH",
                       help="Skip warmup and start EM from a --save_warmup_ckpt file. Refuses a "
                            "checkpoint whose dataset / channel / model args, observations or "
                            "vol_gain differ from this run's.")
    sched.add_argument("--ckpt_dir", type=str, default=None, metavar="DIR",
                       help="Overwrite DIR/latest.pt after every EM step.")
    sched.add_argument("--resume", action="store_true", default=False,
                       help="If --ckpt_dir/latest.pt exists, continue from it (same wandb run); "
                            "else start normally, so one SBATCH file serves the first job and "
                            "every --dependency=afterany continuation. --num_scsi_steps may grow.")
    sched.add_argument("--em_seed", type=int, default=None,
                       help="Reseed the global RNG right after warmup (or after "
                            "--load_warmup_ckpt restores its RNG state).")
    sched.add_argument("--eval_n", type=int, default=16,
                       help="Held-out images scored by eval_metrics every --viz_every EM steps, "
                            "for raw and EMA weights. 16 = the images cryofm's sampling eval "
                            "uses. 0 disables it.")
    sched.add_argument("--eval_seed", type=int, default=1,
                       help="Seeds the eval pool's fixed ODE noise x0.")

    # 64^3 overrides for the shared 2D batch-size defaults. cryofm measured 54 GB peak at batch 12
    # (bf16, A100-80GB) for this network; the E-step is inference only. Each E-step sample is a
    # full ODE integration, so estep_num_samples also stays small.
    parser.set_defaults(
        warmup_batch_size=12,
        mstep_batch_size=12,
        estep_batch_size=32,
        estep_num_samples=2_000,
        viz_n_pool=16,
        viz_n_display=8,
    )

    args = parser.parse_args()
    if args.resume and not args.ckpt_dir:
        parser.error("--resume needs --ckpt_dir")
    if args.resolution % 8:
        parser.error("--resolution must be divisible by 8")
    return args


def config_from_args(args: argparse.Namespace) -> Config:
    dataset = Config_Dataset_IgG(
        data_root=args.data_root,
        cryobench_root=args.cryobench_root,
        resolution=args.resolution,
        snr=args.snr,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        n_observations=args.n_observations,
        seed=args.seed,
        shift_extent_A=args.shift_extent_A,
        pinv_diameter_A=args.pinv_diameter_A or None,
    )

    warmup, scsi_config, viz = scsi_configs_from_args(args)

    return Config(dataset=dataset, warmup=warmup, scsi=scsi_config, viz=viz,
                  pair_frame=args.pair_frame,
                  recenter=args.recenter,
                  vol_gain=args.vol_gain,
                  block_out_channels=tuple(args.block_out_channels),
                  layers_per_block=args.layers_per_block,
                  encoder_channels=tuple(args.encoder_channels),
                  bf16=args.bf16,
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
