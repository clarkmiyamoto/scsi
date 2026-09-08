import sys
from pathlib import Path

# This file lives at scsi_new/experiments/cryoet_mnist3d/main_supervised.py. scsi.py, si.py,
# ode.py, distribution.py, and supervised.py live flat at scsi_new/ and import each other with
# bare imports (e.g. supervised.py does `from scsi import ...`), so scsi_new/ must be on
# sys.path. corruption.py / data.py / model.py / wandb_logging.py need no such fix: Python
# already adds a directly-run script's own directory to sys.path[0].
SCSI_NEW_ROOT = Path(__file__).resolve().parents[2]
if str(SCSI_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SCSI_NEW_ROOT))

import argparse
import functools

import torch
import wandb

from corruption import corruption_channel  # black box forward model
from data import Config_Dataset_MNIST, load_mnist_volumes, build_viz_pool
from distribution import IsotropicGaussian
from model import build_velocity_model
from supervised import Config_Supervised, autodetect_device, build_paired_dataset, train_supervised
from wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw

"""
Supervised baseline for experiments/cryoet_mnist3d -- the paired-data counterpart to main.py.

main.py runs SCSI: only the corrupted tilt series y are observed, and the EM loop bootstraps a
prior from them (warm-started on a pose-blind pseudoinverse). This script instead assumes oracle
access to the ground-truth extruded-digit volumes AND the forward model. It trains the same 3D
conditional velocity net b_t(.|y) directly on frozen {(x_i, F(x_i))} pairs by the
stochastic-interpolant loss -- supervised.train_supervised delegates each step to
scsi.mstep_lifted, so the loss is bit-for-bit the M-step's -- and logs through
cryoet_mnist3d/wandb_logging.py completely unchanged (z-slice grids + the interactive
point-cloud twin). Use it as the upper bound the unsupervised run in main.py is chasing.

Same RAM footprint as main.py: load_mnist_volumes holds the whole (N, 1, V, V, V) pool in RAM
(~30 GB at V=32, n_images_per_class=23k) and build_paired_dataset adds the paired y tensor on
top. Drop --n_images_per_class for a local run.

    uv run python experiments/cryoet_mnist3d/main_supervised.py --n_images_per_class 2000 --n_steps_train 8000
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised stochastic-interpolant baseline for the 3D->2D CryoET channel "
                    "(SO(3) mount + fixed-axis tilt series -> 2D projection -> AWGN). Fits "
                    "b_t(.|y) directly on ground-truth (x, F(x)) volume/tilt-series pairs -- no "
                    "EM loop, no warm start."
    )

    # --- Dataset (mirrors experiments/cryoet_mnist3d/args.py) ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--n_images_per_class", type=int, default=23_000,
                         help="Per-digit draw from EMNIST 'digits' (24k train / 4k test per "
                              "class). 23k/class = 230k volumes -- ~30 GB pool, plus the paired "
                              "y tensor. Lower this for anything but a cluster run.")
    dataset.add_argument("--vol_size", type=int, default=32, help="Must match model.VOL_SIZE (32).")
    dataset.add_argument("--inplane_size", type=int, default=None,
                         help="Digit load resolution; default round(vol_size * 0.65).")
    dataset.add_argument("--depth_extent", type=int, default=None,
                         help="Depth band the digit is extruded across; default round(vol_size * 0.25).")
    dataset.add_argument("--digit_classes", type=int, nargs="+", default=None,
                         help="e.g. --digit_classes 3 7. Default: all 10 digits.")
    dataset.add_argument("--seed", type=int, default=42,
                         help="Seeds the frozen channel draw for {x, F(x)}, model init, and "
                              "(inside train_supervised) the training batch order.")
    dataset.add_argument("--train_split", dest="train", action="store_true", default=True)
    dataset.add_argument("--test_split", dest="train", action="store_false")

    # --- Corruption channel (mirrors experiments/cryoet_mnist3d/args.py) ---
    channel = parser.add_argument_group("corruption channel")
    channel.add_argument("--num_tilts", type=int, default=16)
    channel.add_argument("--tilt_increment_deg", type=float, default=7.5)
    channel.add_argument("--noise_std", type=float, default=3.0)
    channel.add_argument("--tilt_axis", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                         metavar=("W", "H", "D"),
                         help="Fixed physical tilt axis, in grid-sample (W, H, D) order.")
    channel.add_argument("--channel_batch_size", type=int, default=32,
                         help="Samples per forward-model call when materializing {x, F(x)}. Small "
                              "on purpose: the channel expands to (B*num_tilts, 1, D, H, W) "
                              "internally (cf. data._CHANNEL_BATCH).")
    channel.add_argument("--resample_channel", action="store_true",
                         help="Re-draw F(x) every epoch (fresh SO(3) mount + tilt series + "
                              "noise) instead of freezing one realization per volume. Slow: the "
                              "channel then runs per-sample inside the dataloader.")

    # --- Model (mirrors experiments/cryoet_mnist3d/args.py, plus a DiT backbone option) ---
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--arch", type=str, default="unet", choices=["unet", "dit"],
                           help="Velocity-net backbone. 'unet' = ConditionalVelocityCryoET3D "
                                "(diffusers video-UNet, the original / main.py default); 'dit' = "
                                "ConditionalDiTCryoET3D (volumetric 3D-patch DiT, full 3D "
                                "self-attention, adaLN-Zero time conditioning). Both take the "
                                "same (x_t, t, y) and quantise t to 1000 bins identically.")
    model_grp.add_argument("--block_out_channels", type=int, nargs="+",
                           default=[64, 128, 256, 256],
                           help="[--arch unet] Per-level UNet3D channel widths; #levels sets the "
                                "spatial downsampling depth.")
    model_grp.add_argument("--layers_per_block", type=int, default=2,
                           help="[--arch unet] Residual blocks per UNet3D level.")
    model_grp.add_argument("--patch_size", type=int, default=4,
                           help="[--arch dit] Cubic patch edge; token grid is "
                                "(vol_size / patch_size)**3 (32/4 -> 512 tokens). Must divide vol_size.")
    model_grp.add_argument("--dit_hidden", type=int, default=384,
                           help="[--arch dit] Transformer width. Must be divisible by --dit_heads.")
    model_grp.add_argument("--dit_depth", type=int, default=12,
                           help="[--arch dit] Number of transformer blocks (~33M params at the "
                                "384/12/6 default, close to the UNet's ~37M).")
    model_grp.add_argument("--dit_heads", type=int, default=6,
                           help="[--arch dit] Attention heads.")
    model_grp.add_argument("--dit_mlp_ratio", type=float, default=4.0,
                           help="[--arch dit] Feed-forward expansion ratio.")

    # --- Supervised training (-> supervised.Config_Supervised) ---
    train = parser.add_argument_group("supervised training")
    train.add_argument("--interpolant_style", type=str, default="gvp", choices=["linear", "gvp"])
    train.add_argument("--n_steps_train", type=int, default=10_000,
                       help="Total optimizer steps; one (x, y) minibatch each.")
    train.add_argument("--batch_size", type=int, default=8,
                       help="A volume is ~V larger than a 2D image and the video-UNet reshapes "
                            "(B, C, D, H, W) -> (B*D, C, H, W) for its spatial convs -- keep this small.")
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight_decay", type=float, default=0.0)
    train.add_argument("--ema", type=float, default=0.999)
    train.add_argument("--eta_min", type=float, default=1e-5,
                       help="LR floor of the cosine schedule spanning all --n_steps_train.")
    train.add_argument("--checkpoint_steps", type=int, nargs="*", default=[], metavar="STEP",
                       help="Step counts at which to torch.save a checkpoint (model + EMA + "
                            "optimizer + scheduler + args). e.g. --checkpoint_steps 2000 6000 10000.")
    train.add_argument("--checkpoint_dir", type=str, default="checkpoints/cryoet_mnist3d_supervised",
                       help="Directory for step_<n>.pt checkpoints (gitignored).")
    train.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
                       help="Default: autodetect cuda -> mps -> cpu.")

    # --- Visualization / wandb ---
    viz = parser.add_argument_group("visualization / wandb")
    viz.add_argument("--log_every", type=int, default=1_000,
                     help="Training steps between wandb viz-panel dumps (the on_log callback).")
    viz.add_argument("--viz_ema", action="store_true",
                     help="Render viz panels from the EMA weights instead of the live model.")
    viz.add_argument("--viz_seed", type=int, default=0,
                     help="Independent of --seed, so fixed panels match across sweeps.")
    viz.add_argument("--viz_n_pool", type=int, default=24)
    viz.add_argument("--viz_n_display", type=int, default=6)
    viz.add_argument("--viz_n_trajectory_rows", type=int, default=3)
    viz.add_argument("--viz_n_snapshots", type=int, default=8)
    viz.add_argument("--viz_n_steps_sampling", type=int, nargs="+", default=[64], metavar="N",
                     help="ODE step counts to render the reconstruction/trajectory panels at -- "
                          "one panel set per value, keyed viz/{fixed,random}/ode<N>/. "
                          "e.g. --viz_n_steps_sampling 8 32 128.")
    viz.add_argument("--wandb_project", type=str, default="scsi-cryoet-mnist3d-supervised")
    viz.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace) -> Config_Dataset_MNIST:
    return Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class,
        vol_size=args.vol_size,
        inplane_size=args.inplane_size,
        depth_extent=args.depth_extent,
        digit_classes=args.digit_classes,
        num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
        tilt_axis=tuple(args.tilt_axis),
        seed=args.seed,
        train=args.train,
    )


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device or autodetect_device())

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    torch.manual_seed(args.seed)

    V = args.vol_size
    config_dataset = build_dataset_config(args)

    # Black-box forward model with its channel params bound -- one fresh random SO(3) mount + tilt
    # series per volume is drawn through this when the {x, F(x)} pool is materialized below (and
    # again every epoch if --resample_channel). Same binding main.py does for the E-step.
    corruption_channel_bound = functools.partial(
        corruption_channel,
        num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
        tilt_axis=config_dataset.tilt_axis,
    )

    # Model & noise source
    model = build_velocity_model(
        args.arch,
        vol_size=V,
        num_tilts=args.num_tilts,
        block_out_channels=tuple(args.block_out_channels),
        layers_per_block=args.layers_per_block,
        patch_size=args.patch_size,
        dit_hidden=args.dit_hidden,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        dit_mlp_ratio=args.dit_mlp_ratio,
    ).to(device)
    base_dist = IsotropicGaussian(shape=(1, V, V, V), device=device)

    # Supervised training set: {(x_i, F(x_i))} -- clean volumes paired with their tilt series.
    # load_mnist_volumes returns a raw (N, 1, V, V, V) tensor; build_paired_dataset wraps it.
    dataset = build_paired_dataset(
        load_mnist_volumes(config_dataset), corruption_channel_bound,
        batch_size=args.channel_batch_size,
    )

    # Fixed + random viz pools, exactly as in main.py.
    viz_pool = build_viz_pool(config_dataset, n_pool=args.viz_n_pool, viz_seed=args.viz_seed)
    fixed = {k: v[:args.viz_n_display] for k, v in viz_pool.items()}

    def log_all_panels(round_idx, global_step, ema_model):
        """train_supervised's on_log hook -- the supervised analogue of main.py::log_all_panels.
        `round_idx` stands in for `em_step`, so wandb_logging.py is reused verbatim. Each panel
        is rendered once per --viz_n_steps_sampling value (same x0/y draw, different integrator
        resolution), under key suffix ode<N>."""
        viz_model = ema_model if args.viz_ema else model
        rand = random_draw(viz_pool, config_dataset, args.viz_n_display)
        for panel_name, src in [("fixed", fixed), ("random", rand)]:
            for n_ode in args.viz_n_steps_sampling:
                tag = f"{panel_name}/ode{n_ode}"
                log_reconstruction_grid(
                    viz_model, src["x0"], src["y"], src["x_gt"],
                    config_dataset, n_ode,
                    round_idx, global_step, tag, device,
                )
                log_trajectory_grid(
                    viz_model, src["x0"], src["y"],
                    n_ode, args.viz_n_snapshots,
                    args.viz_n_trajectory_rows, round_idx, global_step, tag, device,
                )

    train_supervised(
        model, base_dist, dataset,
        Config_Supervised(
            interpolant_style=args.interpolant_style,
            n_steps_train=args.n_steps_train,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ema=args.ema,
            eta_min=args.eta_min,
            log_every=args.log_every,
            seed=args.seed,
            checkpoint_steps=tuple(args.checkpoint_steps),
            checkpoint_dir=args.checkpoint_dir,
        ),
        on_log=log_all_panels,
        resample_channel=corruption_channel_bound if args.resample_channel else None,
        checkpoint_meta={"args": vars(args)},
    )

    wandb.finish()
