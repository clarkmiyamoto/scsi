import sys
from pathlib import Path

# This file lives at scsi_new/experiments/cryoet_mnist/main_supervised.py. scsi.py, si.py,
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
from data import Config_Dataset_MNIST, load_mnist_subset, build_viz_pool
from distribution import IsotropicGaussian
from model import ConditionalVelocityCryoET
from supervised import Config_Supervised, autodetect_device, build_paired_dataset, train_supervised
from wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw

"""
Supervised baseline for experiments/cryoet_mnist -- the paired-data counterpart to main.py.

main.py runs SCSI: only the corrupted tilt series y are observed, and the EM loop bootstraps a
prior from them. This script instead assumes oracle access to the ground-truth digits AND the
forward model. It trains the same conditional velocity net b_t(.|y) directly on frozen
{(x_i, F(x_i))} pairs by the stochastic-interpolant loss -- supervised.train_supervised delegates
each step to scsi.mstep_lifted, so the loss is bit-for-bit the M-step's -- and logs through
cryoet_mnist/wandb_logging.py completely unchanged. Use it as the upper bound the unsupervised
run in main.py is trying to reach.

    uv run python experiments/cryoet_mnist/main_supervised.py --n_steps_train 40000
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised stochastic-interpolant baseline for the 2D CryoET MNIST channel "
                    "(SO(2) tilt series -> 1D projection -> AWGN). Fits b_t(.|y) directly on "
                    "ground-truth (x, F(x)) pairs -- no EM loop, no warm start."
    )

    # --- Dataset (mirrors experiments/cryoet_mnist/args.py) ---
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--n_images_per_class", type=int, default=5_000)
    dataset.add_argument("--image_size", type=int, default=32,
                         help="Must match model.IMAGE_SIZE (32).")
    dataset.add_argument("--digit_classes", type=int, nargs="+", default=None,
                         help="e.g. --digit_classes 3 7. Default: all 10 digits.")
    dataset.add_argument("--seed", type=int, default=42,
                         help="Seeds the frozen channel draw for {x, F(x)}, model init, and "
                              "(inside train_supervised) the training batch order.")
    dataset.add_argument("--train_split", dest="train", action="store_true", default=True)
    dataset.add_argument("--test_split", dest="train", action="store_false")

    # --- Corruption channel (mirrors experiments/cryoet_mnist/args.py) ---
    channel = parser.add_argument_group("corruption channel")
    channel.add_argument("--num_tilts", type=int, default=16)
    channel.add_argument("--tilt_increment_deg", type=float, default=7.5)
    channel.add_argument("--noise_std", type=float, default=3.0)
    channel.add_argument("--channel_batch_size", type=int, default=512,
                         help="Samples per forward-model call when materializing {x, F(x)}.")
    channel.add_argument("--resample_channel", action="store_true",
                         help="Re-draw F(x) every epoch (fresh tilt series + noise) instead of "
                              "freezing one realization per digit.")

    # --- Model (mirrors experiments/cryoet_mnist/args.py) ---
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--arch", type=str, default="dit", choices=["dit", "unet"])
    model_grp.add_argument("--patch_size", type=int, default=4, help="Only used when --arch dit.")

    # --- Supervised training (-> supervised.Config_Supervised) ---
    train = parser.add_argument_group("supervised training")
    train.add_argument("--interpolant_style", type=str, default="gvp", choices=["linear", "gvp"])
    train.add_argument("--n_steps_train", type=int, default=40_000,
                       help="Total optimizer steps; one (x, y) minibatch each.")
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight_decay", type=float, default=0.0)
    train.add_argument("--ema", type=float, default=0.999)
    train.add_argument("--eta_min", type=float, default=1e-5,
                       help="LR floor of the cosine schedule spanning all --n_steps_train.")
    train.add_argument("--checkpoint_steps", type=int, nargs="*", default=[], metavar="STEP",
                       help="Step counts at which to torch.save a checkpoint (model + EMA + "
                            "optimizer + scheduler + args). e.g. --checkpoint_steps 5000 20000 40000.")
    train.add_argument("--checkpoint_dir", type=str, default="checkpoints/cryoet_mnist_supervised",
                       help="Directory for step_<n>.pt checkpoints (gitignored).")
    train.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
                       help="Default: autodetect cuda -> mps -> cpu.")

    # --- Visualization / wandb ---
    viz = parser.add_argument_group("visualization / wandb")
    viz.add_argument("--log_every", type=int, default=2_000,
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
    viz.add_argument("--wandb_project", type=str, default="scsi-cryoet-mnist-supervised")
    viz.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace) -> Config_Dataset_MNIST:
    return Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class,
        image_size=args.image_size,
        digit_classes=args.digit_classes,
        num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
        seed=args.seed,
        train=args.train,
    )


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device or autodetect_device())

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    torch.manual_seed(args.seed)

    config_dataset = build_dataset_config(args)

    # Black-box forward model with its channel params bound -- one fresh random tilt series per
    # digit is drawn through this when the {x, F(x)} pool is materialized below (and again every
    # epoch if --resample_channel). Same binding main.py does for the E-step's re-corruption.
    corruption_channel_bound = functools.partial(
        corruption_channel,
        num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
    )

    # Model & noise source
    model = ConditionalVelocityCryoET(
        image_size=args.image_size, arch=args.arch, patch_size=args.patch_size,
    ).to(device)
    base_dist = IsotropicGaussian(
        shape=(1, args.image_size, args.image_size), device=device,
    )

    # Supervised training set: {(x_i, F(x_i))} -- clean digits paired with their corruption.
    dataset = build_paired_dataset(
        load_mnist_subset(config_dataset), corruption_channel_bound,
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
                    viz_model, src["x0"], src["theta"], src["y"], src["x_gt"],
                    config_dataset.noise_std, n_ode,
                    round_idx, global_step, tag, device,
                )
                log_trajectory_grid(
                    viz_model, src["x0"], src["theta"], src["y"],
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
