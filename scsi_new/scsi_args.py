import argparse
from dataclasses import dataclass

import torch

from scsi import Config_SCSI, Config_SCSI_EStep, Config_SCSI_MStep

"""
Shared argparse groups + config builders for the part of the CLI that's identical across every
experiments/<name>/args.py: warmup training, SCSI e-step, SCSI m-step, SCSI outer loop, and
visualization/wandb. Each experiment's own args.py still owns its dataset, corruption-channel,
and model argument groups -- those vary per experiment and stay local.

Named scsi_args.py, not args.py, on purpose: main.py inserts scsi_new/ at sys.path[0], AHEAD of
the script's own directory (see the comment at the top of main.py), so a root-level args.py
would shadow each experiment's local args.py instead of living alongside it.
"""


@dataclass
class Config_Viz:
    seed: int = 0
    n_pool: int = 24
    n_display: int = 6
    n_trajectory_rows: int = 3
    n_snapshots: int = 8
    every: int = 1
    wandb_project: str = "scsi"
    wandb_run_name: str | None = None


def autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def add_scsi_args(parser: argparse.ArgumentParser, default_wandb_project: str) -> None:
    """
    Adds every argument group shared across experiments: warmup training, SCSI e-step, SCSI
    m-step, SCSI outer loop, and visualization/wandb. Call after adding the experiment's own
    dataset / corruption-channel / model groups.
    """
    # --- Warmup training (mstep_lifted config for the warm start) ---
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
    viz.add_argument("--wandb_project", type=str, default=default_wandb_project)
    viz.add_argument("--wandb_run_name", type=str, default=None)


def scsi_configs_from_args(args: argparse.Namespace) -> tuple[Config_SCSI_MStep, Config_SCSI, Config_Viz]:
    """
    Builds (warmup, scsi, viz) config objects from the arguments add_scsi_args() adds, plus
    --seed and --device, which the SCSI outer-loop config also needs but which live in each
    experiment's own dataset / outer-loop groups.

    Returned as a plain tuple, not a combined dataclass: each experiment's own top-level Config
    also nests experiment-specific fields (dataset, model arch, ...) that this module has no
    knowledge of, so the caller assembles the final Config itself.
    """
    device = args.device or autodetect_device()

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

    return warmup, scsi_config, viz
