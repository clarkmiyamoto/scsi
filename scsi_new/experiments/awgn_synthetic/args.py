import argparse
from dataclasses import dataclass

from data import Config_Dataset_Synthetic
from scsi import Config_SCSI, Config_SCSI_MStep
from scsi_args import Config_Viz, add_scsi_args, scsi_configs_from_args


@dataclass
class Config:
    dataset: Config_Dataset_Synthetic
    warmup: Config_SCSI_MStep   # mstep_lifted config for the poor-Gaussian warm start
    scsi: Config_SCSI           # nests .estep / .mstep for the EM loop proper
    viz: Config_Viz


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",")]


def _as_2x2(flat: list[float]) -> list[list[float]]:
    assert len(flat) == 4, f"expected 4 comma-separated values for a 2x2 covariance, got {len(flat)}"
    return [flat[0:2], flat[2:4]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCSI on a synthetic 2D point distribution under an AWGN channel, "
                    "warm-started on a user-specified -- and potentially POOR / mismatched -- "
                    "Gaussian instead of anything derived from the data, to study whether the EM "
                    "loop can still recover the full ground-truth distribution."
    )

    # --- Ground-truth distribution + misc dataset ---
    dataset = parser.add_argument_group("ground-truth distribution")
    dataset.add_argument("--gt_distribution", type=str, default="ring_gmm",
                         choices=["gaussian", "gmm", "ring", "ring_gmm", "miffy"],
                         help="'gmm' requires constructing Config_Dataset_Synthetic directly "
                              "with gt_means/gt_covs -- not exposed via CLI.")
    dataset.add_argument("--n_samples", type=int, default=10_000)
    dataset.add_argument("--gt_mean", type=_floats, default=[0.0, 0.0],
                         help="Comma-separated, e.g. '0,0'. Center for "
                              "gaussian/ring/ring_gmm/miffy (miffy: bounding-box center).")
    dataset.add_argument("--gt_cov", type=_floats, default=[1.0, 0.0, 0.0, 1.0],
                         help="Comma-separated 2x2 row-major, e.g. '1,0,0,1'. gaussian only.")
    dataset.add_argument("--ring_radius", type=float, default=4.0)
    dataset.add_argument("--ring_width", type=float, default=0.3,
                         help="ring only: Gaussian radial jitter.")
    dataset.add_argument("--ring_n_modes", type=int, default=8, help="ring_gmm only.")
    dataset.add_argument("--ring_mode_std", type=float, default=0.3,
                         help="ring_gmm only: std of each mode bump (channel noise is separate, "
                              "see --noise_std).")
    dataset.add_argument("--miffy_width", type=float, default=2.0,
                         help="miffy only: bounding-box width to rescale the (native 2.0 x 2.81) "
                              "shape to. Default is native size -- see --noise_std's help for why "
                              "you likely want to raise this (or lower --noise_std) for miffy.")
    dataset.add_argument("--miffy_height", type=float, default=2.81,
                         help="miffy only: bounding-box height to rescale to (native 2.81).")
    dataset.add_argument("--miffy_thickness", type=float, default=0.02,
                         help="miffy only: line-jitter thickness, in the SAME pre-rescale units "
                              "as --miffy_width/--miffy_height's native 2.0 x 2.81 -- i.e. it "
                              "scales along with the shape when width/height change.")
    dataset.add_argument("--seed", type=int, default=42)

    # --- Corruption channel ---
    channel = parser.add_argument_group("corruption channel")
    channel.add_argument("--noise_std", type=float, default=0.5,
                         help="Default is comparable to the default ring's mode spacing "
                              "(2*ring_radius*sin(pi/ring_n_modes) ~= 3.06), not its jitter -- "
                              "too small and y ~= x, making any warm start look fine. See "
                              "data.py's module docstring. For --gt_distribution miffy (native "
                              "size 2.0 x 2.81), this default swamps the shape -- lower it or "
                              "raise --miffy_width/--miffy_height instead.")

    # --- Warmup Gaussian: the deliberately poor/uninformed prior under study ---
    warmup_dist = parser.add_argument_group("warmup gaussian")
    warmup_dist.add_argument("--warmup_mean", type=_floats, default=[0.0, 0.0],
                             help="Comma-separated, e.g. '8,8' for a badly MISLOCATED warm "
                                  "start. Default '0,0' is already poor in shape (unimodal vs. "
                                  "the multi-modal ground truth).")
    warmup_dist.add_argument("--warmup_cov", type=_floats, default=[1.0, 0.0, 0.0, 1.0],
                             help="Comma-separated 2x2 row-major, e.g. '1,0,0,1'.")
    warmup_dist.add_argument("--n_warmup_samples", type=int, default=10_000)

    # --- Warmup training / SCSI e-step / SCSI m-step / SCSI outer loop / viz (shared) ---
    add_scsi_args(parser, default_wandb_project="scsi-awgn-synthetic")

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    dataset = Config_Dataset_Synthetic(
        gt_distribution=args.gt_distribution,
        n_samples=args.n_samples,
        gt_mean=args.gt_mean,
        gt_cov=_as_2x2(args.gt_cov),
        ring_radius=args.ring_radius,
        ring_width=args.ring_width,
        ring_n_modes=args.ring_n_modes,
        ring_mode_std=args.ring_mode_std,
        miffy_width=args.miffy_width,
        miffy_height=args.miffy_height,
        miffy_thickness=args.miffy_thickness,
        noise_std=args.noise_std,
        warmup_mean=args.warmup_mean,
        warmup_cov=_as_2x2(args.warmup_cov),
        n_warmup_samples=args.n_warmup_samples,
        seed=args.seed,
    )

    warmup, scsi_config, viz = scsi_configs_from_args(args)

    return Config(dataset=dataset, warmup=warmup, scsi=scsi_config, viz=viz)
