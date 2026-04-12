"""
Lifted SCSI experiment on synthetic cryo-EM projections.

This script uses `src/data_fake.py` projections as observations in Y and
trains the lifted trainer (`src/scsi_trainer_lifted.py`) in an X != Y setup:

    X: low-resolution image space  (x_side x x_side)
    Y: projection image space      (y_side x y_side)

Forward model used during training:
    F: X -> Y via bilinear upsampling (+ optional additive noise)
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_fake import OBJECT_REGISTRY, generate_dataset
from distribution import DataDistribution
from forward import ForwardModel
from interpolant import LinearInterpolant
from model import MLP
from scsi_trainer_lifted import SCSIConfig, SCSITrainer
from inference import ODE, EulerSimulator, select_schedule


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_observations(
    data_root: Path,
    object_name: str,
    resolution: int,
    num_projections: int,
    data_noise_std: float,
    seed: int,
    max_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    obj_dir = data_root / object_name
    proj_file = obj_dir / "projections.npy"

    if not proj_file.exists():
        print(f"Dataset missing at {proj_file}; generating with data_fake.py...")
        generate_dataset(
            object_names=[object_name],
            resolution=resolution,
            num_projections=num_projections,
            noise_std=data_noise_std,
            seed=seed,
            output_dir=str(data_root),
        )

    projections = torch.from_numpy(np.load(proj_file)).float()  # (K, H, W)
    if max_samples > 0:
        projections = projections[:max_samples]

    # Per-image min-max normalization for stable training.
    mins = projections.amin(dim=(1, 2), keepdim=True)
    maxs = projections.amax(dim=(1, 2), keepdim=True)
    projections = (projections - mins) / (maxs - mins + 1e-8)

    y_side = projections.shape[-1]
    y_flat = projections.flatten(1).to(device)
    return y_flat, y_side



def _plot_volume_figure(volumes: list[torch.Tensor]) -> plt.Figure:
    """Render estimated (and optional true) 3D volumes using voxel plots."""
    fig = plt.figure(figsize=(12, 6))
    for i, vol in enumerate(volumes):
        ax = fig.add_subplot(1, len(volumes), i + 1, projection="3d")
        vol_np = vol.numpy()
        threshold = float(np.quantile(vol_np, 0.85))
        mask = vol_np > threshold
        ax.voxels(mask, facecolors="tab:blue", edgecolor="k", linewidth=0.1)
        ax.set_axis_off()

    fig.tight_layout()
    return fig


def _volume_to_plotly_figure(volume: torch.Tensor, quantile: float = 0.85):
    """Create an interactive 3D point-cloud view of high-density voxels."""
    import plotly.graph_objects as go

    vol_np = volume.detach().cpu().numpy()
    threshold = float(np.quantile(vol_np, quantile))
    mask = vol_np > threshold
    x, y, z = np.where(mask)
    values = vol_np[mask] if np.any(mask) else np.array([0.0], dtype=np.float32)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x if np.any(mask) else np.array([0]),
                y=y if np.any(mask) else np.array([0]),
                z=z if np.any(mask) else np.array([0]),
                mode="markers",
                marker={
                    "size": 3,
                    "opacity": 0.85,
                    "color": values,
                    "colorscale": "Blues",
                },
            )
        ]
    )
    fig.update_layout(
        title="Reconstructed 3D Volume",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Lifted SCSI on fake cryo-EM projections")
    parser.add_argument("--object", type=str, default="cube", choices=list(OBJECT_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--num_projections", type=int, default=128)
    parser.add_argument("--data_noise_std", type=float, default=0.02)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--x_side", type=int, default=16, help="Low-res side length for X")
    parser.add_argument("--forward_noise_std", type=float, default=0.0)
    parser.add_argument("--outer_iterations", type=int, default=10000)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--transport_steps", type=int, default=32)
    parser.add_argument("--num_resamples", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--volume_fit_steps", type=int, default=80)
    parser.add_argument("--volume_fit_batch_size", type=int, default=8)
    parser.add_argument("--volume_fit_lr", type=float, default=5e-2)
    parser.add_argument("--volume_fit_views", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="experiments/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging (default: enabled). Use --no-wandb to disable.",
    )
    parser.add_argument("--wandb_project", type=str, default="scsi", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device()
    print(f"Using device: {device}")

    y_flat, y_side = _prepare_observations(
        data_root=Path(args.data_root),
        object_name=args.object,
        resolution=args.resolution,
        num_projections=args.num_projections,
        data_noise_std=args.data_noise_std,
        seed=args.seed,
        max_samples=args.max_samples,
        device=device,
    )
    if args.x_side >= y_side:
        raise ValueError(f"x_side must be < y_side for lifted X!=Y; got x_side={args.x_side}, y_side={y_side}")

    obs_dist = DataDistribution(y_flat)
    forward_model = UpsampleForwardModel(
        x_side=args.x_side,
        y_side=y_side,
        noise_std=args.forward_noise_std,
    )

    dx = args.x_side * args.x_side
    dy = y_side * y_side
    drift_dim = dx + dy

    interpolant = LinearInterpolant(base=obs_dist, target=obs_dist)
    drift_model = MLP(data_dim=drift_dim, hidden_dim=args.hidden_dim, max_period=2).to(device)

    config = SCSIConfig(
        data_dim=dx,
        obs_dim=dy,
        outer_iterations=args.outer_iterations,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=min(100, args.outer_iterations * args.inner_steps // 4),
        p_mixture=0.9,
        schedule_type="uniform",
        num_transport_steps=args.transport_steps,
        num_resamples=args.num_resamples,
        log_every=50,
    )

    print(
        f"Training lifted SCSI: object={args.object}  X={args.x_side}x{args.x_side}  "
        f"Y={y_side}x{y_side}  samples={len(y_flat)}"
    )

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                **dataclasses.asdict(config),
            },
        )

    def _wandb_log(metrics: dict) -> None:
        if not args.wandb:
            return
        wandb.log(
            {k: v for k, v in metrics.items() if k != "global_step"},
            step=metrics.get("global_step"),
        )

    trainer = SCSITrainer(
        observation_dist=obs_dist,
        forward_model=forward_model,
        interpolant=interpolant,
        drift_model=drift_model,
        config=config,
        log_fn=_wandb_log if args.wandb else None,
    )

    t0 = time.time()
    losses = trainer.train()
    elapsed = time.time() - t0
    steps = len(losses)

    # ---- Restore 3D object from 2D observations via backward transport ----
    print(f"\nRestoring 3D object from 2D observations via backward transport...")
    n_eval = 3
    eval_corrupted = obs_dist.sample(n_eval)
    drift_model.eval()
    with torch.no_grad():
        schedule = select_schedule(config.schedule_type)(args.transport_steps)
        ode = ODE(drift_model)
        simulator = EulerSimulator(ode)
        restored = simulator.solve_backwards(eval_corrupted, schedule)

    # ---- Plot: true vs corrupted vs restored ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_image_path = output_dir / f"{args.object}_recon3d.png"

    volumes_to_plot = [r.detach().cpu() for r in restored]
    results_fig = _plot_volume_figure(volumes_to_plot)
    plt.savefig(volume_image_path)
    plt.show()
    plt.close(results_fig)

    if args.wandb:
        log_data = {
            "train_time_s": elapsed,
            "time_per_step_ms": 1000.0 * elapsed / max(steps, 1),
            "final_loss": losses[-1] if len(losses) > 0 else float("nan"),
        }
        if volume_image_path.exists():
            log_data["reconstructed_3d"] = wandb.Image(str(volume_image_path))
        try:
            interactive_fig = _volume_to_plotly_figure(restored[0])
            log_data["reconstructed_3d_interactive"] = wandb.Plotly(interactive_fig)
        except ImportError:
            print("Plotly is not installed; skipping interactive 3D W&B logging.")
        wandb.log(log_data, step=steps)
        wandb.finish()


if __name__ == "__main__":
    main()
