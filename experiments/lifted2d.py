from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import dataclasses
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from distribution import DataDistribution, select_distribution, AVAILABLE_DISTRIBUTIONS, Gaussian
from forward import ForwardModel
from interpolant import LinearInterpolant
from model import MLP
from scsi_trainer_lifted import SCSIConfig, SCSITrainer
from inference import ODE, EulerSimulator, select_schedule, AVAILABLE_SCHEDULES

class RandomMarginalize(ForwardModel):
    '''
    Given a 2d vector, randomly picks a dimension and returns the value of that dimension.
    '''

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: (batch_size, 2) tensor of 2d vectors.
        Returns:
            (batch_size, 1) tensor of the value of the randomly chosen dimension.
        '''
        batch_size = x.shape[0]
        indices = torch.randint(0, 2, (batch_size,), device=x.device)
        output = x[torch.arange(batch_size, device=x.device), indices]

        return output.unsqueeze(1)

def _auto_lims(arrays, margin: float = 0.15):
    """Compute shared (xmin, xmax, ymin, ymax) from a list of (N, 2) arrays."""
    all_pts = np.concatenate(arrays, axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def plot_scsi_distributions(
    clean_samples, restored_samples,
    dist_name: str = "",
):
    """Side-by-side: true data, corrupted observations, SCSI restoration.

    Returns the matplotlib Figure so the caller can save / log it directly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    xmin, xmax, ymin, ymax = _auto_lims(
        [clean_samples, restored_samples]
    )

    for ax, data, title in zip(axes,
        [clean_samples, restored_samples],
        [f'True data ({dist_name})', 'SCSI restored'],
    ):
        ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="SCSI 2-D synthetic experiment")
    parser.add_argument("--distribution", type=str, default="checkerboard",
                        choices=AVAILABLE_DISTRIBUTIONS,
                        help=f"Target distribution (default: checkerboard). "
                             f"Options: {', '.join(AVAILABLE_DISTRIBUTIONS)}")
    parser.add_argument("--mode", type=str, default="ode",
                        choices=["ode", "sde"],
                        help="Interpolant mode: 'ode' (LinearInterpolant, drift only) "
                             "or 'sde' (TrigInterpolant, drift + denoiser). Default: ode")
    parser.add_argument("--gamma_0", type=float, default=0.05,
                        help="Noise scale for TrigInterpolant in SDE mode (defaultar: 1.0)")
    parser.add_argument("--denoiser_weight", type=float, default=1.0,
                        help="Weight of denoiser loss in SDE training (default: 1.0)")
    parser.add_argument("--outer_iterations", type=int, default=int(1e4),
                        help="Number of outer-loop iterations K (paper: 20,000)")
    parser.add_argument("--inner_steps", type=int, default=1,
                        help="Number of inner SGD steps T_tr per outer iteration")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size per SGD step")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--transport_steps", type=int, default=64,
                        help="Number of ODE/SDE steps for backward transport")
    parser.add_argument("--schedule_type", type=str, default="uniform",
                        choices=AVAILABLE_SCHEDULES,
                        help="Time discretization schedule: 'uniform' or 'exponential' "
                             "(default: uniform)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="scsi",
                        help="W&B project name (default: scsi)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Make finite size dataset
    n_dataset = 10000
    clean_dist = select_distribution(args.distribution)
    clean_data = clean_dist.sample(n_dataset)
    forward_model = RandomMarginalize()
    corrupted_data = forward_model(clean_data)
    
    observed_distribution = DataDistribution(corrupted_data.to(device))

    dim_target   = 2
    dim_observed = 1

    auxiliary_distribution = Gaussian(dim=dim_target, scale=0.3, device=device)

    # Drift model
    drift_model = MLP(data_dim=dim_target, hidden_dim=128, max_period=2, conditional_dim=dim_observed).to(device)

    # Config
    config = SCSIConfig(
        outer_iterations=args.outer_iterations,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=500,
        p_mixture=0.9,
        schedule_type=args.schedule_type,
        num_transport_steps=args.transport_steps,
        num_resamples=2,
        log_every=10,
    )

    # ---- Weights & Biases ----
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                **vars(args),
                **dataclasses.asdict(config),
            },
        )
    
    # ---- Train ----
    print(f"\n{'='*60}")
    print(f"SCSI Training ({args.mode.upper()})  |  dist={args.distribution}  "
          f"K={config.outer_iterations}  T_tr={config.inner_steps}  B={config.batch_size}")
    print(f"{'='*60}\n")

    def _wandb_log(metrics: dict) -> None:
        """Forward metrics to wandb, using global_step as the explicit step."""
        wandb.log(
            {k: v for k, v in metrics.items() if k != "global_step"},
            step=metrics.get("global_step"),
        )

    if args.mode == "ode":
        trainer = SCSITrainer(
            observation_dist=observed_distribution,
            auxiliary_dist=auxiliary_distribution,
            forward_model=forward_model,
            interpolant=LinearInterpolant(base=observed_distribution, target=auxiliary_distribution),
            drift_model=drift_model,
            config=config,
            log_fn=_wandb_log if not args.no_wandb else None,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    train_start = time.time()
    losses = trainer.train()
    train_elapsed = time.time() - train_start
    print(f"Training time: {train_elapsed:.2f}s")

    # ---- Restore corrupted observations via backward transport ----
    print(f"\nRestoring observations via backward {args.mode.upper()} transport...")
    n_eval = 5000
    eval_corrupted = observed_distribution.sample(n_eval)
    w = auxiliary_distribution.sample(n_eval)
    drift_model.eval()
    with torch.no_grad():
        schedule = select_schedule(args.schedule_type)(args.transport_steps)
        if args.mode == "ode":
            ode = ODE(drift_model)
            simulator = EulerSimulator(ode)
            restored = simulator.solve_backwards(w, schedule, conditional=eval_corrupted)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    # ---- Plot: true vs corrupted vs restored ----
    eval_clean = clean_dist.sample(n_eval)
    results_fig = plot_scsi_distributions(
        eval_clean.cpu().numpy(),
        restored.cpu().numpy(),
        dist_name=args.distribution,
    )
    

    # ---- Log images to W&B and finish ----
    if not args.no_wandb:
        total_steps = config.outer_iterations * config.inner_steps
        wandb.log(
            {
                "scsi_results": wandb.Image(results_fig),
                "training_time_s": train_elapsed,
            },
            step=total_steps,
        )
        wandb.finish()
    plt.close(results_fig)

    print(f"\nDone! Check out the results on W&B.")


if __name__ == '__main__':
    main()
