"""
Self-Consistent Stochastic Interpolant (SCSI) - 2-D synthetic experiment.

Reproduces the AWGN experiment from Section 6.1 of the paper:
    - True data distribution:  selectable (checkerboard, spiral, …)
    - Forward model:           y = x + sigma * z   (AWGN channel)
    - We only see corrupted observations y ~ mu.
    - SCSI iteratively learns a transport map that inverts the channel.

Usage:
    python main.py --distribution checkerboard                   # default (ODE)
    python main.py --distribution spiral --sigma 0.1
    python main.py --mode sde --distribution checkerboard        # SDE mode
    python main.py --mode sde --gamma_0 0.05 --distribution spiral
"""

import argparse
import dataclasses
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from distribution import DataDistribution, select_distribution, AVAILABLE_DISTRIBUTIONS
from forward import AWGNForwardModel
from interpolant import LinearInterpolant, TrigInterpolant
from model import MLP
from inference import ODE, EulerSimulator, SDE, EulerMaruyamaSimulator, select_schedule, AVAILABLE_SCHEDULES
from scsi_trainer import SCSITrainer, SCSITrainerSDE, SCSIConfig


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _auto_lims(arrays, margin: float = 0.15):
    """Compute shared (xmin, xmax, ymin, ymax) from a list of (N, 2) arrays."""
    all_pts = np.concatenate(arrays, axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def plot_scsi_distributions(
    clean_samples, corrupted_samples, restored_samples,
    sigma: float, dist_name: str = "",
):
    """Side-by-side: true data, corrupted observations, SCSI restoration.

    Returns the matplotlib Figure so the caller can save / log it directly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    xmin, xmax, ymin, ymax = _auto_lims(
        [clean_samples, corrupted_samples, restored_samples]
    )

    for ax, data, title in zip(axes,
        [clean_samples, corrupted_samples, restored_samples],
        [f'True data ({dist_name})', f'Observations (sigma={sigma})', 'SCSI restored'],
    ):
        ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="AWGN noise std (default: 0.05)")
    parser.add_argument("--gamma_0", type=float, default=0.05,
                        help="Noise scale for TrigInterpolant in SDE mode (default: 1.0)")
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
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="scsi",
                        help="W&B project name (default: scsi)")
    args = parser.parse_args()

    # ---- Setup ----
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    torch.manual_seed(args.seed)

    # ---- Device selection ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ---- Ground-truth clean distribution (never used during training) ----
    clean_dist = select_distribution(args.distribution)

    # ---- Forward model (black-box corruption) ----
    forward_model = AWGNForwardModel(sigma=args.sigma)

    # ---- Generate corrupted dataset ----
    n_dataset = 50_000
    clean_data = clean_dist.sample(n_dataset)
    with torch.no_grad():
        corrupted_data = forward_model(clean_data)
    obs_dist = DataDistribution(corrupted_data.to(device))

    # ---- Interpolant (schedule only, no clean data) ----
    #      base/target on the interpolant don't matter for SCSI training –
    #      we pass a dummy.  The SCSI trainer constructs its own (x0, x1).
    if args.mode == "ode":
        interpolant = LinearInterpolant(base=obs_dist, target=obs_dist)
    else:
        interpolant = TrigInterpolant(base=obs_dist, target=obs_dist, gamma_0=args.gamma_0)

    # ---- Drift model ----
    # Paper Table 3: max positional embedding = 2 for the SI (not 10000)
    drift_model = MLP(data_dim=2, hidden_dim=256, max_period=2).to(device)

    # ---- Denoiser model (SDE mode only) ----
    denoiser_model = None
    if args.mode == "sde":
        denoiser_model = MLP(data_dim=2, hidden_dim=256, max_period=2).to(device)

    # ---- SCSI config ----
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
    if args.wandb:
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
    print(f"SCSI Training ({args.mode.upper()})  |  dist={args.distribution}  sigma={args.sigma}  "
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
            observation_dist=obs_dist,
            forward_model=forward_model,
            interpolant=interpolant,
            drift_model=drift_model,
            config=config,
            log_fn=_wandb_log if args.wandb else None,
        )
    else:
        trainer = SCSITrainerSDE(
            observation_dist=obs_dist,
            forward_model=forward_model,
            interpolant=interpolant,
            drift_model=drift_model,
            denoiser_model=denoiser_model,
            config=config,
            denoiser_weight=args.denoiser_weight,
            log_fn=_wandb_log if args.wandb else None,
        )

    train_start = time.time()
    losses = trainer.train()
    train_elapsed = time.time() - train_start
    print(f"Training time: {train_elapsed:.2f}s")

    # ---- Restore corrupted observations via backward transport ----
    print(f"\nRestoring observations via backward {args.mode.upper()} transport...")
    n_eval = 5000
    eval_corrupted = obs_dist.sample(n_eval)
    drift_model.eval()
    with torch.no_grad():
        schedule = select_schedule(args.schedule_type)(args.transport_steps)
        if args.mode == "ode":
            ode = ODE(drift_model)
            simulator = EulerSimulator(ode)
            restored = simulator.solve_backwards(eval_corrupted, schedule)
        else:
            denoiser_model.eval()
            # Paper Appendix D.2: epsilon_t = gamma_t
            sde = SDE(drift_model, denoiser_model, interpolant,
                       noise_schedule=interpolant.gamma)
            simulator = EulerMaruyamaSimulator(sde)
            restored = simulator.solve_backwards(eval_corrupted, schedule)

    # ---- Plot: true vs corrupted vs restored ----
    eval_clean = clean_dist.sample(n_eval)
    results_fig = plot_scsi_distributions(
        eval_clean.cpu().numpy(),
        eval_corrupted.cpu().numpy(),
        restored.cpu().numpy(),
        sigma=args.sigma,
        dist_name=args.distribution,
    )
    

    # ---- Log images to W&B and finish ----
    if args.wandb:
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
