import sys
from pathlib import Path

# This file lives at scsi_new/experiments/awgn_synthetic/main.py. scsi.py,
# si.py, ode.py, and distribution.py live flat at scsi_new/ and import each
# other with bare imports (e.g. scsi.py does `from si import ...`), so
# scsi_new/ must be on sys.path for those to resolve. corruption.py/data.py
# need no such fix: Python already adds a directly-run script's own
# directory to sys.path[0].
SCSI_NEW_ROOT = Path(__file__).resolve().parents[2]
if str(SCSI_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SCSI_NEW_ROOT))

import functools

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from corruption import corruption_channel  # black box forward model
from data import build_observations, build_warmup, build_viz_pool, build_reference_pool, get_mode_centers
from distribution import IsotropicGaussian
from model import ConditionalVelocityMLP, POINT_DIM
from scsi import EMA, estep, mstep_lifted
from args import parse_args, config_from_args
from wandb_logging import log_distribution_scatter, log_distribution_metrics, log_trajectory_lines, random_draw

"""
SCSI on a synthetic 2D point distribution under an AWGN channel (y = x + noise), warm-started on
a user-specified -- and potentially POOR / mismatched -- Gaussian prior instead of anything
derived from the observations. The default ground truth (an 8-mode ring of Gaussians) and default
warm start (a single Gaussian centered at the origin) are deliberately mismatched in shape, so
this studies whether the SCSI EM loop can still recover the full multi-modal ground-truth
distribution from a poor/uninformed initialization -- see data.py's module docstring for why
`--noise_std` (relative to the ground truth's mode spacing) is the knob that decides whether this
is a hard question or a trivial one.

Structurally mirrors experiments/mra_mnist/main.py (same channel shape: corruption_channel(x,
noise_std)). Two departures:
  - build_warmup() here takes only `config`, not `observations` -- the warm-start x_hat is drawn
    independent of the data, that's the whole point (see data.py).
  - Points are carried as (B, 2, 1, 1) tensors -- "1x1 images with 2 channels" -- everywhere, so
    the shared scsi.py/si.py/ode.py machinery (built for image-shaped (B, C, H, W) tensors) works
    completely unmodified. See model.py's docstring for why that shape, specifically, matters.
"""

if __name__ == "__main__":
    args = parse_args()
    config = config_from_args(args)

    wandb.init(project=config.viz.wandb_project, name=config.viz.wandb_run_name, config=vars(args))

    torch.manual_seed(config.scsi.seed)
    device = torch.device(config.scsi.device)

    # Model & optimizer
    model = ConditionalVelocityMLP(dim=POINT_DIM).to(device)
    base_dist = IsotropicGaussian(shape=(POINT_DIM, 1, 1), device=device)
    optimizer = AdamW(model.parameters(),
                      lr=config.scsi.mstep.lr,
                      weight_decay=config.scsi.mstep.weight_decay)

    total_train_steps = config.warmup.n_steps_train + config.scsi.num_scsi_steps * config.scsi.mstep.n_steps_train
    scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=config.scsi.lr_eta_min)
    ema = EMA(model, decay=config.scsi.mstep.ema)

    # Ground-truth observations: y = x + noise, x drawn from the configured target distribution.
    observations = build_observations(config.dataset)

    # Fixed, clean (never trained on) reference pool + analytic mode centers, for the
    # distribution-recovery scatter/metrics below.
    n_ref = min(config.dataset.n_samples, 5_000)
    x_gt_ref = build_reference_pool(config.dataset, n_ref)
    mode_centers = get_mode_centers(config.dataset)

    # Small fixed/random pools for the cheap per-example trajectory panel.
    global_step = [0]
    viz_pool = build_viz_pool(config.dataset, n_pool=config.viz.n_pool, viz_seed=config.viz.seed)
    fixed = {k: v[:config.viz.n_display] for k, v in viz_pool.items()}

    def log_all_panels(em_step, x_hat_pool, warmup_ref):
        log_distribution_scatter(x_hat_pool, x_gt_ref, warmup_ref, em_step, global_step[0])
        log_distribution_metrics(x_hat_pool, x_gt_ref, mode_centers, em_step, global_step[0])
        rand = random_draw(viz_pool, config.dataset, config.viz.n_display)
        for panel_name, src in [("fixed", fixed), ("random", rand)]:
            log_trajectory_lines(
                model, src["x0"], src["y"],
                config.scsi.estep.n_steps_sampling, config.viz.n_snapshots,
                config.viz.n_trajectory_rows, em_step, global_step[0], panel_name, device,
            )

    # Warmup model on (x_hat, y_hat) pairs from the user-specified (possibly poor) Gaussian --
    # NOT derived from `observations`. This is the deliberately-bad initialization under study.
    observations_warmup = build_warmup(config.dataset)
    warmup_ref = observations_warmup.tensors[0][:n_ref]  # fixed baseline shown in every panel
    mstep_lifted(
        model, base_dist, observations_warmup, optimizer, config.warmup,
        scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="warmup",
    )

    # ŷ = F(x̂) must use the SAME channel params, yet still be random
    corruption_channel_bound = functools.partial(corruption_channel, noise_std=config.dataset.noise_std)

    # No real E-step has run yet after warmup -- sample one ourselves (same call as the loop's
    # E-step below) purely to have a proposal pool to log at em_step=0.
    warmup_posterior = estep(model, base_dist, observations, corruption_channel_bound, config.scsi.estep)
    log_all_panels(em_step=0, x_hat_pool=warmup_posterior.tensors[0], warmup_ref=warmup_ref)

    # Run SCSI algorithm
    for k in range(config.scsi.num_scsi_steps):
        # E-step: sample from the posterior distribution of latent variables given observations
        posterior_samples = estep(
            model, base_dist, observations, corruption_channel_bound, config.scsi.estep
        )

        # M-step: update model parameters to maximize expected log-likelihood
        mstep_lifted(
            model, base_dist, posterior_samples, optimizer, config.scsi.mstep,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="train",
        )

        if (k + 1) % config.viz.every == 0:
            # Reuse this iteration's own E-step proposals -- no extra integration needed.
            log_all_panels(em_step=k + 1, x_hat_pool=posterior_samples.tensors[0], warmup_ref=warmup_ref)

    wandb.finish()
