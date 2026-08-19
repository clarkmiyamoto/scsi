import sys
from pathlib import Path

# This file lives at scsi_new/experiments/cryoet_mnist/main.py. scsi.py,
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

from corruption import corruption_channel # black box forward model
from data import build_observations, build_warmup, build_viz_pool
from distribution import IsotropicGaussian
from model import ConditionalVelocityCryoET
from scsi import EMA, estep, mstep_lifted
from args import parse_args, config_from_args
from wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw


if __name__ == "__main__":
    args = parse_args()
    config = config_from_args(args)

    wandb.init(project=config.viz.wandb_project, name=config.viz.wandb_run_name, config=vars(args))

    torch.manual_seed(config.scsi.seed)
    device = torch.device(config.scsi.device)

    # Model & optimizer
    model = ConditionalVelocityCryoET(
        image_size=config.dataset.image_size, arch=config.arch, patch_size=config.patch_size,
    ).to(device)
    base_dist = IsotropicGaussian(
        shape=(1, config.dataset.image_size, config.dataset.image_size), device=device,
    )
    optimizer = AdamW(model.parameters(),
                        lr=config.scsi.mstep.lr,
                        weight_decay=config.scsi.mstep.weight_decay)

    total_train_steps = config.warmup.n_steps_train + config.scsi.num_scsi_steps * config.scsi.mstep.n_steps_train
    scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=config.scsi.lr_eta_min)
    ema = EMA(model, decay=config.scsi.mstep.ema)

    # Load observations from MNIST dataset
    observations = build_observations(config.dataset)

    # Visualization setup
    global_step = [0]
    viz_pool = build_viz_pool(config.dataset, n_pool=config.viz.n_pool, viz_seed=config.viz.seed)
    fixed = {k: v[:config.viz.n_display] for k, v in viz_pool.items()}

    def log_all_panels(em_step):
        rand = random_draw(viz_pool, config.dataset, config.viz.n_display)
        for panel_name, src in [("fixed", fixed), ("random", rand)]:
            log_reconstruction_grid(
                model, src["x0"], src["theta"], src["y"], src["x_gt"],
                config.dataset.noise_std, config.scsi.estep.n_steps_sampling,
                em_step, global_step[0], panel_name, device,
            )
            log_trajectory_grid(
                model, src["x0"], src["theta"], src["y"],
                config.scsi.estep.n_steps_sampling, config.viz.n_snapshots,
                config.viz.n_trajectory_rows, em_step, global_step[0], panel_name, device,
            )

    # Warmup model on (x_hat, y) pairs from pseudoinverse
    observations_pseudoinverse = build_warmup(observations, config.dataset)
    mstep_lifted(
        model, base_dist, observations_pseudoinverse, optimizer, config.warmup,
        scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="warmup",
    )
    log_all_panels(em_step=0)

    # ŷ = F(x̂) must use the SAME channel params, yet still be random
    corruption_channel_bound = functools.partial(
        corruption_channel,
        num_tilts=config.dataset.num_tilts,
        tilt_increment_deg=config.dataset.tilt_increment_deg,
        noise_std=config.dataset.noise_std,
    )

    # Run SCSI algorithm
    for k in range(config.scsi.num_scsi_steps):
        # E-step: Sample from the posterior distribution of latent variables given observations
        posterior_samples = estep(
            model, base_dist, observations, corruption_channel_bound, config.scsi.estep
        )

        # M-step: Update model parameters to maximize expected log-likelihood
        mstep_lifted(
            model, base_dist, posterior_samples, optimizer, config.scsi.mstep,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="train",
        )

        if (k + 1) % config.viz.every == 0:
            log_all_panels(em_step=k + 1)

    wandb.finish()
