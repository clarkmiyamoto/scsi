import sys
from pathlib import Path

# This file lives at scsi_new/experiments/cryoet_mnist3d/main.py. scsi.py, si.py, ode.py, and
# distribution.py live flat at scsi_new/ and import each other with bare imports (e.g. scsi.py
# does `from si import ...`), so scsi_new/ must be on sys.path for those to resolve.
# corruption.py/data.py/args.py/model.py/wandb_logging.py need no such fix: Python already adds
# a directly-run script's own directory to sys.path[0].
SCSI_NEW_ROOT = Path(__file__).resolve().parents[2]
if str(SCSI_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SCSI_NEW_ROOT))

import functools

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset

from corruption import corruption_channel, build_pair_sample  # black box forward model
from data import build_observations, build_warmup, build_viz_pool
from distribution import IsotropicGaussian
from model import ConditionalVelocityCryoET3D
from scsi import EMA, ResampledPairs, estep, mstep_lifted
from args import parse_args, config_from_args
from wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw


if __name__ == "__main__":
    args = parse_args()
    config = config_from_args(args)

    wandb.init(project=config.viz.wandb_project, name=config.viz.wandb_run_name, config=vars(args))

    torch.manual_seed(config.scsi.seed)
    device = torch.device(config.scsi.device)

    V = config.dataset.vol_size

    # Model & optimizer
    model = ConditionalVelocityCryoET3D(
        vol_size=V,
        num_tilts=config.dataset.num_tilts,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
    ).to(device)
    base_dist = IsotropicGaussian(shape=(1, V, V, V), device=device)
    optimizer = AdamW(model.parameters(),
                      lr=config.scsi.mstep.lr,
                      weight_decay=config.scsi.mstep.weight_decay)

    total_train_steps = config.warmup.n_steps_train + config.scsi.num_scsi_steps * config.scsi.mstep.n_steps_train
    scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=config.scsi.lr_eta_min)
    ema = EMA(model, decay=config.scsi.mstep.ema)

    # Load observations: extruded-MNIST volumes run through the 3D->2D tilt-series channel
    observations = build_observations(config.dataset)

    # Visualization setup
    global_step = [0]
    viz_pool = build_viz_pool(config.dataset, n_pool=config.viz.n_pool, viz_seed=config.viz.seed)
    fixed = {k: v[:config.viz.n_display] for k, v in viz_pool.items()}

    def log_all_panels(em_step):
        rand = random_draw(viz_pool, config.dataset, config.viz.n_display)
        for panel_name, src in [("fixed", fixed), ("random", rand)]:
            log_reconstruction_grid(
                model, src["x0"], src["y"], src["x_gt"],
                config.dataset, config.scsi.estep.n_steps_sampling,
                em_step, global_step[0], panel_name, device,
            )
            log_trajectory_grid(
                model, src["x0"], src["y"],
                config.scsi.estep.n_steps_sampling, config.viz.n_snapshots,
                config.viz.n_trajectory_rows, em_step, global_step[0], panel_name, device,
            )

    # ŷ = F(x̂) -- for the warm start below AND the E-step -- must use the SAME channel params as
    # build_observations, yet still draw a fresh random SO(3) mount + tilt series on every call.
    corruption_channel_bound = functools.partial(
        corruption_channel,
        num_tilts=config.dataset.num_tilts,
        tilt_increment_deg=config.dataset.tilt_increment_deg,
        noise_std=config.dataset.noise_std,
        tilt_axis=config.dataset.tilt_axis,
    )
    # (x̂) -> (target, ŷ). --lift makes target = R·x̂ for a fresh independent random SO(3) R.
    pair_sample = build_pair_sample(corruption_channel_bound, lift=config.lift)

    # Warm start on RESAMPLED (target, ŷ) pairs generated on the fly from the pseudoinverse
    # recons X alone -- NOT build_warmup's frozen (x_hat, y_obs) pairs. X = pseudoinverse.py's
    # filtered backprojection of the observed tilt series, renormed to [-1, 1] (build_warmup);
    # we keep only that tensor and discard its paired y_obs. ResampledPairs then re-draws, per
    # __getitem__, ŷ = F(X) through a fresh random mount + tilt series and (under --lift)
    # target = R·X for a fresh independent SO(3) R -- the SAME pair_sample the E-step uses, so
    # the warm start already trains on the rotation-symmetrized (R·X, F(X)) objective instead of
    # one fixed (X, y_obs) draw. Cost: ResampledPairs runs the channel per-sample inside the
    # dataloader (num_workers=0 in mstep_lifted), so warmup steps slow down -- same tradeoff as
    # main_supervised.py's --resample_channel.
    warmup_x = build_warmup(observations, config.dataset).tensors[0]
    warmup_pairs = ResampledPairs(TensorDataset(warmup_x), pair_sample)
    mstep_lifted(
        model, base_dist, warmup_pairs, optimizer, config.warmup,
        scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="warmup",
    )
    log_all_panels(em_step=0)

    # Run SCSI algorithm
    for k in range(config.scsi.num_scsi_steps):
        # E-step: sample from the posterior over latent clean volumes given the observations
        posterior_samples = estep(
            model, base_dist, observations, pair_sample, config.scsi.estep
        )

        # M-step: update model parameters to maximize expected log-likelihood
        mstep_lifted(
            model, base_dist, posterior_samples, optimizer, config.scsi.mstep,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="train",
        )

        if (k + 1) % config.viz.every == 0:
            log_all_panels(em_step=k + 1)

    wandb.finish()
