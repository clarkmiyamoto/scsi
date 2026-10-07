import functools

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .corruption import corruption_channel, build_pair_sample # black box forward model
from .data import build_observations, build_warmup, build_viz_pool
from scsi_new.distribution import IsotropicGaussian
from .model import ConditionalVelocityCryoET
from scsi_new.scsi import EMA, estep, mstep_lifted
from .args import parse_args, config_from_args
from .wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw


if __name__ == "__main__":
    args = parse_args()
    config = config_from_args(args)

    wandb.init(project=config.viz.wandb_project, name=config.viz.wandb_run_name, config=vars(args))

    torch.manual_seed(config.scsi.seed)
    device = torch.device(config.scsi.device)

    def build_model():
        return ConditionalVelocityCryoET(
            image_size=config.dataset.image_size, arch=config.arch, patch_size=config.patch_size,
        ).to(device)

    def build_optimizer(net, train_config, n_steps):
        # AdamW + cosine from train_config.lr down to lr_eta_min over n_steps.
        optimizer = AdamW(net.parameters(),
                          lr=train_config.lr,
                          weight_decay=train_config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=config.scsi.lr_eta_min)
        return optimizer, scheduler

    # Model & optimizer
    model = build_model()
    base_dist = IsotropicGaussian(
        shape=(1, config.dataset.image_size, config.dataset.image_size), device=device,
    )
    if config.student_init == "fresh":
        # Every model trains on its own schedule: the warmup gets --warmup_lr and a cosine over
        # just the warmup steps, so warmup and M-step settings don't leak into each other.
        optimizer, scheduler = build_optimizer(model, config.warmup, config.warmup.n_steps_train)
        ema = EMA(model, decay=config.warmup.ema)
    else:
        # One optimizer at --mstep_lr and one cosine spanning warmup + every M-step
        # (--warmup_lr / --warmup_weight_decay / --warmup_ema are unused in this mode).
        total_train_steps = config.warmup.n_steps_train + config.scsi.num_scsi_steps * config.scsi.mstep.n_steps_train
        optimizer, scheduler = build_optimizer(model, config.scsi.mstep, total_train_steps)
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
    # (x̂) -> (target, ŷ). --lift makes target = R·x̂ for a fresh independent random SO(2) R.
    pair_sample = build_pair_sample(corruption_channel_bound, lift=config.lift)

    # Run SCSI algorithm
    for k in range(config.scsi.num_scsi_steps):
        # E-step: Sample from the posterior distribution of latent variables given observations
        posterior_samples = estep(
            model, base_dist, observations, pair_sample, config.scsi.estep
        )

        # --student_init fresh: the teacher's only job was the E-step above. Discard it and train
        # a newly initialized student (new optimizer, EMA, and cosine LR restarted over this M-step)
        # instead of fine-tuning the teacher in place.
        if config.student_init == "fresh":
            model = build_model()
            optimizer, scheduler = build_optimizer(model, config.scsi.mstep, config.scsi.mstep.n_steps_train)
            ema = EMA(model, decay=config.scsi.mstep.ema)

        # M-step: Update model parameters to maximize expected log-likelihood
        mstep_lifted(
            model, base_dist, posterior_samples, optimizer, config.scsi.mstep,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="train",
        )

        if (k + 1) % config.viz.every == 0:
            log_all_panels(em_step=k + 1)

    wandb.finish()
