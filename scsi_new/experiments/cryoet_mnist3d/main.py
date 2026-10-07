import functools
import math
import os
import sys
import time

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset

from .corruption import corruption_channel, build_pair_sample  # black box forward model
from .data import build_observations, build_warmup, build_viz_pool
from scsi_new.distribution import IsotropicGaussian
from .eval_metrics import AlignedCorrelation, calibrate, evaluate
from .model import ConditionalVelocityCryoET3D
from scsi_new.train_utils import atomic_save, check_args, make_lr_lambda, rng_state, set_rng_state
from scsi_new.scsi import EMA, ResampledPairs, estep, mstep_lifted
from .args import parse_args, config_from_args
from .wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw


# Args that fix the observations, the warm start and the network. --load_warmup_ckpt refuses a
# checkpoint that disagrees with this run on any of them.
_WARMUP_CKPT_KEYS = (
    "n_images_per_class", "vol_size", "digit_scale", "inplane_size", "depth_extent",
    "digit_classes", "seed", "train", "num_tilts", "tilt_increment_deg", "noise_std", "tilt_axis",
    "filtered", "filter_type", "block_out_channels", "layers_per_block", "lift",
    "warmup_n_steps_train",
)
# --resume also refuses a latest.pt whose EM-phase args differ: a continuation job must run the
# same experiment. num_scsi_steps may change, so a finished run can be extended.
_RESUME_KEYS = _WARMUP_CKPT_KEYS + (
    "estep_num_samples", "estep_batch_size", "estep_n_steps_sampling",
    "mstep_n_steps_train", "mstep_batch_size", "mstep_lr", "mstep_weight_decay", "mstep_ema",
    "mstep_interpolant_style", "eta_min", "lr_schedule", "lr_horizon_scsi_steps",
    "sample_with_ema", "student_init",
)
# Args added after checkpoints already existed: a checkpoint without the key ran with this value.
_ARG_DEFAULTS = {"student_init": "teacher"}


if __name__ == "__main__":
    args = parse_args()
    config = config_from_args(args)

    # --resume: continue from --ckpt_dir/latest.pt when it exists, else start normally, so one
    # SBATCH file serves both the first job and every chained continuation. Read before
    # wandb.init, which reopens the checkpoint's run.
    resume = None
    latest = os.path.join(config.ckpt_dir, "latest.pt") if config.ckpt_dir else None
    if config.resume and os.path.exists(latest):
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        if "wandb_id" not in resume:
            raise ValueError(f"{latest} predates --resume (no wandb_id / obs_checksum)")
        check_args(resume["args"], args, _RESUME_KEYS, latest, arg_defaults=_ARG_DEFAULTS)
        if resume["em_step"] >= config.scsi.num_scsi_steps:
            print(f"{latest} is already at EM step {resume['em_step']}/"
                  f"{config.scsi.num_scsi_steps}; nothing to do.", flush=True)
            sys.exit(0)
        print(f"resuming from {latest} at EM step {resume['em_step']}", flush=True)

    if resume is None:
        wandb.init(project=config.viz.wandb_project, name=config.viz.wandb_run_name,
                   config=vars(args))
    else:
        # allow_val_change: extending a run (a larger --num_scsi_steps) changes a logged value.
        wandb.init(project=config.viz.wandb_project, name=config.viz.wandb_run_name,
                   id=resume["wandb_id"], resume="allow")
        wandb.config.update(vars(args), allow_val_change=True)

    torch.manual_seed(config.scsi.seed)
    device = torch.device(config.scsi.device)

    V = config.dataset.vol_size

    def build_model():
        return ConditionalVelocityCryoET3D(
            vol_size=V,
            num_tilts=config.dataset.num_tilts,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
        ).to(device)

    def build_optimizer(net, train_config, n_steps):
        # --student_init fresh: AdamW + a cosine from train_config.lr down to --eta_min over
        # n_steps, private to this one model (make_lr_lambda's cosine with no EM steps).
        optimizer = AdamW(net.parameters(),
                          lr=train_config.lr,
                          weight_decay=train_config.weight_decay)
        scheduler = LambdaLR(optimizer, make_lr_lambda(
            "cosine", warmup_steps=n_steps, mstep_steps=0, horizon_scsi_steps=0,
            floor=config.scsi.lr_eta_min / train_config.lr))
        return optimizer, scheduler

    # Model & optimizer
    model = build_model()
    base_dist = IsotropicGaussian(shape=(1, V, V, V), device=device)
    if config.student_init == "fresh":
        # Every model trains on its own schedule: the warmup gets --warmup_lr / _weight_decay /
        # _ema and a cosine over just the warmup steps, so warmup and M-step settings don't leak
        # into each other.
        optimizer, scheduler = build_optimizer(model, config.warmup, config.warmup.n_steps_train)
        ema = EMA(model, decay=config.warmup.ema)
    else:
        optimizer = AdamW(model.parameters(),
                          lr=config.scsi.mstep.lr,
                          weight_decay=config.scsi.mstep.weight_decay)
        ema = EMA(model, decay=config.scsi.mstep.ema)
    # Weights the E-step samples with and the panels show. eval/ scores both regardless.
    sample_model = ema.ema_model if config.sample_with_ema else model

    # Load observations: extruded-MNIST volumes run through the 3D->2D tilt-series channel
    observations = build_observations(config.dataset)
    obs_checksum = observations.tensors[0].sum(dtype=torch.float64).item()

    # Visualization setup
    global_step = [0]
    viz_pool = build_viz_pool(config.dataset, n_pool=config.viz.n_pool, viz_seed=config.viz.seed)
    fixed = {k: v[:config.viz.n_display] for k, v in viz_pool.items()}

    def log_all_panels(em_step):
        rand = random_draw(viz_pool, config.dataset, config.viz.n_display)
        for panel_name, src in [("fixed", fixed), ("random", rand)]:
            log_reconstruction_grid(
                sample_model, src["x0"], src["y"], src["x_gt"],
                config.dataset, config.scsi.estep.n_steps_sampling,
                em_step, global_step[0], panel_name, device,
            )
            log_trajectory_grid(
                sample_model, src["x0"], src["y"],
                config.scsi.estep.n_steps_sampling, config.viz.n_snapshots,
                config.viz.n_trajectory_rows, em_step, global_step[0], panel_name, device,
            )

    # Scalar eval (eval_metrics). build_viz_pool saves/restores the global RNG and the metric
    # draws only from private generators, so enabling it leaves the training RNG stream as is.
    if config.eval_n > 0:
        eval_pool = build_viz_pool(config.dataset, n_pool=config.eval_n, viz_seed=config.eval_seed)
        align = AlignedCorrelation(device)
        calib = calibrate(eval_pool, align, device)
        wandb.run.summary.update({f"eval_calib/{k}": v for k, v in calib.items()})
        print("eval calibration (perfect recon, unknown frame): "
              + "  ".join(f"{k}={v:.4f}" for k, v in calib.items()), flush=True)

    def log_eval(em_step):
        if config.eval_n <= 0:
            return
        metrics = {}
        for weights, m in (("raw", model), ("ema", ema.ema_model)):
            scores = evaluate(m, eval_pool, config.scsi.estep.n_steps_sampling, align, device,
                              batch_size=config.scsi.estep.batch_size)
            metrics.update({f"eval/{weights}/{k}": v for k, v in scores.items()})
        wandb.log({**metrics, "em/step": em_step}, step=global_step[0])
        print(f"[em {em_step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                             if "/by_class/" not in k), flush=True)

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

    # Start from a checkpoint instead of training warmup: --resume's latest.pt (mid-EM), else
    # --load_warmup_ckpt -- a shared warm start: identical em-0 weights, optimizer moments and RNG
    # state across every run that loads it, so runs differ only in what they change after warmup.
    # CPU: torch.set_rng_state needs the saved RNG state as a CPU tensor. The load_state_dict
    # calls below move weights and optimizer moments onto the params' device themselves.
    resumed = resume is not None
    ckpt, ckpt_path = resume, latest
    if not resumed and config.load_warmup_ckpt:
        ckpt_path = config.load_warmup_ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check_args(ckpt["args"], args, _WARMUP_CKPT_KEYS, ckpt_path, arg_defaults=_ARG_DEFAULTS)
    start_em = 0
    if ckpt is not None:
        if not math.isclose(ckpt["obs_checksum"], obs_checksum, rel_tol=1e-9):
            raise ValueError(f"{ckpt_path}: observations differ from this run's "
                             f"(checksum {ckpt['obs_checksum']} vs {obs_checksum})")
        model.load_state_dict(ckpt["model"])
        ema.ema_model.load_state_dict(ckpt["ema_model"])
        # --student_init fresh: these weights only run the next E-step, and a new student with
        # its own optimizer takes the next optimizer step, so there is no optimizer to restore.
        if config.student_init == "teacher":
            optimizer.load_state_dict(ckpt["optimizer"])
            # load_state_dict restores the checkpoint run's lr (and the base lr LambdaLR scales).
            # Replace them with this run's, so --mstep_lr / --lr_schedule apply from the first EM
            # step. The scheduler below is rebuilt at the restored global step, not loaded: the
            # schedule is a pure function of that step.
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = config.scsi.mstep.lr
                group["weight_decay"] = config.scsi.mstep.weight_decay
        global_step[0] = ckpt["global_step"]
        start_em = ckpt.get("em_step", 0)
        set_rng_state(ckpt["rng"])
    del ckpt, resume

    if config.student_init == "teacher":
        horizon = (config.lr_horizon_scsi_steps if config.lr_horizon_scsi_steps is not None
                   else config.scsi.num_scsi_steps)
        scheduler = LambdaLR(optimizer, make_lr_lambda(
            config.lr_schedule, config.warmup.n_steps_train, config.scsi.mstep.n_steps_train,
            horizon, floor=config.scsi.lr_eta_min / config.scsi.mstep.lr,
            start_step=global_step[0]))

    if not resumed and not config.load_warmup_ckpt:
        # Warm start on RESAMPLED (target, ŷ) pairs generated on the fly from the pseudoinverse
        # recons X alone -- NOT build_warmup's frozen (x_hat, y_obs) pairs. X = pseudoinverse.py's
        # filtered backprojection of the observed tilt series, renormed to [-1, 1] (build_warmup);
        # we keep only that tensor and discard its paired y_obs. ResampledPairs then re-draws, per
        # __getitem__, ŷ = F(X) through a fresh random mount + tilt series and (under --lift)
        # target = R·X for a fresh independent SO(3) R -- the SAME pair_sample the E-step uses, so
        # the warm start already trains on the rotation-symmetrized (R·X, F(X)) objective instead
        # of one fixed (X, y_obs) draw. Cost: ResampledPairs runs the channel per-sample inside the
        # dataloader (num_workers=0 in mstep_lifted), so warmup steps slow down -- same tradeoff
        # as main_supervised.py's --resample_channel.
        warmup_x = build_warmup(observations, config.dataset).tensors[0]
        warmup_pairs = ResampledPairs(TensorDataset(warmup_x), pair_sample)
        mstep_lifted(
            model, base_dist, warmup_pairs, optimizer, config.warmup,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="warmup",
        )
        del warmup_x, warmup_pairs
        if config.save_warmup_ckpt:
            atomic_save({
                "model": model.state_dict(), "ema_model": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "global_step": global_step[0],
                "rng": rng_state(), "obs_checksum": obs_checksum, "args": vars(args),
            }, config.save_warmup_ckpt)

    # A resumed run already has its RNG state (and its em-0 panels / eval) from the first job.
    if not resumed:
        if config.em_seed is not None:
            torch.manual_seed(config.em_seed)
        log_all_panels(em_step=0)
        log_eval(em_step=0)

    # Run SCSI algorithm
    for k in range(start_em, config.scsi.num_scsi_steps):
        t0 = time.time()
        # E-step: sample from the posterior over latent clean volumes given the observations
        posterior_samples = estep(
            sample_model, base_dist, observations, pair_sample, config.scsi.estep
        )
        t1 = time.time()

        # --student_init fresh: the teacher's only job was the E-step above. Discard it and train
        # a newly initialized student (new optimizer, EMA, and cosine LR restarted over this
        # M-step) instead of fine-tuning the teacher in place.
        if config.student_init == "fresh":
            model = build_model()
            optimizer, scheduler = build_optimizer(model, config.scsi.mstep,
                                                   config.scsi.mstep.n_steps_train)
            ema = EMA(model, decay=config.scsi.mstep.ema)
            sample_model = ema.ema_model if config.sample_with_ema else model

        # M-step: update model parameters to maximize expected log-likelihood
        mstep_lifted(
            model, base_dist, posterior_samples, optimizer, config.scsi.mstep,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="train",
        )
        t2 = time.time()

        if (k + 1) % config.viz.every == 0:
            log_all_panels(em_step=k + 1)
            log_eval(em_step=k + 1)

        if config.ckpt_dir:
            atomic_save({
                "model": model.state_dict(), "ema_model": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "em_step": k + 1, "global_step": global_step[0], "rng": rng_state(),
                "obs_checksum": obs_checksum, "wandb_id": wandb.run.id, "args": vars(args),
            }, os.path.join(config.ckpt_dir, "latest.pt"))
        # Slurm-log progress line, for checking wall time against --time.
        print(f"[em {k + 1}/{config.scsi.num_scsi_steps}] estep {t1 - t0:.0f}s  "
              f"mstep {t2 - t1:.0f}s  viz+eval+ckpt {time.time() - t2:.0f}s  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    wandb.finish()
