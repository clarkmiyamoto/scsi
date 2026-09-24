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
import math
import os
import time

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset

from corruption import corruption_channel, build_pair_sample  # black box forward model
from data import build_observations, build_warmup, build_viz_pool
from distribution import IsotropicGaussian
from eval_metrics import AlignedCorrelation, calibrate, evaluate
from model import ConditionalVelocityCryoET3D
from scsi import EMA, ResampledPairs, estep, mstep_lifted
from args import parse_args, config_from_args
from wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw


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
    "sample_with_ema",
)


def _check_args(ckpt_args: dict, args, keys: tuple[str, ...], path: str) -> None:
    # argparse gives nargs values as lists and their defaults as tuples; compare as lists.
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v
    mismatched = {k: (ckpt_args.get(k), vars(args).get(k)) for k in keys
                  if norm(ckpt_args.get(k)) != norm(vars(args).get(k))}
    if mismatched:
        raise ValueError(f"{path} was made with different args (checkpoint, this run): "
                         f"{mismatched}")


def make_lr_lambda(schedule: str, warmup_steps: int, mstep_steps: int, horizon_scsi_steps: int,
                   floor: float, start_step: int = 0):
    """
    LambdaLR multiplier on --mstep_lr, as a function of the scheduler's own step count i (the
    global optimizer step is i + start_step, so a run resumed from a warmup checkpoint picks the
    schedule up where the checkpoint left off). floor = eta_min / mstep_lr.

    cosine: floor + (1 - floor) * (1 + cos(pi * g / T)) / 2 with T = warmup + horizon * mstep --
        CosineAnnealingLR's closed form, except that g is clamped at T. CosineAnnealingLR climbs
        back up past T_max, which would silently re-warm a run with horizon < num_scsi_steps.
    constant: 1.
    cosine_per_mstep: the same cosine restarted over the warmup and over every M-step.
    """
    def cos(frac: float) -> float:
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0)))

    total = max(1, warmup_steps + horizon_scsi_steps * mstep_steps)

    def lr_lambda(i: int) -> float:
        g = i + start_step
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            return cos(g / total)
        if schedule == "cosine_per_mstep":
            if g < warmup_steps:
                return cos(g / warmup_steps)
            return cos(((g - warmup_steps) % mstep_steps) / mstep_steps)
        raise ValueError(f"unknown lr_schedule {schedule!r}")

    return lr_lambda


def _rng_state() -> dict:
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj: dict, path: str) -> None:
    # A walltime kill mid-write leaves the previous file intact instead of a truncated one.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


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
        _check_args(resume["args"], args, _RESUME_KEYS, latest)
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
        _check_args(ckpt["args"], args, _WARMUP_CKPT_KEYS, ckpt_path)
    start_em = 0
    if ckpt is not None:
        if not math.isclose(ckpt["obs_checksum"], obs_checksum, rel_tol=1e-9):
            raise ValueError(f"{ckpt_path}: observations differ from this run's "
                             f"(checksum {ckpt['obs_checksum']} vs {obs_checksum})")
        model.load_state_dict(ckpt["model"])
        ema.ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict restores the checkpoint run's lr (and the base lr LambdaLR scales).
        # Replace them with this run's, so --mstep_lr / --lr_schedule apply from the first EM step.
        # The scheduler below is rebuilt at the restored global step, not loaded: the schedule
        # is a pure function of that step.
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = config.scsi.mstep.lr
            group["weight_decay"] = config.scsi.mstep.weight_decay
        global_step[0] = ckpt["global_step"]
        start_em = ckpt.get("em_step", 0)
        _set_rng_state(ckpt["rng"])
    del ckpt, resume

    horizon = (config.lr_horizon_scsi_steps if config.lr_horizon_scsi_steps is not None
               else config.scsi.num_scsi_steps)
    scheduler = LambdaLR(optimizer, make_lr_lambda(
        config.lr_schedule, config.warmup.n_steps_train, config.scsi.mstep.n_steps_train, horizon,
        floor=config.scsi.lr_eta_min / config.scsi.mstep.lr, start_step=global_step[0]))

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
            _atomic_save({
                "model": model.state_dict(), "ema_model": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "global_step": global_step[0],
                "rng": _rng_state(), "obs_checksum": obs_checksum, "args": vars(args),
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
            _atomic_save({
                "model": model.state_dict(), "ema_model": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "em_step": k + 1, "global_step": global_step[0], "rng": _rng_state(),
                "obs_checksum": obs_checksum, "wandb_id": wandb.run.id, "args": vars(args),
            }, os.path.join(config.ckpt_dir, "latest.pt"))
        # Slurm-log progress line, for checking wall time against --time.
        print(f"[em {k + 1}/{config.scsi.num_scsi_steps}] estep {t1 - t0:.0f}s  "
              f"mstep {t2 - t1:.0f}s  viz+eval+ckpt {time.time() - t2:.0f}s  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    wandb.finish()
