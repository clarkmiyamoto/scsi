import sys
from pathlib import Path

# This file lives at scsi_new/experiments/cryoet_igg1d/main.py. scsi.py, si.py, ode.py, and
# distribution.py live flat at scsi_new/ and import each other with bare imports, so scsi_new/
# must be on sys.path. This directory's own modules (and the vendored unet3d/) resolve because
# Python puts a directly-run script's directory at sys.path[0].
SCSI_NEW_ROOT = Path(__file__).resolve().parents[2]
if str(SCSI_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SCSI_NEW_ROOT))

import functools
import math
import os
import time

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from corruption import corruption_channel, build_pair_sample  # black box forward model
from data import PseudoinverseVolumes, build_eval_pool, build_observations, calibrate_gain
from distribution import IsotropicGaussian
from eval_metrics import calibrate, evaluate
from model import ConditionalVelocityIgG
from scsi import EMA, ResampledPairs, estep, mstep_lifted
from args import parse_args, config_from_args
from wandb_logging import log_reconstruction_grid, log_trajectory_grid, random_draw, select

"""
SCSI on CryoBench IgG-1D (see data.py): the E-step samples x_hat ~ p(x | y) for observed particle
images y, the M-step fits b_t(x | y) on (target, F(x_hat)) pairs from the forward model
(corruption.py), with the supervised cryofm run's network (model.py). Only the images and their
CTF parameters are observed; GT volumes, poses and conformations reach eval_metrics and the
panels only. The loop, checkpointing and LR schedule are cryoet_mnist3d/main.py's.

The warm start trains on ResampledPairs of pseudoinverse.py's pose-blind reconstructions of the
observed images (--warmup_n_steps_train 0 skips it: EM from a randomly initialised network).
"""

# Args that fix the observations, the channel and the network. --load_warmup_ckpt refuses a
# checkpoint that disagrees with this run on any of them (and on vol_gain, the noise sigma and
# the observations checksum, checked separately).
_WARMUP_CKPT_KEYS = (
    "resolution", "snr", "val_fraction", "split_seed", "n_observations", "seed",
    "shift_extent_A", "pair_frame", "recenter", "pinv_diameter_A",
    "block_out_channels", "layers_per_block", "encoder_channels",
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


def _check_close(name: str, ckpt_value: float, value: float, path: str) -> None:
    if not math.isclose(ckpt_value, value, rel_tol=1e-9):
        raise ValueError(f"{path}: {name} differs from this run's ({ckpt_value} vs {value})")


def make_lr_lambda(schedule: str, warmup_steps: int, mstep_steps: int, horizon_scsi_steps: int,
                   floor: float, start_step: int = 0):
    """
    LambdaLR multiplier on --mstep_lr, as a function of the scheduler's own step count i (the
    global optimizer step is i + start_step). floor = eta_min / mstep_lr. As cryoet_mnist3d:

    cosine: floor + (1 - floor) * (1 + cos(pi * g / T)) / 2 with T = warmup + horizon * mstep,
        g clamped at T (held at eta_min afterwards, unlike CosineAnnealingLR).
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


def _subpool(pool: dict, n: int) -> dict:
    # n items evenly spaced through the pool (the whole pool when n == its size), plus the shared
    # gt_volumes. The pool is itself spread over the held-out images, so this stays spread.
    idx = torch.from_numpy(np.linspace(0, pool["x_gt"].size(0) - 1, n).round().astype(np.int64))
    return {**select(pool, idx), "gt_volumes": pool["gt_volumes"]}


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
    D = config.dataset.resolution

    # Model & optimizer
    model = ConditionalVelocityIgG(
        resolution=D,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        encoder_channels=config.encoder_channels,
        bf16=config.bf16,
    ).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params", flush=True)
    base_dist = IsotropicGaussian(shape=(1, D, D, D), device=device)
    optimizer = AdamW(model.parameters(),
                      lr=config.scsi.mstep.lr,
                      weight_decay=config.scsi.mstep.weight_decay)
    ema = EMA(model, decay=config.scsi.mstep.ema)
    # Weights the E-step samples with and the panels show. eval/ scores both regardless.
    sample_model = ema.ema_model if config.sample_with_ema else model

    # Observations: the particle images (phase-flipped, noise std 1) and their CTF parameters
    observations, obs_info = build_observations(config.dataset)
    obs_checksum = observations.tensors[0].sum(dtype=torch.float64).item()
    sigma, apix = obs_info["sigma"], obs_info["apix"]
    ctf_pool = obs_info["ctf_pool"].to(device)

    # Start from a checkpoint instead of training warmup: --resume's latest.pt (mid-EM), else
    # --load_warmup_ckpt -- a shared warm start: identical em-0 weights, optimizer moments and RNG
    # state across every run that loads it. CPU: torch.set_rng_state needs the saved RNG state as
    # a CPU tensor; load_state_dict moves weights and optimizer moments onto the params' device.
    global_step = [0]
    resumed = resume is not None
    ckpt, ckpt_path = resume, latest
    if not resumed and config.load_warmup_ckpt:
        ckpt_path = config.load_warmup_ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _check_args(ckpt["args"], args, _WARMUP_CKPT_KEYS, ckpt_path)
    start_em = 0
    if ckpt is not None:
        _check_close("observations checksum", ckpt["obs_checksum"], obs_checksum, ckpt_path)
        _check_close("noise sigma", ckpt["sigma"], sigma, ckpt_path)
        model.load_state_dict(ckpt["model"])
        ema.ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict restores the checkpoint run's lr. Replace it with this run's, so
        # --mstep_lr / --lr_schedule apply from the first EM step; the scheduler below is rebuilt
        # at the restored global step (the schedule is a pure function of that step).
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = config.scsi.mstep.lr
            group["weight_decay"] = config.scsi.mstep.weight_decay
        global_step[0] = ckpt["global_step"]
        start_em = ckpt.get("em_step", 0)
        _set_rng_state(ckpt["rng"])

    # vol_gain fixes the units of the generated volumes, so it must stay the same for the whole
    # run: --vol_gain, else the checkpoint's, else calibrated on the pseudoinverse warm start. A
    # checkpoint's must match.
    if config.vol_gain is not None:
        vol_gain = config.vol_gain
    elif ckpt is not None:
        vol_gain = ckpt["vol_gain"]
    else:
        vol_gain = calibrate_gain(observations, ctf_pool, apix, config.dataset, device)
    if ckpt is not None:
        _check_close("vol_gain", ckpt["vol_gain"], vol_gain, ckpt_path)
    del ckpt, resume
    wandb.config.update({"resolved/vol_gain": vol_gain, "resolved/noise_sigma": sigma,
                         "resolved/n_observations": len(observations)}, allow_val_change=True)
    print(f"vol_gain {vol_gain:.4g}", flush=True)

    # y_hat = F(x_hat) -- for the warm start AND the E-step -- with the observations' channel
    # parameters, drawing a fresh pose, shift and CTF on every call.
    corruption_channel_bound = functools.partial(
        corruption_channel,
        ctf_pool=ctf_pool,
        vol_gain=vol_gain,
        apix=apix,
        shift_extent_px=config.dataset.shift_extent_px,
        recenter=config.recenter,
    )
    # (x_hat) -> (target, y_hat); --pair_frame picks the target.
    pair_sample = build_pair_sample(corruption_channel_bound, frame=config.pair_frame,
                                    recenter=config.recenter)

    # Held-out images with GT, for the panels and the metrics only. One pool, spread over the
    # held-out set; with --eval_n == --viz_n_pool (16, the default) both use all of it, which is
    # the image set cryofm's sampling eval uses.
    pool = build_eval_pool(config.dataset, n_pool=max(config.eval_n, config.viz.n_pool),
                           seed=config.eval_seed, sigma=sigma)
    viz_pool = _subpool(pool, config.viz.n_pool)
    fixed = select(viz_pool, slice(0, config.viz.n_display))

    def log_all_panels(em_step):
        rand = random_draw(viz_pool, config.viz.n_display)
        for panel_name, src in [("fixed", fixed), ("random", rand)]:
            log_reconstruction_grid(
                sample_model, src, apix, vol_gain, config.scsi.estep.n_steps_sampling,
                em_step, global_step[0], panel_name, device,
            )
            log_trajectory_grid(
                sample_model, src["x0"], src["y"],
                config.scsi.estep.n_steps_sampling, config.viz.n_snapshots,
                config.viz.n_trajectory_rows, em_step, global_step[0], panel_name, device,
            )

    if config.eval_n > 0:
        eval_pool = _subpool(pool, config.eval_n)
        calib = calibrate(eval_pool, device, apix)
        wandb.run.summary.update({f"eval_calib/{k}": v for k, v in calib.items()})
        print("eval calibration (GT targets): "
              + "  ".join(f"{k}={v:.4f}" for k, v in calib.items()), flush=True)

    def log_eval(em_step):
        if config.eval_n <= 0:
            return
        metrics = {}
        for weights, m in (("raw", model), ("ema", ema.ema_model)):
            scores = evaluate(m, eval_pool, config.scsi.estep.n_steps_sampling, device, apix,
                              batch_size=config.scsi.estep.batch_size)
            metrics.update({f"eval/{weights}/{k}": v for k, v in scores.items()})
        wandb.log({**metrics, "em/step": em_step}, step=global_step[0])
        print(f"[em {em_step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
              flush=True)

    horizon = (config.lr_horizon_scsi_steps if config.lr_horizon_scsi_steps is not None
               else config.scsi.num_scsi_steps)
    scheduler = LambdaLR(optimizer, make_lr_lambda(
        config.lr_schedule, config.warmup.n_steps_train, config.scsi.mstep.n_steps_train, horizon,
        floor=config.scsi.lr_eta_min / config.scsi.mstep.lr, start_step=global_step[0]))

    if not resumed and not config.load_warmup_ckpt:
        # Warm start on RESAMPLED (target, y_hat) pairs from the pseudoinverse reconstructions x0
        # of the observed images, as cryoet_mnist3d: each __getitem__ re-draws y_hat = F(x0)
        # through a fresh pose, shift and CTF, with the E-step's own pair_sample. x0 is computed
        # per item on the GPU (data.PseudoinverseVolumes).
        warmup_pairs = ResampledPairs(
            PseudoinverseVolumes(observations, ctf_pool, apix, config.dataset, vol_gain, device),
            pair_sample)
        t0 = time.time()
        mstep_lifted(
            model, base_dist, warmup_pairs, optimizer, config.warmup,
            scheduler=scheduler, ema=ema, global_step=global_step, log_prefix="warmup",
        )
        # Slurm-log line for sizing --time of a full-length warmup.
        dt = time.time() - t0
        print(f"warmup: {config.warmup.n_steps_train} steps in {dt:.0f}s "
              f"({config.warmup.n_steps_train / max(dt, 1e-9):.2f} it/s)", flush=True)
        del warmup_pairs
        if config.save_warmup_ckpt:
            _atomic_save({
                "model": model.state_dict(), "ema_model": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "global_step": global_step[0],
                "rng": _rng_state(), "obs_checksum": obs_checksum, "sigma": sigma,
                "vol_gain": vol_gain, "args": vars(args),
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
        del posterior_samples
        t2 = time.time()

        if (k + 1) % config.viz.every == 0:
            log_all_panels(em_step=k + 1)
            log_eval(em_step=k + 1)

        if config.ckpt_dir:
            _atomic_save({
                "model": model.state_dict(), "ema_model": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "em_step": k + 1, "global_step": global_step[0], "rng": _rng_state(),
                "obs_checksum": obs_checksum, "sigma": sigma, "vol_gain": vol_gain,
                "wandb_id": wandb.run.id, "args": vars(args),
            }, os.path.join(config.ckpt_dir, "latest.pt"))
        # Slurm-log progress line, for checking wall time against --time.
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"[em {k + 1}/{config.scsi.num_scsi_steps}] estep {t1 - t0:.0f}s  "
              f"mstep {t2 - t1:.0f}s  viz+eval+ckpt {time.time() - t2:.0f}s  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  peak GPU mem {peak:.1f} GiB", flush=True)

    wandb.finish()
