"""
Helpers every EM `main.py` shares verbatim: checkpoint-compatibility check, the LR schedule,
RNG-state capture for exact resume, and atomic checkpoint writes. Moved here from
experiments/cryoet_mnist3d/main.py and cryoet_igg1d/main.py, where they were copies (the two
`make_lr_lambda` docstrings differed; the code did not). Behaviour is unchanged on purpose: runs
chained with --resume across this move must keep working.
"""
import math
import os
from pathlib import Path

import torch


def check_args(ckpt_args: dict, args, keys: tuple[str, ...], path: str,
               arg_defaults: dict | None = None) -> None:
    """
    Raise if the checkpoint at `path` was made with different values for any of `keys`.
    arg_defaults: args added after checkpoints already existed -- a checkpoint without the key
    ran with this value.
    """
    ckpt_args = {**(arg_defaults or {}), **ckpt_args}
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


def rng_state() -> dict:
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(obj: dict, path: str) -> None:
    # A walltime kill mid-write leaves the previous file intact instead of a truncated one.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
