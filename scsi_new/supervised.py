"""
Supervised stochastic-interpolant training -- the paired-data counterpart to scsi.py.

scsi.py (UNSUPERVISED): only corrupted observations y are available. The EM loop alternates
estep() -- propose clean x_hat by flowing noise through the current velocity field's ODE,
conditioned on y -- and mstep_lifted() -- train the conditional velocity net on the proposed
(x_hat, y_hat) pairs.

supervised.py (SUPERVISED): a dataset of ground-truth pairs {(x_i, F(x_i))}_i is available, so
there is no posterior to estimate. Fit the conditional velocity net b_t(.|y) directly, by the
same stochastic-interpolant loss the M-step already uses:

    L = E_{t, z~N(0,I), (x,y)} || b_t(alpha_t z + beta_t x | y) - (alpha_dot_t z + beta_dot_t x) ||^2

The per-step math is delegated verbatim to scsi.mstep_lifted, so a supervised run and one
unsupervised M-step differ ONLY in where the (x, y) pairs come from -- which is the whole point
of keeping this around as a baseline.

Each experiment (MNIST / synthetic / MNIST-3D / ...) plugs in via its existing
data.py / corruption.py / model.py -- this module knows nothing about datatype, model
architecture, or visualization:

    # main_supervised.py, sitting next to the experiment's own main.py
    import functools
    from corruption import corruption_channel
    from data import load_mnist_subset, build_viz_pool          # experiment-specific
    from model import ConditionalDiT                             # experiment-specific
    from distribution import IsotropicGaussian
    from scsi import basic_pair
    from supervised import (Config_Supervised, build_paired_dataset,
                            train_supervised, autodetect_device)

    device = autodetect_device()
    F = functools.partial(corruption_channel, noise_std=cfg.noise_std)  # bind channel params
    dataset = build_paired_dataset(load_mnist_subset(cfg), basic_pair(F))   # {(x, F(x))}
    model   = ConditionalDiT(image_size=cfg.image_size).to(device)
    base    = IsotropicGaussian(shape=(1, cfg.image_size, cfg.image_size), device=device)

    def on_log(round_idx, global_step, ema_model):
        log_all_panels(em_step=round_idx)      # experiment's existing wandb panels, unchanged

    train_supervised(model, base, dataset, Config_Supervised(), on_log=on_log)

Like scsi.py, this file has no CLI -- each experiment keeps its own argparse in main_supervised.py.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, TensorDataset

from distribution import Distribution
from scsi import Config_SCSI_MStep, EMA, ResampledPairs, mstep_lifted


def autodetect_device() -> str:
    """cuda -> mps -> cpu, at every entry point. Mirror of scsi_args.autodetect_device (kept
    here so this library file doesn't pull in the CLI-args module)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config_Supervised:
    interpolant_style: str = "gvp"   # "linear" | "gvp" -- handed straight to si.load_interpolant
    n_steps_train: int = 20_000      # total optimizer steps; one (x, y) minibatch per step
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    ema: float = 0.999               # EMA decay on the weights, updated once per optimizer step
    eta_min: float = 1e-5            # LR floor of the cosine schedule spanning ALL n_steps_train
    log_every: int = 1_000           # steps between on_log callbacks; <= 0 -> only at start & end
    seed: int | None = 42           # seeds the global RNG at entry (batch order / x0 / channel
                                     # resampling); the experiment's entry point should still seed
                                     # before model construction, as main.py does. None -> skip.
    checkpoint_steps: tuple[int, ...] = ()   # step counts at which to torch.save a checkpoint;
                                             # training pauses exactly on each. Out-of-range
                                             # values (<= 0 or > n_steps_train) are dropped.
    checkpoint_dir: str = "checkpoints"      # dir for step_<n>.pt files (gitignored as checkpoints*)


def build_paired_dataset(x_source: torch.Tensor | Dataset,
                         pair_sample: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                         *,
                         batch_size: int = 32) -> TensorDataset:
    """
    Materialize a supervised training set {(target_i, y_i)}_i -- the analogue of an experiment's
    build_observations(), but keeping a clean target alongside the observation y.

    Args:
        x_source: clean ground-truth samples -- either a Tensor (N, *shape), or a Dataset that
            yields 1-tuples (x,) (e.g. the ConcatDataset an experiment's load_mnist_subset
            returns). x must already be in the shape the velocity net / base_dist expect
            (synthetic 2D points, for instance, are carried as (2, 1, 1)).
        pair_sample: `(x_batch) -> (target_batch, y_batch)`. The default is scsi.basic_pair(F)
            -- identity target, y = F(x) -- with F ALREADY functools.partial-bound with its
            channel params exactly as main.py binds it. The CryoET experiments pass
            corruption.build_pair_sample(F, lift=...) instead, whose lifted variant returns
            (R.x, F(x)) for a fresh independent Haar rotation R. One realization is drawn per
            sample here, then frozen into the returned dataset.
        batch_size: how many samples to push through pair_sample at once. Small default on
            purpose: the 3D CryoET channel expands each sample to (B*num_tilts, 1, D, H, W)
            internally for a single grid_sample, so a large batch OOMs (cf.
            cryoet_mnist3d/data.py's _CHANNEL_BATCH = 32).

    Returns:
        TensorDataset(target, y), both on x_source's original device.
    """
    base = x_source if not isinstance(x_source, torch.Tensor) else TensorDataset(x_source)
    loader = DataLoader(base, batch_size=batch_size, shuffle=False)

    xs, ys = [], []
    for (x_batch,) in loader:  # 1-tuple: TensorDataset / load_mnist_subset both yield (x,)
        t_batch, y_batch = pair_sample(x_batch)
        xs.append(t_batch)
        ys.append(y_batch)
    return TensorDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0))


def _training_plan(total: int, log_every: int,
                   checkpoint_steps: tuple[int, ...]) -> list[tuple[int, int, bool, bool]]:
    """
    Ordered (chunk_len, step, do_log, do_checkpoint) blocks covering [0, total]. Training pauses
    at every multiple of `log_every` (do_log), at each `checkpoint_steps` value (do_checkpoint),
    and always at `total` (do_log). `chunk_len` values are exact integers summing to EXACTLY
    `total`, so the single CosineAnnealingLR(T_max=total) -- stepped once per optimizer step
    inside mstep_lifted -- is unaffected by where the pauses fall. `checkpoint_steps` is assumed
    pre-filtered to 0 < s <= total.
    """
    log_pts = set(range(log_every, total, log_every)) if log_every > 0 else set()
    log_pts.add(total)
    ckpt_pts = set(checkpoint_steps)

    plan, prev = [], 0
    for s in sorted(log_pts | ckpt_pts):
        if s - prev > 0:  # a 0-length block only arises at total == 0
            plan.append((s - prev, s, s in log_pts, s in ckpt_pts))
            prev = s
    return plan


def _save_checkpoint(path: Path, step: int, model: torch.nn.Module, ema: EMA,
                     optimizer: AdamW, scheduler: CosineAnnealingLR,
                     config: "Config_Supervised", meta: dict | None) -> None:
    """torch.save a resumable checkpoint: live + EMA weights, optimizer + LR-schedule state, the
    Config_Supervised (as a plain dict), and whatever call-site `meta` the entry point passed
    (typically vars(args), so the run is fully reconstructible)."""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema.ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(config),
    }
    if meta is not None:
        ckpt["meta"] = meta
    torch.save(ckpt, path)


def _check_dataset(dataset: Dataset, base_dist: Distribution) -> None:
    first = dataset[0]
    if not isinstance(first, (tuple, list)) or len(first) < 2:
        raise ValueError(
            "train_supervised expects `dataset` to yield (target, y) pairs -- e.g. "
            "build_paired_dataset(...) output. To regenerate y from a clean-x dataset instead, "
            "pass resample_pair_sample=pair_sample."
        )
    expected = getattr(base_dist, "shape", None)
    if expected is not None and tuple(first[0].shape) != tuple(expected):
        raise ValueError(
            f"train_supervised: dataset x sample has shape {tuple(first[0].shape)}, but "
            f"base_dist samples {tuple(expected)}. si.Interpolant does alpha_t*z + beta_t*x with "
            f"no broadcasting slack -- reshape x when you build the dataset (synthetic 2D points, "
            f"for instance, are carried as (2, 1, 1): x.view(-1, 2, 1, 1))."
        )


def train_supervised(
    model: torch.nn.Module,
    base_dist: Distribution,
    dataset: Dataset,
    config: Config_Supervised,
    *,
    on_log: Callable[[int, int, torch.nn.Module], None] | None = None,
    resample_pair_sample: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    checkpoint_meta: dict | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Fit `model` -- a conditional velocity net with forward(x_t, t, y) -> v -- on ground-truth
    (x, y) pairs by the stochastic-interpolant loss. Builds its own AdamW + a single global
    CosineAnnealingLR spanning the full config.n_steps_train (so the LR reaches config.eta_min
    exactly once, at the end) + EMA, then feeds each block of steps to scsi.mstep_lifted.

    Training pauses -- without perturbing the LR schedule -- at every config.log_every steps to
    call on_log, and at each config.checkpoint_steps value to torch.save
    config.checkpoint_dir/step_<n>.pt (model + EMA + optimizer + scheduler + config +
    checkpoint_meta).

    Args:
        model: velocity net, already placed on its device by the caller (this function moves
            nothing).
        base_dist: noise source z. base_dist.shape, if present, MUST equal the dataset's x
            sample shape -- see _check_dataset.
        dataset: yields (target, y) pairs, e.g. build_paired_dataset(...) output. A clean-x
            dataset (or bare tensor) is also fine when resample_pair_sample is given.
        config: Config_Supervised.
        on_log: optional callback(round_idx, global_step, ema_model). Fired once before any
            training (round_idx = 0) and after every config.log_every steps. Pass round_idx as
            the `em_step` argument of the experiment's existing wandb panels and they work
            verbatim; ema_model is handed in because the caller can't close over an EMA this
            function builds.
        resample_pair_sample: optional `(x) -> (target, y)` callable (scsi.basic_pair(F) or an
            experiment's corruption.build_pair_sample(F, lift=...)). When given, (target, y) is
            re-drawn on every batch fetch (fresh channel noise -- and fresh rotation R when
            lifting -- each epoch) rather than read from `dataset`. Default None -> frozen pool.
        checkpoint_meta: optional dict stashed verbatim into every checkpoint under "meta"
            (e.g. {"args": vars(args)} so the run is reconstructible from the .pt alone).

    Returns:
        (model, ema_model) -- the live weights and their EMA.
    """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    if resample_pair_sample is not None:
        dataset = ResampledPairs(dataset, resample_pair_sample)
    _check_dataset(dataset, base_dist)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_steps_train, eta_min=config.eta_min)
    ema = EMA(model, decay=config.ema)

    # mstep_lifted reads only .interpolant_style / .n_steps_train / .batch_size off this; lr /
    # weight_decay / ema are already baked into optimizer + ema above. n_steps_train is swapped
    # per chunk below via dataclasses.replace.
    mstep_cfg = Config_SCSI_MStep(
        interpolant_style=config.interpolant_style,
        n_steps_train=config.n_steps_train,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        ema=config.ema,
    )

    # mstep_lifted logs to wandb unconditionally when global_step is not None; pass None when no
    # run is active so a wandb-less supervised run still trains (CLAUDE.md: logging degrades
    # gracefully). The callback still gets a monotone step count either way.
    wandb_on = getattr(wandb, "run", None) is not None
    global_step = [0]

    ckpt_steps = tuple(sorted({s for s in config.checkpoint_steps
                               if 0 < s <= config.n_steps_train}))
    dropped = sorted(set(config.checkpoint_steps) - set(ckpt_steps))
    if dropped:
        print(f"[supervised] ignoring out-of-range checkpoint_steps {dropped} "
              f"(training runs {config.n_steps_train} steps)")
    ckpt_dir = Path(config.checkpoint_dir)
    if ckpt_steps:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(config.n_steps_train))

    if on_log is not None:
        on_log(0, 0, ema.ema_model)

    round_idx = 0
    for chunk, step, do_log, do_ckpt in _training_plan(config.n_steps_train, config.log_every,
                                                       ckpt_steps):
        mstep_lifted(
            model, base_dist, dataset, optimizer,
            replace(mstep_cfg, n_steps_train=chunk),
            scheduler=scheduler, ema=ema,
            global_step=global_step if wandb_on else None,
            log_prefix="train",
        )
        gs = global_step[0] if wandb_on else step  # chunks sum from 0, so step == cumulative

        if do_ckpt:
            path = ckpt_dir / f"step_{step:0{width}d}.pt"
            _save_checkpoint(path, step, model, ema, optimizer, scheduler, config, checkpoint_meta)
            print(f"[supervised] checkpoint @ step {step} -> {path}")

        if do_log and on_log is not None:
            round_idx += 1
            on_log(round_idx, gs, ema.ema_model)

    return model, ema.ema_model
