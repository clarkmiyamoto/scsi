# SCSI — project notes for Claude

Self-Consistent Stochastic Interpolants: an EM loop (E-step = sample posteriors with the current
velocity model, M-step = retrain on those samples) for recovering clean data from corrupted
observations. See `README.md`; algorithm in `scsi_new/`, one directory per problem under
`scsi_new/experiments/`.

## Layout

```
scsi_new/                      package: scsi.py (EM), si.py, ode.py, sde.py, supervised.py, ...
  paths.py                     cluster profiles + every filesystem location (no hardcoded paths elsewhere)
  train_utils.py               shared by the EM main.py files: check_args, make_lr_lambda, rng/atomic save
  sbatch_gen.py                generates sbatch scripts from each experiment's sweeps.py
  experiments/<name>/          main.py, args.py, data.py, corruption.py, model.py, ... (+ sweeps.py, JOURNAL.md)
    sbatch/<sweep>/README.md   design notes for a sweep (legacy *.SBATCH files live beside them)
tests/                         pytest: imports, train_utils, paths, sbatch_gen
```

Experiments: `cryoet_mnist` (2D tilt series), `cryoet_mnist3d` (3D->2D, SO(3) mount),
`cryoet_igg1d` (CryoBench IgG-1D), `mra_mnist` (in-plane rotation), `awgn_synthetic`.

## Running

Everything is a module run **from the repo root** (relative imports; no `cd` into the experiment):

```bash
uv run python -m scsi_new.experiments.cryoet_mnist3d.main --help
uv run python -m scsi_new.experiments.cryoet_mnist3d.data --help      # demos too: always -m
```

Never run Python on the login node. Use an interactive node
(`srun --cpus-per-task=4 --mem=8G --pty bash`) or a job. Never submit jobs yourself: generate
the scripts and give Clark the `bash .../submit_all.sh` command.

## Clusters

Two profiles in `scsi_new/paths.py`, chosen by `$SCSI_CLUSTER` (`nyu` | `rusty`; autodetected
from `/mnt/ceph`). Override any location with `SCSI_REPO_DIR`, `SCSI_RUNS_DIR`, `SCSI_DATA_DIR`,
`SCSI_IGG_DATA_ROOT`, `SCSI_CRYOBENCH_ROOT`. NYU runs/data: `/scratch/cm6627/scsi_runs`,
`/scratch/cm6627/scsi_data`. The NYU IgG-1D / CryoBench defaults are placeholders.
NYU sbatch scripts set no partition/constraint (not pinned down); Rusty ones use
`--partition=gpu --constraint=rocky9&(h100|a100-80gb)` and `module load python/3.13.2 uv`.

## Sweeps -> sbatch

A sweep is a `Sweep` in `experiments/<name>/sweeps.py`. To add or change one, edit that file, then:

```bash
uv run python -m scsi_new.sbatch_gen --list
uv run python -m scsi_new.sbatch_gen cryoet_mnist3d lr_schedule_ablation --cluster nyu
bash scsi_new/experiments/cryoet_mnist3d/sbatch/lr_schedule_ablation/generated/nyu/submit_all.sh
```

Output goes to `<experiment>/sbatch/<sweep>/generated/<cluster>/` (gitignored). `--repo` points
jobs at a frozen checkout (git worktree) so editing the main one does not change queued jobs.
Regenerated scripts write checkpoints under `<runs>/<exp>/<sweep>/checkpoints`, not where the old
scripts did, so with `--resume` they start **new** runs. To continue an existing chain (3D
`fresh_lr_tied_steps_grid`, `lr_schedule_ablation`, `tenclass_sweep`) generate with `--legacy-ckpt`.
The older hand-written `sbatch/*/*.SBATCH` files predate the package layout (`cd` into the
experiment, `python main.py`) and **no longer run as written**; they are history. The IgG-1D
scripts (`cryoet_igg1d/sbatch/`) are still hand-written but were updated to the `-m` launch.

## Journal rule

After every sweep, **append an entry to that experiment's `JOURNAL.md`** (template and rules:
`scsi_new/experiments/JOURNAL_TEMPLATE.md`): question, hypothesis, change from baseline, setup,
result, interpretation, next. Write the entry when you add the sweep; fill Result with numbers and
wandb ids when runs finish. Never invent a result: `TBD` until it is measured.

## Compatibility rules

- Do not rename flags or change defaults of the EM `main.py`s: `--resume` compares the saved
  `vars(args)` (`train_utils.check_args`) and Rusty `afterany` chains may be in flight.
- Do not `git pull` on a cluster while a continuation chain is queued: later jobs would pick up
  the new code.
- Metric to read: `eval/{raw,ema}/corr_gap` (train loss is not a quality signal in this loop).

## Checks

`uv run pytest -q` (CPU, ~1 min). Install the env with `UV_CACHE_DIR=/scratch/cm6627/.uv_cache uv sync --frozen`.
