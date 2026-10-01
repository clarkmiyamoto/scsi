# 3D fresh-student lr x training-length grid (warmup = M-step), digit 3 only

12 SBATCH jobs (Rusty), one per cell of a 3x4 grid, all with **`--student_init fresh`**: after
every E-step the teacher is discarded and a newly initialized student is trained from scratch on
the E-step samples. 3D counterpart of `../../../cryoet_mnist/sbatch/fresh_lr_warmup_mstep_grid/`.
The difference: the warmup and M-step lengths are **tied** (`--warmup_n_steps_train` =
`--mstep_n_steps_train`), so every model in a run, the warm start and each fresh student, gets the
same training budget.

Reconstructs **only EMNIST digit 3** (`--digit_classes 3`, `--n_images_per_class 23000`) under the
3D->2D tilt-series channel in `experiments/cryoet_mnist3d/main.py`. Fixed everywhere:
`--num_scsi_steps 12`, `--warmup_lr 3e-4`, batch sizes 16 / 16 / 32 (warmup / M-step / E-step, as
in `../warmup_mstep_grid/`), `--estep_num_samples 2000`, `--eta_min 1e-5` and `--eval_n 32`
(defaults).

Swept:

| axis | flag | values |
|---|---|---|
| M-step learning rate (peak of each student's cosine) | `--mstep_lr` | 1e-4 / 3e-4 / 1e-3 |
| warmup = M-step training length | `--warmup_n_steps_train` = `--mstep_n_steps_train` | 10,000 / 25,000 / 40,000 / 60,000 |

File naming: `submit_fresh3d_lr{lr}_w{N}k_m{N}k.SBATCH`. Job name `fresh3d_lr{lr}_w{N}k_m{N}k`,
wandb run name `lr{lr}_w{N}k_m{N}k`, all in the `scsi-cryoet-mnist3d-three-fresh-lr-steps` wandb
project. Unlike the 2D grid these runs log `eval/{raw,ema}/corr_gap`, so cells can be ranked by a
number rather than by eye. A perfect reconstruction's ceiling is `eval_calib/corr_gap` in the
run summary.

## How the LR schedule works in fresh mode

Same as the 2D grid. Each model trains on its own schedule:

- **Warmup:** its own AdamW at `--warmup_lr 3e-4`, cosine down to `--eta_min` over
  `--warmup_n_steps_train` steps. It doesn't depend on `--mstep_lr`, so the 3 cells of the same
  length train the same warmup (same seed) and their EM-step-0 panels and `eval/` should match up
  to GPU nondeterminism.
- **Each M-step:** a new model, a new AdamW at `--mstep_lr`, and a new EMA, with a cosine down to
  `--eta_min` over `--mstep_n_steps_train` steps. `train/lr` in wandb looks like a sawtooth.

`--lr_schedule` / `--lr_horizon_scsi_steps` don't apply in fresh mode (`args.py` refuses them).

## Submitting

On a Rusty login node:

```bash
./submit_all.sh                                   # all 12 cells
./submit_all.sh submit_fresh3d_lr3e-4_*.SBATCH    # a subset, e.g. just the lr 3e-4 row
```

Each cell is queued twice (`N_JOBS=2`): the first job, then a continuation of the same file
with `--dependency=afterany`. `--ckpt_dir` + `--resume` make the continuation pick up from that
cell's `latest.pt` after a walltime kill, or exit within a minute if the cell already finished.
The script also syncs the uv env and checks for EMNIST before queuing anything.

Logs: `cryoet_mnist3d/logs/slurm-<jobname>-<jobid>.{out,err}`. Every EM step prints
`[em k/12] estep ..s  mstep ..s  viz+eval+ckpt ..s  lr ..` and an `[em k] eval/...` line.
Use those lines to check wall time against `--time`.

To check that Slurm accepts the resource request without queuing anything:
`sbatch --test-only submit_fresh3d_lr3e-4_w25k_m25k.SBATCH`.

## Compute and walltime

Not measured on Rusty. Extrapolated from the L40S numbers in `../tenclass_sweep/README.md`:
~0.68 s per warmup step, ~0.62 s per M-step step, ~0.86 s per E-step posterior sample (2000 per
EM step, so ~29 min), plus a few minutes per EM step for panels and eval. H100s should be faster.

| N = warmup = M-step | est. hours (L40S-class) | `--time` |
|---|---|---|
| 10k | ~29 | 2 days |
| 25k | ~62 | 4 days |
| 40k | ~96 | 6 days |
| 60k | ~141 | 7 days (Rusty `gpu` QoS max) |

That is ~330 GPU-hours per LR row and ~1000 for the whole grid. The `gpu` QoS caps a user at 16
GPUs, so all 12 cells can run at once.

## Resources and storage

- `--partition=gpu --gres=gpu:1 --constraint=rocky9&(h100|a100-80gb)`, as in the 2D grid. Batch 16
  peaks near 37 GB, so the constraint also keeps jobs off the 20 GB A100 MIG slices.
- `--mem=48G`: one class at 23k images is a ~3 GB volume pool, with ~9 GB peak host RSS measured
  for `../warmup_mstep_grid/`.
- Checkpoints (`latest.pt`, ~600 MB, overwritten every EM step) go to
  `/mnt/ceph/users/cmiyamoto/scsi_checkpoints/cryoet_mnist3d/fresh_lr_tied_steps_grid/<run>/`,
  ~7 GB for the grid.
- EMNIST (~2.3 GB) lives in `/mnt/ceph/users/cmiyamoto/scsi_data/cryoet_mnist3d/`.
  `cryoet_mnist3d/data` is a symlink to it.

## Known caveats

- **Walltime is an estimate, not a measurement.** See above. If the first `[em k/12]` lines show
  the big cells won't fit, the continuation covers one more `--time` window. Raise `N_JOBS` for
  more.
- **A resume redoes the interrupted EM step.** In fresh mode that means the interrupted student
  retrains from scratch: up to a full M-step (60k steps for the largest cells) is lost per kill.
  The continuation's `train/` logs for that stretch are dropped by wandb until the step count
  passes where the killed job stopped.
- **`afterany` also fires after a crash** (for example an OOM), so a crashing cell is retried once.
- **Longer cells also mean more epochs.** Each fresh student sees `N x 16` examples drawn from
  2000 posterior samples: 80 epochs at 10k, 480 at 60k. The length axis therefore also measures
  how far a student overfits one E-step's samples. The 2D grid's longest cells reached ~500
  epochs.
- **`--warmup_lr` is fixed at 3e-4** while `--mstep_lr` varies, as in the 2D grid. In the
  lr 1e-4 / 1e-3 rows, the students train at a different LR than the warm start did.
- **The warm start is pose-blind** (`../warmup_mstep_grid/README.md`): EM-step 0 is a rough
  symmetry-breaking seed, not an upper bound.
- **Not directly comparable to the `teacher`-mode 3D grids at EM step 0.** In `teacher` mode
  the warmup runs at `--mstep_lr` under one cosine that spans the whole run. Here it doesn't.
