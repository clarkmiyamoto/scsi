# LR schedule × warmup length × M-step length (Flatiron Rusty)

The first IgG-1D ablation with the pseudoinverse warm start. It crosses three factors:

| factor | flag | values |
|---|---|---|
| LR schedule | `--lr_schedule` | `cosine`, `constant`, `cosine_per_mstep` |
| warm-start steps | `--warmup_n_steps_train` | 30k, 40k |
| M-step steps per EM step | `--mstep_n_steps_train` | 5k, 10k |

That makes 12 arms, named `w<W>k_m<M>k_<cos|const|cospm>` (e.g. `w30k_m10k_cospm`).
- wandb project: `scsi-cryoet-igg1d-ablation`.
- Logs: `/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/logs/igg1d-<sched>-w<W>k-m<M>k_<jobid>.out`.
- Checkpoints: `.../checkpoints/sched_warmup_mstep/`.

## Order

1. **`debug.sh` on an interactive GPU node** (~40 min; see its header). It measures the warmup
   it/s and the E-step / M-step times, and checks warmup save → load → EM → resume for every
   schedule. Use its numbers to correct the wall-time table below.
2. **`bash submit_all.sh`**. It snapshots the code, submits the two warmups, then the 12 arms
   (each `afterok` on its warmup), each with one `afterany` continuation. `WARMUPS=`, `SCHEDS=`,
   `MSTEPS=` and `N_JOBS=` select a subset or change the number of continuations.

To (re)submit one arm by hand, give it the same snapshot its first job ran:
`CODE=<snapshot printed by submit_all.sh> sbatch --job-name=igg1d-cosine-w30k-m5k em.SBATCH cosine 30000 5000`.

## Design

**One shared warm start per warmup length.** `warmup.SBATCH W` trains it once and saves
`warmup_w<W>k.pt`: model, EMA, AdamW moments, global step, RNG state and `vol_gain`. All six arms
with that W load it, so they start EM from identical weights and differ only in their EM-phase
flags. The em-0 eval in the warmup job's wandb run is therefore the warm start's own score, and
the 30k vs 40k comparison is made there first.

**Warmup LR.** Both warmups use the base recipe's global cosine (`--lr_schedule cosine`, 12 EM
steps × 5000), as the mnist3d ablations did. Each arm rebuilds its own schedule at the restored
global step, so EM starts at:

| arm | LR at end of warmup | LR at EM 1 | then |
|---|---|---|---|
| `cos`, M=5k | 2.27e-4 (W=30k), 2.00e-4 (W=40k) | same: the un-split recipe | cosine to 1e-5 at EM 12 |
| `cos`, M=10k | same | 2.72e-4 (W=30k), 2.58e-4 (W=40k) | cosine to 1e-5 at EM 12 |
| `const` | same | 3e-4 | 3e-4 throughout |
| `cospm` | same | 3e-4 | 3e-4 → 1e-5 inside every M-step |

**Fixed everywhere** (main.py defaults unless stated):
- data: all ~98k training images at 64 px, `--pinv_diameter_A 192`, `vol_gain` calibrated
  by the warmup job;
- E-step: 2000 posterior samples per EM step, 64 Euler steps;
- batch sizes: 12 / 12 / 32 (warmup / M-step / E-step);
- optimisation: `--mstep_lr 3e-4`, `--eta_min 1e-5`, raw weights for sampling;
- 12 EM steps, `--lr_horizon_scsi_steps 12`, `--eval_n 16`, seed 42.

## Reading the results

Every EM step logs `eval/{raw,ema}/*` on the 16 held-out images cryofm's sampling eval uses.
`eval_calib/*` in each run summary is the GT ceiling.

- **Collapse:** `pred_conf_std` falling towards 0 means one conformation for every image, even
  when `train/loss` keeps falling. Train loss is not a quality signal in this loop.
- **Quality:** `conf_acc_within_3`, `conf_err_deg`, `corr_target_hand`, `fsc0.5_res_A`.
- **M-step length:** 5k / 10k steps at batch 12 are 30 / 60 epochs over the 2000 E-step samples.
  mnist3d saw collapse at ~40 epochs, so if the 10k arms collapse, suspect memorisation as well as
  the LR.
- The grid has no seed replicate. Before reading a small difference as real, add one:
  `SUFFIX=_seed2 sbatch em.SBATCH cosine 30000 5000 --em_seed 2` (with that arm's `CODE=`).

## Wall time and budget (estimated, not measured)

The rates are from the supervised cryofm job: 0.81 it/s at batch 12 in bf16 on an A100-80GB,
with the E-step assumed to cost ~1 h per 2000 samples (`../first_run/README.md`).

| job | estimate | `--time` |
|---|---|---|
| warmup 30k / 40k | ~10.5 h / ~14 h | 36 h |
| arm, M=5k | ~2.8 h per EM step, ~33 h | 3 days + 1 continuation |
| arm, M=10k | ~4.5 h per EM step, ~54 h | 3 days + 1 continuation |

That is ~550 GPU-hours in total. 12 arms run at once (the `gpu` QoS allows 16 GPUs and 288 CPUs
per user; this uses 12 and 192), so the grid finishes about 3 days after the warmups start, given
free GPUs. Each `latest.pt` is ~3 GB (model, EMA and AdamW moments of 180 M params), ~40 GB in all.

## Caveats

- **Volume scale.** `vol_gain` calibrates to 0.0131 at 64 px, which makes the warm-start volumes
  unit-std (`python data.py`). The GT-derived value is 0.0597. The pseudoinverse spreads each
  image through a 192 Å ball, so it is less concentrated than the particle. If EM reaches
  GT-like volumes, they will have std ~4.6 against the N(0, 1) base. To run in GT-std units
  instead, append `--vol_gain 0.0597` to the `main.py` call in `warmup.SBATCH`. The arms take
  `vol_gain` from the checkpoint.
- **What the warm start is.** It is pose-blind: one Wiener CTF-corrected image smeared through a
  ball (see `pseudoinverse.py`). That makes it a symmetry-breaking start, not a reconstruction.
  `python pseudoinverse.py` scores it against GT: r 0.42 with the image-frame target, 0.16 with
  another image's.
- **The code is a snapshot.** Jobs run `~/scsi_snapshots/igg1d_sched_warmup_mstep_<time>/`, so
  checkout edits do not reach queued jobs. A hand-submitted continuation needs the same `CODE=`.
- **`--resume` redoes the interrupted EM step.** Warmups are not checkpointed mid-way; a killed
  warmup reruns from scratch and its arms never start (`afterok`).
- **If a warmup fails**, its arms pend with `DependencyNeverSatisfied`. Cancel them and their
  continuations together: once an arm's first job is gone, `afterany` starts the continuation,
  which then fails on the missing `warmup_w<W>k.pt`.
