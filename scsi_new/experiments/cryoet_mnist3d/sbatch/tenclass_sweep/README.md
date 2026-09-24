# Ten-class sweep: digits 0–9, following the digit-3 LR ablation

This moves the digit-3 recipe (`../lr_schedule_ablation/`) to all ten EMNIST digits. It asks
how three things should scale with the number of classes: the E-step / M-step size per EM
iteration, the warmup length, and the LR schedule. wandb project: `scsi-cryoet-mnist3d-ten-sweep`.

## What the digit-3 ablation showed

`eval/raw/corr_gap` pulled from wandb on 2026-09-24. A perfect reconstruction scores
`eval_calib/corr_gap` = 0.259.

| arm | EM 11 | EM 19 | EM 24 | shape |
|---|---|---|---|---|
| `cos_h12` | 0.071 | 0.090 | **0.096** | the only arm still climbing at EM 24, held at eta_min from EM 12 |
| `cos_h12_seed2` | 0.056 | 0.082 | — | seed-to-seed noise ≈ 0.01–0.017 |
| `cos_per_mstep` | **0.076** (peak) | 0.055 | — | decays after EM 11 while `corr_own` rises (0.69 → 0.73): drift toward a template |
| `const_1e-4` | 0.067 | 0.054 | — | same shape as `cos_per_mstep` |
| `const_3e-5` | 0.046 | 0.078 | — | slow, still rising |
| `cos_h24` | 0.055 | 0.059 | 0.068 | drifts, recovers partly once annealed |
| `cos_h50` | 0.051 | 0.042 | 0.046 | drifts |

`cos_per_mstep` looked good only up to about EM 11. Over the full run, what decides the outcome is
how fast the weights keep moving across EM steps, i.e. LR × steps. Annealing inside each M-step
does not help. Every arm that keeps a high average LR peaks around EM 8–11 and then decays.
Annealing globally and then holding at `eta_min` keeps improving.

## How the knobs should scale with 10 classes

**Steps per EM iteration: scale them with the E-step size, and grow the E-step first.** Ten
classes change the E-step most. At `--estep_num_samples 2000` each digit gets ~200 posterior
samples per EM step, a tenth of what digit 3 had. `--mstep_n_steps_train` controls two things:

- epochs over those samples (40 in the digit-3 recipe, which already risks memorizing them);
- how far the weights move per EM step.

The ablation only varied LR, not M-step length. Its arms decayed late whenever the LR stayed
high across EM steps. So the working hypothesis is that per-EM-step movement (LR × steps) drives
the decay, which would make more steps as risky as more LR. Hence: hold M-step steps at 5k and
grow the E-step to 4k–8k samples. That gives more data per EM step with the same movement, and
20–10 epochs instead of 40. `s4k_m10k` tests the hypothesis: steps proportional to samples, i.e.
constant epochs.

**Warmup: expected sublinear in the number of classes; 2× (40k) as the default.** This is a
prior, not a measurement. Most of what warmup learns should be shared across digits: inverting
the tilt-series channel and denoising. Only the shape prior is per-class. The w20k / w40k / w80k side jobs measure this directly from their em-0
`eval/`. Longer warmup also uses up more of the global cosine: at m5k/h12, EM starts at 2.58e-4
after w20k, 2.0e-4 after w40k and 1.23e-4 after w80k. That is why every arm shares one warmup
instead of comparing warmups downstream.

**LR schedule: global cosine to `eta_min` by EM 12, then hold (`cos_h12`).** Keep the horizon
in EM steps. `s4k_m5k_h24` tests whether 10 classes need a longer high-LR phase.
`s4k_m5k_permstep` rechecks `cos_per_mstep`. To extend a run past 24 EM steps, raise
`--num_scsi_steps` and resubmit. `--resume` continues at `eta_min`, as digit 3's EM 13–24 did.

## Design

One shared warm start (`submit_ten_warmup_w40k.SBATCH` → `checkpoints/warmup_w40k.pt`), then 7
arms that `--load_warmup_ckpt` it. Shared by every job:

- `--digit_classes 0 1 2 3 4 5 6 7 8 9`, passed explicitly. The checkpoint check treats `None`
  and `[0..9]` as different.
- `--n_images_per_class 5000` (50k volumes). The E-step draws at most 8k per EM step, so more
  would add RAM, not signal.
- Batch sizes 16 / 16 / 32 (warmup / M-step / E-step), lift off.
- `--eval_n 100` (10 per digit), `--viz_n_pool 30 --viz_n_display 10`. The fixed panel shows
  one of each digit, 0–9.

| arm | `estep_num_samples` | `mstep_n_steps_train` | schedule | samples / digit / EM | epochs / M-step | LR at EM 1 start → EM 12 end | est. L40S hours (24 EM) |
|---|---|---|---|---|---|---|---|
| `s2k_m5k` | 2000 | 5000 | cosine h12 | 200 | 40 | 2.0e-4 → 1e-5 | 33 |
| `s4k_m5k` | 4000 | 5000 | cosine h12 | 400 | 20 | 2.0e-4 → 1e-5 | 45 |
| `s4k_m5k_seed2` | 4000 | 5000 | cosine h12, `--em_seed 2` | 400 | 20 | same | 45 |
| `s8k_m5k` | 8000 | 5000 | cosine h12 | 800 | 10 | 2.0e-4 → 1e-5 | 68 |
| `s4k_m10k` | 4000 | 10000 | cosine h12 | 400 | 40 | 2.58e-4 → 1e-5 | 66 |
| `s4k_m5k_h24` | 4000 | 5000 | cosine h24 | 400 | 20 | 2.58e-4 → 9.95e-5 (1e-5 at EM 24) | 45 |
| `s4k_m5k_permstep` | 4000 | 5000 | `cosine_per_mstep` | 400 | 20 | 3e-4 → 1e-5 inside every M-step | 45 |

Side jobs, with nothing chained on them: `warmup_w20k` and `warmup_w80k`, each warmup plus the
em-0 eval, with checkpoints saved for follow-up arms.

How to read the arms:

- **E-step size**: `s2k_m5k` vs `s4k_m5k` vs `s8k_m5k` share the same LR curve and M-step, so
  they differ only in posterior samples per EM step.
- **Steps per EM step**: `s4k_m10k` vs `s4k_m5k`. The m10k arm moves further per EM step in two
  ways: twice the steps, and a higher LR curve, because the global cosine's warmup share
  shrinks. Its LR-vs-EM-step curve is exactly digit-3 `cos_h12`'s. If it loses to `s4k_m5k`,
  more movement per EM step hurts at 10 classes too. This arm cannot separate steps from LR.
- **Schedule**: `s4k_m5k_h24` and `s4k_m5k_permstep` vs `s4k_m5k`.
- **Warmup**: em-0 `eval/` of `warmup_w20k` / `warmup_w40k` / `warmup_w80k`.
- **Noise floor**: differences smaller than `s4k_m5k` vs `s4k_m5k_seed2` are noise.

## Metric (`eval_metrics.py`, updated for multi-class)

Rank by `eval/{raw,ema}/corr_gap` as before. A class-ordered pool would have mixed same-class
and cross-class pairs, so pairing is now by label:

- `corr_gap` = `corr_own` − `corr_other`, where `corr_other` scores against a **different
  instance of the same digit**. It measures instance specificity: 0 for any y-independent
  template, and ~0 for a per-digit template. On a single-class pool it is the old `roll(1)`
  value exactly (checked).
- `corr_gap_xcls` scores against a **different digit**. It measures class identity. A per-digit
  template has `corr_gap` ≈ 0 but `corr_gap_xcls` > 0. A single template has both ≈ 0.
- `by_class/<d>/{corr_own,corr_gap}` report 10 items per digit. They are noisy, but show digits
  EM never resolves.
- Read everything against `eval_calib/*` in the run summary: a perfect reconstruction in an
  unknown frame. The ceilings differ from digit 3's, so compare arms with each other, not with
  the digit-3 numbers.

## Code this sweep relies on

- `build_viz_pool` (`data.py`) picks a class-balanced pool interleaved 0, 1, …, 9, 0, 1, …
  and returns `label`. Single-class pools are bit-identical to before (checked).
- `--resume` (`main.py`): with `--ckpt_dir DIR`, continues from `DIR/latest.pt` if it exists,
  restoring model, EMA, AdamW, global + EM step, RNG and the same wandb run. It rebuilds the LR
  schedule at the restored step, refuses a `latest.pt` with different data / model / EM args,
  and exits at once if the run is already done. Tested on CPU: 4 EM steps straight vs 2 +
  resume to 4 gives bit-identical weights, EMA, Adam moments and RNG.

## Where the code runs

An isolated copy, `/scratch/cm6627/scsi_tenclass/scsi_new`, rsynced from the local working tree,
which holds the uncommitted changes above. The lr ablation's copy
(`/scratch/cm6627/scsi_lr_ablation`) still has arms running, so leave it alone. The copy uses
the existing `/scratch/cm6627/scsi` venv.

```bash
# laptop (rides the ssh ControlMaster)
rsync -a --exclude '__pycache__/' --exclude 'wandb/' --exclude 'data/' --exclude '*.pt' \
    ~/nyu/scsi/scsi_new/ torch:/scratch/cm6627/scsi_tenclass/scsi_new/
# cluster: reuse the lr ablation's EMNIST symlink, then submit
E=/scratch/cm6627/scsi_tenclass/scsi_new/experiments/cryoet_mnist3d
ln -s "$(readlink -f /scratch/cm6627/scsi_lr_ablation/scsi_new/experiments/cryoet_mnist3d/data)" $E/data
cd $E/sbatch/tenclass_sweep && bash submit_all.sh
```

`submit_all.sh` submits the w40k warmup, the w20k / w80k side warmups, and every arm with
`--dependency=afterok:<w40k>`. It also chains one continuation per arm with
`--dependency=afterany` (`N_JOBS=2`, i.e. up to 96h). Logs go to
`/scratch/cm6627/scsi_tenclass/logs/slurm-<job name>-<id>.out`; each EM step prints an
`[em k/24] estep .. mstep .. lr ..` line and an `[em k] eval/...` line.

## Wall time and resources

Estimates are extrapolated from the ablation's L40S measurements, not measured for 10 classes:
E-step ~0.86 s per posterior sample (batch 32 × 64 Euler steps), M-step ~0.62 s per step, and
~5 min per EM step for panels and eval at `eval_n 100` (100 items × 64 Euler steps, raw + EMA,
integrated in E-step-sized chunks). A100 / H100 nodes are faster. Warmup is
~0.68 s per step: w20k ≈ 4h, w40k ≈ 8h, w80k ≈ 15h. The whole sweep is ~375 GPU-hours
(L40S worst case). It runs up to 10 jobs at once, within the gpu48 16-GPU cap, but the
ablation's last arms also count until they finish.

`--mem=64G`: the host-RAM peak of the 10-class data pipeline (`build_observations`, both
viz / eval pools, `build_warmup`) at 5000 per class, measured locally, is 18.3 GB. It takes
about 6 min before training starts, on every job including continuations. The ~600 MB
checkpoint and the CUDA context add a few GB on top.

## Caveats

- **Resume redoes the interrupted EM step.** Continuation jobs re-log `train/` steps the killed
  job already logged. wandb drops those with a warning until the step count passes where the
  killed job stopped, so that stretch of `train/loss` comes from the first attempt.
- If the w40k warmup fails before writing `warmup_w40k.pt`, `scancel` the arms **and** their
  continuations (see `submit_all.sh`). If it fails after writing it, in the em-0 panels / eval,
  release the arms instead of resubmitting: `scontrol update jobid=<arm id> dependency=`.
- **GPU type is not pinned** (`--gres=gpu:1`). Batch 16 peaks near 37 GB, so a card under
  40 GB OOMs. Node type changes wall time and non-bitwise numerics; the seed2 arm bounds the
  latter.
- `--warmup_lr` is still ignored: the warmup trains at `--mstep_lr`.
