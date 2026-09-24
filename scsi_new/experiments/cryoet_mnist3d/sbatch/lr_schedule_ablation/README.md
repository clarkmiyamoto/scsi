# LR / schedule ablation — digit 3, w20k / m5k

Follow-up to two `warmup_mstep_grid` runs that differed only in `--num_scsi_steps`:

| wandb run | `num_scsi_steps` | LR over EM 1–12 | outcome |
|---|---|---|---|
| `u6isb1wy` (w20k_m5k) | 12 | 2.58e-4 → 1e-5 (annealed out) | better panels |
| `n9n4qgae` (w20k_m5k) | 50 (walltime-killed at EM ~21) | 2.96e-4 → 2.42e-4 | worse |

(The long run was `num_scsi_steps 50`, not 100: that is the cluster-side edit to
`../warmup_mstep_grid/submit_three3d_w20k_m5k.SBATCH`, job 17324170, `TIMEOUT` at 24h.)

`main.py`'s single `CosineAnnealingLR` has `T_max = warmup + num_scsi_steps * mstep`, so
`num_scsi_steps` also sets the LR every EM step runs at. What "worse" looks like: by EM 20 the
N=50 run's six `viz/fixed/reconstruction` samples are nearly the same shape in the same pose.
That is one template regardless of y, i.e. EM collapse. Meanwhile its M-step loss kept falling
(0.036 vs 0.077), because collapsed posterior samples are easy to fit. **Train loss is not a
quality signal in this loop**, so every run here logs `eval/` metrics (below).

## Design

One shared warm start, then 8 arms that differ **only** in the EM-phase LR schedule.

- `submit_lrabl_warmup.SBATCH` trains the 20k-step warmup once (on the original 12-step
  recipe's cosine) and writes `checkpoints/warmup_w20k.pt`: model, EMA, AdamW moments, global
  step and RNG state.
- Every arm `--load_warmup_ckpt`s it. EM starts from identical weights, optimizer state and RNG
  stream in every arm, and no arm spends ~3h re-running warmup.
- Fixed everywhere: digit 3, `--n_images_per_class 23000`, batch sizes 16 / 16 / 32
  (warmup / M-step / E-step), `--mstep_n_steps_train 5000`, `--num_scsi_steps 24`, lift off.

| arm | flags | LR at EM1 start / EM12 end / EM24 end | question |
|---|---|---|---|
| `cos_h12` | `--lr_horizon_scsi_steps 12` | 2.58e-4 / 1e-5 / 1e-5 | baseline (original 12-step recipe, then held at eta_min) |
| `cos_h12_seed2` | same + `--em_seed 2` | same | noise floor |
| `cos_h24` | `--lr_horizon_scsi_steps 24` | 2.86e-4 / 1.23e-4 / 1e-5 | do more EM steps help once the anneal matches the run length? |
| `cos_h50` | `--lr_horizon_scsi_steps 50` | 2.96e-4 / 2.42e-4 / 1.47e-4 | replay of the long run's LR |
| `cos_h50_ema` | same + `--sample_with_ema` | same | does sampling with EMA weights substitute for annealing? |
| `const_1e-4` | `--lr_schedule constant --mstep_lr 1e-4` | 1e-4 flat | same mean LR over EM 1–12 as `cos_h12`, no anneal tail |
| `const_3e-5` | `--lr_schedule constant --mstep_lr 3e-5` | 3e-5 flat | bottom of the LR-magnitude sweep |
| `cos_per_mstep` | `--lr_schedule cosine_per_mstep` | 3e-4 → 1e-5 inside every M-step | anneal each M-step, never globally small |

How to read the arms against each other:

- **LR magnitude drives collapse** if quality orders `const_3e-5` ≥ `const_1e-4` ≈ `cos_h12` > `cos_h50`.
- **The anneal tail is what matters** if `cos_h12` clearly beats `const_1e-4` at EM 12, even
  though their mean LR is the same, and `cos_per_mstep` ≈ `cos_h12`.
- **EMA substitutes for annealing** if `cos_h50_ema` ≈ `cos_h12`. A long run then needs no
  LR tuned to its length.
- **Run length**: `cos_h24` vs `cos_h12` at EM 24.
- Differences smaller than `cos_h12` vs `cos_h12_seed2` are noise.

## Metric (`eval_metrics.py`)

Logged every EM step for **both** raw and EMA weights (`eval/raw/*`, `eval/ema/*`) on a fixed
32-item eval pool (fixed x0 and y, so it is deterministic given the weights):

- `corr_own`: Pearson r between x_hat and its own GT, maximized over SO(3) (8192 Haar-random
  coarse rotations + 4 rounds of local refinement). The model's frame is arbitrary, so the
  alignment lives inside the metric only; panels stay raw.
- `corr_other`: same, against a *different* item's GT.
- `corr_gap = corr_own - corr_other`: the y-specific part. **Read this one first.** Any
  y-independent output scores exactly 0. A mean-of-GT template already gets `corr_own` ≈ 0.83,
  so `corr_own` alone cannot flag collapse.

`eval_calib/*` in each run's summary is the perfect-recon reference (GT vs a randomly rotated
copy of itself): `corr_own` ≈ 0.98, `corr_gap` ≈ 0.26–0.28 depending on the pool.

## Where the code runs

The jobs run an isolated copy, `/scratch/cm6627/scsi_lr_ablation/scsi_new`, rsynced from the
local working tree (including its uncommitted `digit_scale` edits). It uses the existing
`/scratch/cm6627/scsi` venv via `uv run --project`, and `./data` is a symlink to the main
checkout's EMNIST. The cluster git checkout (`/scratch/cm6627/scsi`, at `fadcc9a`) is untouched.

```bash
cd /scratch/cm6627/scsi_lr_ablation/scsi_new/experiments/cryoet_mnist3d/sbatch/lr_schedule_ablation
bash submit_all.sh     # warmup, then all 8 arms with --dependency=afterok:<warmup>
```

Logs: `/scratch/cm6627/scsi_lr_ablation/logs/slurm-<job name>-<id>.out`. Each EM step prints
an `[em k/24] estep .. mstep .. lr ..` timing line and an `[em k] eval/...` line.
wandb project: `scsi-cryoet-mnist3d-three-lr-schedule`.

**Wall time, measured on an L40S** (interactive smoke run, real batch sizes and model):

| phase | L40S |
|---|---|
| warmup step | ~0.68 s (the `ResampledPairs` channel runs on CPU per sample) |
| E-step integration (batch 32 × 64 Euler steps) | ~27.6 s → 2000 samples ≈ 29 min |
| M-step | ~0.62 s/step → 5000 steps ≈ 52 min |
| panels + eval + checkpoint | ~1.3 min |

That comes to ~83 min per EM step and ~34h for 24 on an L40S. The earlier grid ran ~60 min per
EM step on A100s. Hence `--time=2-00:00:00` for the arms (the `gpu48` QoS max) and `12:00:00`
for the warmup (~4h). The warmup checkpoint is ~600 MB, as is each arm's `latest.pt`.

## Caveats

- **Not a rerun of the two runs above.** The cluster checkout predates `b41a04b`, so those
  runs used the older frozen-pair warm start. This ablation uses the current `ResampledPairs`
  warm start. Compare arms to each other, not to `u6isb1wy` / `n9n4qgae`.
- **GPU type is not pinned** (`--gres=gpu:1` lands on A100 / L40S / H100 nodes). This changes
  wall time and non-bitwise numerics, not the comparison; `cos_h12_seed2` bounds run-to-run
  noise.
- **~40 epochs per M-step.** 5000 steps × batch 16 over 2000 E-step samples lets the model
  memorize its own posterior samples: a plausible co-driver of collapse that this ablation
  does not vary. A natural follow-up is `--estep_num_samples 8000` or
  `--mstep_n_steps_train 2000`.
- `--warmup_lr` is still ignored: the warmup trains at `--mstep_lr`.
- `latest.pt` is written every EM step, but `main.py` cannot resume from it yet.
