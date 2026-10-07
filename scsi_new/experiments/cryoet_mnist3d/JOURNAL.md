# cryoet_mnist3d — experiment journal

Newest first. Format and rules: [`../JOURNAL_TEMPLATE.md`](../JOURNAL_TEMPLATE.md).
Primary metric: `eval/{raw,ema}/corr_gap` (0 for any y-independent output; ceiling =
`eval_calib/corr_gap`, ≈0.26 for digit 3).

> Entries below dated before 2026-10-07 were **seeded on 2026-10-07** from the sweep READMEs and
> `RECIPIES.md`. Dates are the README's first-commit date. Results are filled in only where the
> README recorded them; everything else is `TBD`, not inferred. The READMEs stay the detailed
> record (resources, caveats, wall time).

## 2026-10-01 — Fresh student per M-step: lr × training length (warmup = M-step)

- **Status:** TBD (sbatch files target Rusty; no outcome recorded in the repo)
- **Question:** if every EM step trains a freshly initialised student instead of fine-tuning the
  teacher, how do the M-step lr and the training budget trade off, with the warm start getting
  the same budget as each student?
- **Hypothesis:** not recorded.
- **Change from baseline:** baseline = the teacher-mode digit-3 recipe (`warmup_mstep_grid`).
  `--student_init fresh --warmup_lr 3e-4`; `--warmup_n_steps_train` = `--mstep_n_steps_train` =
  N ∈ {10k, 25k, 40k, 60k}; `--mstep_lr` ∈ {1e-4, 3e-4, 1e-3}; `--num_scsi_steps 12`, `--resume`.
- **Setup:** sweep `cryoet_mnist3d/fresh_lr_tied_steps_grid` (12 jobs), wandb
  `scsi-cryoet-mnist3d-three-fresh-lr-steps`. Design: `sbatch/fresh_lr_tied_steps_grid/README.md`.
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** compare against `cos_h12` of the LR-schedule ablation at equal EM step.

## 2026-09-24 — Ten classes: E-step size, M-step length, warmup, schedule

- **Status:** TBD (sbatch files target NYU; no outcome recorded in the repo)
- **Question:** how should E-step samples per iteration, M-step steps, warmup length and the LR
  schedule scale when going from digit 3 to all ten digits?
- **Hypothesis (from the README):** per-EM-step weight movement (LR × steps) drives the late
  decay seen in the digit-3 ablation, so grow the E-step (more data, same movement) before
  growing M-step steps; warmup length should grow sublinearly with class count (prior, not
  measured). Noise floor = `s4k_m5k` vs `s4k_m5k_seed2`.
- **Change from baseline:** baseline = `cos_h12` of the entry below. Digits 0–9,
  `--n_images_per_class 5000`, shared w40k warm start; arms vary `--estep_num_samples`
  (2k/4k/8k), `--mstep_n_steps_train` (5k/10k), schedule (`cosine` h12/h24, `cosine_per_mstep`),
  `--em_seed 2`; side jobs warm up 20k and 80k steps.
- **Setup:** sweep `cryoet_mnist3d/tenclass_sweep` (10 jobs), wandb
  `scsi-cryoet-mnist3d-ten-sweep`, `--resume` with one afterany continuation per arm.
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** decided by this sweep.

## 2026-09-24 — LR / schedule ablation (digit 3, w20k / m5k)

- **Status:** done (arms ran on NYU; numbers below are from the ten-class README, pulled from
  wandb 2026-09-24)
- **Question:** why did the 50-EM-step run (`n9n4qgae`) look worse than the 12-step run
  (`u6isb1wy`) in the 3D warmup × M-step grid? The two differed only in `--num_scsi_steps`,
  which also sets the LR via the global cosine (2.58e-4 → 1e-5 vs 2.96e-4 → 2.42e-4 over EM 1–12).
- **Hypothesis:** LR magnitude drives collapse, i.e. quality orders `const_3e-5` ≥ `const_1e-4` ≈
  `cos_h12` > `cos_h50`. Rival: the anneal tail is what matters. Differences under the
  `cos_h12` vs `cos_h12_seed2` gap are noise.
- **Change from baseline:** baseline = `cos_h12` (original 12-step recipe, then held at
  `eta_min`). Arms differ only in the EM-phase schedule off one shared 20k warm start:
  `--lr_schedule`, `--lr_horizon_scsi_steps` (12/24/50), `--sample_with_ema`, `--mstep_lr`
  (constant 1e-4 / 3e-5), `--em_seed 2`; 24 EM steps.
- **Setup:** sweep `cryoet_mnist3d/lr_schedule_ablation` (9 jobs), wandb
  `scsi-cryoet-mnist3d-three-lr-schedule`.
- **Result:** `eval/raw/corr_gap` (perfect = 0.259):

  | arm | EM 11 | EM 19 | EM 24 |
  |---|---|---|---|
  | `cos_h12` | 0.071 | 0.090 | 0.096 |
  | `cos_h12_seed2` | 0.056 | 0.082 | — |
  | `cos_per_mstep` | 0.076 (peak) | 0.055 | — |
  | `const_1e-4` | 0.067 | 0.054 | — |
  | `const_3e-5` | 0.046 | 0.078 | — |
  | `cos_h24` | 0.055 | 0.059 | 0.068 |
  | `cos_h50` | 0.051 | 0.042 | 0.046 |

  Seed-to-seed noise ≈ 0.01–0.017. `cos_h50_ema` is not in the README table (TBD).
- **Interpretation (README's):** every arm that keeps a high average LR peaks around EM 8–11 and
  then decays; annealing globally and holding at `eta_min` keeps improving. Annealing inside each
  M-step does not help. Rank is driven by how far the weights keep moving across EM steps
  (LR × steps), not by the anneal alone. Train loss is **not** a quality signal here (collapsed
  samples are easy to fit): the 50-step run's M-step loss fell to 0.036 vs 0.077 while its
  samples collapsed to one template.
- **Next:** the ten-class sweep above; README also suggests `--estep_num_samples 8000` or
  `--mstep_n_steps_train 2000` (the model memorises ~2000 samples at ~40 epochs per M-step).

## 2026-09-09 — Supervised upper bound, lifted target (`--lift`)

- **Status:** TBD (sbatch files target NYU; no outcome recorded in the repo)
- **Question:** what is the true upper bound for the pose-agnostic SCSI loop? `y = F(X)` is
  pose-blind, so training toward the canonical `X` asks for an orientation `y` does not fix.
- **Hypothesis:** the lifted target `(R·X, F(X))`, R a fresh Haar SO(3) per volume, is the
  relevant bound; `x_hat` will come out in a random orientation.
- **Change from baseline:** baseline = the 2026-09-08 entry. `--lift` added.
- **Setup:** sweep `cryoet_mnist3d/supervised_arch_interp_lift` (4 jobs: `{dit,unet}` ×
  `{gvp,linear}`), wandb `scsi-cryoet-mnist3d-supervised-arch-interp-lift`, 300k steps.
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** compare with the unsupervised `corr_gap` curves above.

## 2026-09-08 — Supervised upper bound: backbone × interpolant

- **Status:** TBD (sbatch files target NYU; no outcome recorded in the repo)
- **Question:** paired-data ceiling on digit 3 for the DiT (full 3D attention, ~33M params) vs
  the factorised (2+1)D video-UNet (~37M), each with the GVP vs linear interpolant.
- **Hypothesis:** not recorded.
- **Change from baseline:** baseline = none (first supervised run). `main_supervised.py`,
  canonical target, 300k steps, batch 16, EMA panels, ODE-step sweep 16…1024.
- **Setup:** sweep `cryoet_mnist3d/supervised_arch_interp` (4 jobs), wandb
  `scsi-cryoet-mnist3d-supervised-arch-interp`.
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** the lifted variant above.

## 2026-08-27 — 3D warmup × M-step grid (digit 3)

- **Status:** done for the cells whose outcome is recorded
- **Question:** how much warmup and M-step training does the 3D→2D tilt-series problem need?
- **Hypothesis:** not recorded.
- **Change from baseline:** baseline = the 2D grid (`cryoet_mnist/warmup_mstep_grid`). EMNIST
  digit 3, 23k volumes, batch 16/16/32, `--num_scsi_steps 12`; warmup ∈ {10k, 20k, 40k} ×
  M-step ∈ {2k, 5k, 10k}.
- **Setup:** sweep `cryoet_mnist3d/warmup_mstep_grid` (9 jobs), wandb
  `scsi-cryoet-mnist3d-three-warmup-mstep-grid`.
- **Result:** by panels only (no scalar metric yet): `u6isb1wy` (`w20k_m5k`, 12 EM steps) gave
  better panels than `n9n4qgae` (`w20k_m5k`, edited to 50 steps, killed by the 24h walltime at EM
  ~21), whose six fixed samples had collapsed to one template. Other cells: TBD.
- **Interpretation:** the 12- vs 50-step gap was an LR-schedule effect, followed up in the
  2026-09-24 ablation. The `w40k_m2k` cell is a degenerate corner (62.5% of the cosine spent in
  warmup).
- **Next:** the LR / schedule ablation.
