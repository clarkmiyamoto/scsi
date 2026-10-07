# cryoet_mnist — experiment journal

Newest first. Format and rules: [`../JOURNAL_TEMPLATE.md`](../JOURNAL_TEMPLATE.md).

> Entries dated before 2026-10-07 were **seeded on 2026-10-07** from the sweep READMEs and
> `RECIPIES.md`; dates are the README's first-commit date. Results appear only where the repo
> recorded them; everything else is `TBD`, not inferred.

## 2026-10-01 — Fresh student per M-step: lr × warmup × M-step length (digit 3)

- **Status:** TBD (sbatch files target Rusty; no outcome recorded in the repo)
- **Question:** if each M-step trains a newly initialised student instead of fine-tuning the
  teacher, how do the M-step lr, the warmup length and the M-step length interact?
- **Hypothesis:** not recorded. README flags a risk: short M-steps (m5k, especially at lr 1e-4)
  may underfit a from-scratch student.
- **Change from baseline:** baseline = `warmup_mstep_grid` (teacher mode, 2026-08-19).
  `--student_init fresh --warmup_lr 3e-4`; `--mstep_lr` ∈ {1e-4, 3e-4, 1e-3}, warmup ∈ {10k, 20k,
  40k}, M-step ∈ {5k, 10k, 20k}. Warmup and M-step each get their own cosine, so the axes no
  longer leak into each other's LR schedule (they did in the earlier grids).
- **Setup:** sweep `cryoet_mnist/fresh_lr_warmup_mstep_grid` (27 jobs), wandb
  `scsi-cryoet-mnist-three-fresh-lr-warmup-mstep`.
- **Result:** TBD
- **Interpretation:** TBD (no scalar metric in the 2D experiment: rank by panels)
- **Next:** TBD

## 2026-08-21 — Best launch recipe so far (RECIPIES.md)

- **Status:** done (informal note)
- **Question:** which settings give the best reconstructions?
- **Hypothesis:** the sample count and training steps needed are much larger than expected.
- **Change from baseline:** `--n_images_per_class 6000 --warmup_n_steps_train 40000
  --mstep_n_steps_train 5000`.
- **Setup:** manual launch; run ids not recorded.
- **Result:** "the best performance so far" (qualitative, as written in `RECIPIES.md`). With a DiT
  backbone the samples show ghost/residual pixelation matching the `patchify` size.
- **Interpretation:** as written above; confidence unknown.
- **Next:** the two grids below map warmup × M-step length systematically.

## 2026-08-20 — Warmup × M-step grid, all ten digits

- **Status:** TBD (sbatch files target NYU; no outcome recorded in the repo)
- **Question:** does the digit-3 grid's picture of "how much pretraining" hold with all ten
  MNIST digits (60k images)?
- **Hypothesis:** not recorded.
- **Change from baseline:** baseline = the 2026-08-19 grid. `--digit_classes` omitted (all ten).
- **Setup:** sweep `cryoet_mnist/multiple_digits` (20 jobs), wandb
  `scsi-cryoet-mnist-multi-warmup-mstep-grid`.
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** TBD

## 2026-08-19 — Warmup × M-step grid, digit 3

- **Status:** TBD (sbatch files target NYU; no outcome recorded in the repo)
- **Question:** how many warmup and per-EM-step M-step training steps does the tilt-series
  channel need?
- **Hypothesis:** not recorded.
- **Change from baseline:** none (first grid). Digit 3, 6000 images, 40 EM steps; warmup ∈ {5k,
  10k, 20k, 40k} × M-step ∈ {2k, 5k, 10k, 15k, 20k}. README caveats: one global cosine couples
  the two axes, and `--warmup_lr` had no effect in teacher mode.
- **Setup:** sweep `cryoet_mnist/warmup_mstep_grid` (20 jobs), wandb
  `scsi-cryoet-mnist-three-warmup-mstep-grid`.
- **Result:** TBD (the 2026-08-21 recipe note is the only recorded outcome)
- **Interpretation:** TBD
- **Next:** the all-digits and fresh-student grids above.
