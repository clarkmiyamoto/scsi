# mra_mnist — experiment journal

Newest first. Format and rules: [`../JOURNAL_TEMPLATE.md`](../JOURNAL_TEMPLATE.md).

> The entry below was **seeded on 2026-10-07** from the sweep README (date = README first
> commit). No outcome is recorded in the repo, so the result is `TBD`.

## 2026-08-19 — Warmup × M-step grid under the MRA channel (digit 3)

- **Status:** TBD (sbatch files target NYU; no outcome recorded in the repo)
- **Question:** how much pretraining (warmup and per-EM-step M-step training) does the
  multi-reference-alignment channel need (unknown SO(2) in-plane rotation + AWGN, no tilt
  series)?
- **Hypothesis:** not recorded.
- **Change from baseline:** baseline = `cryoet_mnist/warmup_mstep_grid` (same two axes, tilt-series
  channel). MRA channel instead; warmup ∈ {5k, 10k, 20k, 40k} × M-step ∈ {2k…20k} (no w5k/m2k
  cell), 40 EM steps, digit 3, 6000 images.
- **Setup:** sweep `mra_mnist/warmup_mstep_grid` (19 jobs), wandb
  `scsi-mra-mnist-three-warmup-mstep-grid`.
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** TBD
