# cryoet_igg1d — experiment journal

Newest first. Format and rules: [`../JOURNAL_TEMPLATE.md`](../JOURNAL_TEMPLATE.md).
Cluster: Rusty (the sbatch scripts and data paths point at `/mnt/ceph`, `/mnt/home`).

> Entries below were **seeded on 2026-10-07** from the sweep READMEs and `RECIPIES.md` (dates =
> first commit). Wall times in those READMEs are estimates; results are `TBD` unless a number is
> quoted from the repo.

## 2026-10-01 — LR schedule × warmup length × M-step length (pseudoinverse warm start)

- **Status:** TBD (scripts written; no outcome recorded in the repo)
- **Question:** with the pseudoinverse warm start, how do the LR schedule, warm-start steps and
  M-step steps per EM step affect reconstruction of IgG-1D conformations?
- **Hypothesis:** not recorded. README's reading guide: `pred_conf_std` falling towards 0 means
  collapse to one conformation for every image.
- **Change from baseline:** baseline = `first_run` (no warm start). `--lr_schedule` ∈ {cosine,
  constant, cosine_per_mstep}; `--warmup_n_steps_train` ∈ {30k, 40k} (one shared warm start per
  length); `--mstep_n_steps_train` ∈ {5k, 10k}; 12 arms, 12 EM steps, `--lr_horizon_scsi_steps 12`.
- **Setup:** `sbatch/sched_warmup_mstep/` (hand-written parametrised `em.SBATCH`, not yet in
  `sweeps.py`), wandb `scsi-cryoet-igg1d-ablation`.
- **Result:** TBD (warm-start check recorded in `RECIPIES.md`: corrected image r 0.70 with the GT
  projection, volume r 0.42 with its image-frame target and 0.16 with another image's)
- **Interpretation:** TBD
- **Next:** TBD

## 2026-10-01 — First IgG-1D EM run, no warm start

- **Status:** TBD (script written; no outcome recorded in the repo)
- **Question:** can EM from a randomly initialised network make progress on IgG-1D at 64 px?
  Also the no-warm-start baseline for the ablation above.
- **Hypothesis:** not recorded; README expects the EM-1 E-step samples (untrained net) to be
  close to Gaussian noise.
- **Change from baseline:** baseline = none. `--warmup_n_steps_train 0 --vol_gain 0.0602` (the
  GT-derived gain), 12 EM steps × 2000 posterior samples × 5000 M-step steps, one global cosine.
- **Setup:** `sbatch/first_run/em.SBATCH`, wandb `scsi-cryoet-igg1d`. Check forward model first:
  `python -m scsi_new.experiments.cryoet_igg1d.data` (matched correlation 0.349 vs 0.351
  noiseless ceiling, residual std 1.003, per `RECIPIES.md`).
- **Result:** TBD
- **Interpretation:** TBD
- **Next:** the warm-start ablation above.
