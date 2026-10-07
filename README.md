# scsi
Implementation of "Self Consistent Stochastic Interpolants".

- `scsi_new/` — the package (EM loop, interpolants, ODE/SDE) and `experiments/<name>/` (one
  directory per problem: data, forward model, network, `main.py`).
- Run an experiment from the repo root: `uv run python -m scsi_new.experiments.cryoet_mnist3d.main --help`.
- Sweeps are defined in `experiments/<name>/sweeps.py` and turned into sbatch scripts for either
  cluster with `uv run python -m scsi_new.sbatch_gen` (`--list` to see them).
- Each experiment keeps a `JOURNAL.md` (question → hypothesis → result → next); see
  `scsi_new/experiments/JOURNAL_TEMPLATE.md`.
- `CLAUDE.md` has the full layout, cluster profiles and conventions. Tests: `uv run pytest -q`.
