# fresh-student lr x warmup x mstep grid — digit 3 only

27 SBATCH jobs (Rusty), one per cell of a 3x3x3 full factorial, all with **`--student_init fresh`**:
after every E-step the teacher is discarded and a newly initialized student is trained from
scratch on the E-step samples. Reconstructs **only MNIST digit 3** (`--digit_classes 3`,
`--n_images_per_class 6000`) under the cryoET-style tilt-series channel in
`experiments/cryoet_mnist/main.py`. Fixed everywhere: `--num_scsi_steps 40`, `--warmup_lr 3e-4`,
`--eta_min 1e-5` (default).

Swept:

| axis | flag | values |
|---|---|---|
| M-step learning rate (peak of each student's cosine) | `--mstep_lr` | 1e-4 / 3e-4 / 1e-3 |
| warmup training length | `--warmup_n_steps_train` | 10,000 / 20,000 / 40,000 |
| M-step training length (per fresh student) | `--mstep_n_steps_train` | 5,000 / 10,000 / 20,000 |

File naming: `submit_fresh_lr{lr}_w{warmup}_m{mstep}.SBATCH`, e.g.
`submit_fresh_lr3e-4_w20k_m10k.SBATCH` = `mstep_lr=3e-4, warmup_n_steps_train=20000,
mstep_n_steps_train=10000`. The wandb run name matches (`lr3e-4_w20k_m10k`), and the whole grid
logs to the `scsi-cryoet-mnist-three-fresh-lr-warmup-mstep` wandb project.

## How the LR schedule works in fresh mode

Each model trains on its own schedule, so the three axes don't leak into each other:

- **Warmup:** its own AdamW at `--warmup_lr 3e-4`, cosine down to `--eta_min` over
  `--warmup_n_steps_train` steps. It doesn't depend on `--mstep_lr` or `--mstep_n_steps_train`.
  All 9 cells with the same warmup length therefore train the same warmup model (same seed), so
  their EM-step-0 panels should match.
- **Each M-step:** a new model, a new AdamW at `--mstep_lr`, and a new EMA, with a cosine down to
  `--eta_min` over `--mstep_n_steps_train` steps. `train/lr` in wandb looks like a sawtooth.

3e-4 for the warmup is the LR every earlier grid's warmup effectively trained at (in `teacher`
mode the warmup shares the M-step optimizer at `--mstep_lr`, default 3e-4).

## Submitting

On a Rusty login node:

```bash
./submit_all.sh
```

This downloads MNIST once before queuing anything, because jobs that start together would
otherwise race on `data.py`'s first-use download. Submitting individual files with plain `sbatch`
works too once `cryoet_mnist/data/MNIST` exists. Logs go to
`cryoet_mnist/slurm-<jobname>-<jobid>.{out,err}`.

To check that Slurm accepts the resource request without queuing anything:
`sbatch --test-only submit_fresh_lr3e-4_w20k_m10k.SBATCH`.

## Resources

- `--partition=gpu --gres=gpu:1 --constraint=rocky9&(h100|a100-80gb)`. Without the constraint, a
  plain `gpu:1` can land on an A100 MIG slice (`a100_1g.20gb` / `a100_2g.20gb`). `rocky9` matches
  the OS the shared `.venv` was built on.
- These are 27 separate jobs, not one disBatch allocation. Each cell runs for days and the cells
  differ ~4x in length, so one big allocation would leave GPUs idle once the short cells finish.

## Known caveats

- **Walltime is an estimate, not a measurement.** The M-step dominates each EM step, so `--time`
  scales with M-step length: m5k = 1 day, m10k = 2 days, m20k = 4 days (about 2x headroom over
  a "~1 day" run). Jobs end as soon as they finish. If the first few jobs are much faster or
  slower, adjust the rest.
- **No checkpointing.** `main.py` never calls `torch.save`, so a walltime kill loses the model.
  wandb still has the reconstruction and trajectory panels for every EM step reached
  (`--viz_every` default 1).
- **Short M-steps may underfit.** Each fresh student only sees `--mstep_n_steps_train` steps from
  scratch, compared with 10k–40k for the warmup. The m5k cells (especially at lr 1e-4) are where
  that would show up. Measuring it is part of what this grid is for.
- **Not directly comparable to the earlier `teacher`-mode grids at EM step 0.** In `teacher` mode
  the warmup's LR schedule depends on the M-step settings (one cosine over the whole run). Here it
  doesn't.
