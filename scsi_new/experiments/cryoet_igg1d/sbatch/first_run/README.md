# First IgG-1D runs (Flatiron Rusty)

wandb projects: `scsi-cryoet-igg1d` for runs, `scsi-cryoet-igg1d-debug` for `debug.sh`. Logs,
checkpoints and wandb files go under `/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/`
(`logs/`, `checkpoints/<name>/latest.pt`, `wandb/`).

## Order

1. **Debug on an interactive GPU node** (~15 min):

   ```bash
   salloc -p gpu --gres=gpu:1 -C "a100-80gb|h100|h200" --cpus-per-task=16 --mem=64G --time 1:00:00
   bash /mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_igg1d/sbatch/first_run/debug.sh
   ```

   The script checks the channel conventions, then runs 2 EM steps at production batch sizes
   (M-step 12, E-step 32, 64 Euler steps) on 2000 observations, then resumes to EM 3. Read off:
   - `[em k/…] estep …s mstep …s viz+eval+ckpt …s … peak GPU mem …`: E-step seconds / 64 samples,
     M-step seconds / 50 steps;
   - `eval calibration (GT targets)`: `conf_acc` should be near 1;
   - the second invocation should print `resuming from … at EM step 2`.
2. **`bash submit_all.sh`**: `em.SBATCH` plus one `--dependency=afterany` continuation. Pass
   `N_JOBS=3` for a third.

## em.SBATCH

This run starts EM from a randomly initialised network (`--warmup_n_steps_train 0`) with the
GT-derived `--vol_gain` from `python data.py`. It is the no-warm-start baseline for
`../sched_warmup_mstep`, which uses the `pseudoinverse.py` warm start. The EM-1 E-step samples an untrained network, so its x̂ is close to Gaussian noise.

Settings:
- 12 EM steps of 2000 posterior samples and 5000 M-step steps;
- one global cosine to `--eta_min` over the 12 steps;
- raw weights for sampling;
- all ~98k training images observed;
- `--eval_n 16`, the images cryofm's sampling eval uses.

## Wall time (estimated, not measured)

Extrapolated from the supervised cryofm debug job (0.81 it/s at batch 12, bf16, A100-80GB, which
included its EMA update):

| part | per EM step |
|---|---|
| M-step, 5000 steps at batch 12 | ~1.7 h |
| E-step, 2000 samples x 64 Euler steps | ~1 h if a forward is ~1/3 of a training step |
| eval + panels (16 images x 64 steps x raw+EMA, 2 panels of 8) | a few min |

That is about 3 h per EM step, so 12 EM steps take ~36 h, plus ~5 min at startup to read the images.
Two chained 2-day jobs cover it. Replace these numbers with `debug.sh`'s.

## Caveats

- **Jobs run the live code** in `~/scsi/scsi_new/experiments/cryoet_igg1d` (there is no rsynced
  copy as on NYU Torch). Editing it while jobs are queued changes what they run.
- **`--resume` redoes the interrupted EM step.** wandb drops re-logged `train/` steps until the
  step count passes where the killed job stopped (as in cryoet_mnist3d).
- **Keep `--vol_gain` fixed** across continuations. Checkpoints refuse a different one.
- `--mem=64G`: the observations take ~1.6 GB of host RAM, and the eval pool loads the 100
  released 128³ volumes (~0.8 GB). The E-step keeps its samples on the GPU (~2 GB at 2000 x 64³).
