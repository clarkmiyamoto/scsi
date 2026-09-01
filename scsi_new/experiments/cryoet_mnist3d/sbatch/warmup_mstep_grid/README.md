# 3D warmup x mstep grid — digit 3 only

9 SBATCH jobs, one per cell of a 3x3 grid, all reconstructing **only EMNIST digit 3**
(`--digit_classes 3`, `--n_images_per_class 23000` — nearly all of EMNIST's 24k threes) under
the 3D->2D CryoET-style tilt-series channel in `experiments/cryoet_mnist3d/main.py`. The 3D
counterpart of `../../../cryoet_mnist/sbatch/warmup_mstep_grid/` (the 2D digit-3 grid).

Fixed everywhere except the two swept axes: `--num_scsi_steps 12`, `--warmup_batch_size 16`,
`--mstep_batch_size 16`, `--estep_batch_size 32`.

Swept:

| axis | flag | values | 2D grid had |
|---|---|---|---|
| warmup training length | `--warmup_n_steps_train` | 10,000 / 20,000 / 40,000 | 5k / 10k / 20k / 40k |
| SCSI M-step training length (per EM iteration) | `--mstep_n_steps_train` | 2,000 / 5,000 / 10,000 | 2k / 5k / 10k / 15k / 20k |

The 3x3 brackets the 2D optimum (`w40k / m5k`, from `cryoet_mnist/RECIPIES.md`) on all sides.
5k warmup is dropped (too short to stabilise batch-16 training at this model size); 15k/20k
mstep are dropped (wall-clock prohibitive — see caveats).

## Differences from a naive copy of the 2D grid

- **`--num_scsi_steps 12`, not 40.** At `vol_size 32` an EM step is ~30x the compute of the 2D
  sibling's (`RECIPIES.md`). The M-step axis also stops at 10k instead of the 2D grid's 20k.
- **`--warmup_batch_size 16` is passed explicitly.** `cryoet_mnist3d/args.py` lowers the shared
  `mstep_batch_size` / `estep_batch_size` / `estep_num_samples` defaults for 3D but leaves
  `warmup_batch_size` at the shared default of 258 — which OOMs immediately in warmup on the
  ~37M-param UNet3D (258 volumes -> 258*32 = 8256 rows through the spatial convs). Every cell
  sets it to 16, matching `mstep_batch_size` so the two swept axes are comparable in
  examples-seen per step.
- **Batch sizes come from a real profile:** `--warmup_batch_size 16 --mstep_batch_size 16
  --estep_batch_size 32` measured at ~77% memory on a 48 GB L40. Batch, not pool size, sets
  GPU memory, so `--n_images_per_class` does not change this.
- **`--n_images_per_class 23000`** — nearly all of EMNIST's 24k digit-3 images (MNIST had only
  ~6.1k threes). Single-class, so the pool is ~3 GB; measured peak host RSS is ~9 GB through
  `build_observations` / `build_warmup`, comfortably inside the unchanged `--mem=48G`.
  All-10-digit runs are a different story (~30 GB pool, ~78 GB peak); this grid is not that.
- `--mstep_lr` is left at the `3e-4` default — batch is only 2x the 3D arg default, so the
  sqrt-rule bump (~4e-4) is within noise.

File naming: `submit_three3d_w{warmup}_m{mstep}.SBATCH`, e.g. `submit_three3d_w20k_m10k.SBATCH` =
`warmup_n_steps_train=20000, mstep_n_steps_train=10000`. Job name and wandb run name match
(`three3d_w20k_m10k` / `w20k_m10k`); all logged to the
`scsi-cryoet-mnist3d-three-warmup-mstep-grid` wandb project so the whole grid is filterable in
one place.

## Submitting

```bash
for f in submit_three3d_*.SBATCH; do sbatch "$f"; done
```

`./data/EMNIST` already exists in `experiments/cryoet_mnist3d/`, so the first-use download race
that the 2D grid warns about is a non-issue here — but if you run this on a fresh checkout where
`./data/EMNIST` is absent, submit one job first (or run `uv run python data.py
--n_images_per_class 2 --digit_classes 3` once) before firing the rest, so 9 jobs don't race on
the ~560 MB EMNIST zip download.

## Known caveats (apply to every cell)

- **`--time=24:00:00` is a flat budget, not a measured one** (set at the user's request). A 3D
  EM step is ~30x the 2D sibling's, partly offset by `num_scsi_steps` 12 vs 40 and the M-step
  axis stopping at 10k. Per-cell SGD-step totals (`warmup + 12 * mstep`) run from 34k
  (`w10k / m2k`) to 160k (`w40k / m10k`), plus ~48k E-step forward passes total
  (`ceil(2000 / 32) * 64` per EM step, `estep_num_samples` default 2000). Read the `train/` step
  rate off a run and adjust `--time` if the big cells look like they will not finish 12 EM steps
  in a day.
- **No checkpointing.** `main.py` never calls `torch.save`. A walltime kill loses the model
  entirely — nothing to resume from. What survives: wandb logs the reconstruction + trajectory
  panels every EM step (`--viz_every` default 1), so a killed run still leaves a partial
  progression up to whatever EM step it reached.
- **`--gres=gpu:1` is generic.** Batch 16 peaks near 37 GB, so a job scheduled onto a <40 GB
  card (e.g. a V100, 32 GB) will OOM. If that happens, either add a `--constraint` / a typed
  `--gres` (e.g. `--gres=gpu:l40:1` — check `sinfo -o "%n %G"`), or drop
  `--warmup_batch_size` / `--mstep_batch_size` to 12 in **all 9** cells.
- **The two swept axes are not cleanly isolated — worse than in 2D.** `main.py` builds one
  `CosineAnnealingLR` with `T_max = warmup_n_steps_train + 12 * mstep_n_steps_train` spanning the
  whole run, so changing either axis changes the LR schedule the *other* phase trains under.
  With `num_scsi_steps` only 12, warmup's share of the cosine is large: 7.7% at `w10k / m10k`
  but **62.5% at `w40k / m2k`** — that cell's entire 12-step SCSI loop runs in the last third of
  the cosine, mostly near `eta_min` (`--eta_min` default 1e-5). Treat `w40k / m2k` as the
  degenerate corner, not a clean data point. (The 2D grid's worst corner was 33%.)
- **`--warmup_lr` has no effect.** The optimizer is constructed once with `mstep.lr`
  (`main.py:48-50`) and reused for both warmup and the SCSI loop — warmup always trains at the
  M-step learning rate regardless of any `--warmup_lr` flag.
- **The warm start is pose-blind.** `data.build_warmup` resamples its own SO(3) tilt-series
  rotations (they are not observed), and 3D backprojection cannot absorb a wrong Haar mount into
  a global reorientation the way the 2D sibling absorbs a wrong angle offset. The em-step-0
  panels are a rough symmetry-breaking seed, not an upper bound (see `RECIPIES.md` /
  `pseudoinverse.py`).
- **Further OOM fallbacks** (change in **all 9** cells so the grid stays comparable):
  `--mstep_batch_size 8` / `--estep_batch_size 16`; `--block_out_channels 48 96 192 192`
  (~21M params); `--vol_size 24 --num_tilts 12`.
