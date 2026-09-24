# cryoet_mnist3d — launch recipes

3D->2D counterpart of `../cryoet_mnist`. Extruded EMNIST-digit voxel volumes under a Haar-uniform
SO(3) "mount" + fixed-axis tilt series -> 2D parallel-beam projection -> AWGN, recovered with
an EM loop over a `diffusers` `UNet3DConditionModel` velocity field.

## Starting point

```bash
cd experiments/cryoet_mnist3d
uv run python main.py \
    --n_images_per_class 500 \
    --num_tilts 16 \
    --warmup_n_steps_train 10000 \
    --mstep_n_steps_train 5000 \
    --num_scsi_steps 12
```

`num_scsi_steps 12` is deliberately below the 2D recipe's 40: at `vol_size 32` an EM step is
~30x more compute, `main.py` has **no checkpointing**, and a SLURM walltime kill loses the
weights (only the per-EM-step wandb panels survive). Raise it once you know your per-step
wall time. Start small (`--n_images_per_class 50 --num_scsi_steps 3`) to time a step first.

## Differences from the 2D experiment

- **Warm start is a classical reconstruction with known pose.** `data.build_warmup` ->
  `(x_hat, y)` pairs where `x_hat` is `pseudoinverse.py`'s filtered backprojection (3D
  weighted backprojection) of a tilt series -- identity/label-blind, but pose-supervised: 3D
  backprojection can't absorb a wrong Haar SO(3) mount the way the 2D sibling absorbs a wrong
  angle offset, so `build_warmup` rebuilds its own observations and keeps the rotations it
  draws. Every panel here is therefore an upper bound conditional on known pose (same caveat
  as `classical_recon_3d.py`). Toggle with `--no_filtered` / `--filter_type {hann,ramp}`.
- **Tilt conditioning is by channel, not by layout.** `UNet3DConditionModel` is the ModelScope
  video UNet — per-slice 2D convs sharing weights across depth, cross-depth mixing only in the
  temporal blocks. The `T` tilt projections go in as `T` extra input channels
  (`in_channels = 1 + num_tilts`) so every depth slice sees the whole series. See
  `model.stack_tilt_series`.
- **Noise regime isn't comparable to the 2D sibling.** `noise_std=3.0` is inherited unchanged,
  but a 3D observation is `T*H*W` numbers vs `T*W` in 2D — ~32x more measurements at the same
  per-pixel noise. Per-tilt SNR is still low (a single tilt reads as a faint blob — check with
  the gallery below), but don't read a 3D-vs-2D difficulty comparison off the shared `noise_std`.

## Memory knobs

A volume is ~`V` larger than a 2D image, and the video-UNet reshapes `(B,C,D,H,W)` ->
`(B*D,C,H,W)` for its spatial convs, so the shared 2D batch defaults OOM immediately. `args.py`
already lowers them to `mstep_batch_size=8`, `estep_batch_size=16`, `estep_num_samples=2000`.
`mstep_batch_size=8` is an **unmeasured starting point** (developed on darwin/CPU, no CUDA to
profile). If you OOM:

- `--mstep_batch_size 4` (or `2`)
- `--block_out_channels 48 96 192 192` — narrower net (~21M params vs ~37M; verified to build)
- `--vol_size 24 --num_tilts 12` — smaller grid (24 -> 12 -> 6 -> 3 through the 4 levels; verified)

## Speed knobs

- `--estep_n_steps_sampling 32` — the E-step runs `ceil(estep_num_samples / estep_batch_size)`
  ODE integrations of `n_steps_sampling` forwards each; halving it ~halves E-step wall time.
  (`euler_integration` is `@torch.no_grad()` and updates in place, so this is time, not memory.)
- `--estep_num_samples 1000` — fewer posterior samples per M-step dataset.

## LR schedule, EMA, checkpoints, eval

The default LR schedule is one cosine over warmup + all EM steps, so `--num_scsi_steps` also
sets the LR every EM step runs at. A 50-step run spends its first 20 EM steps above 1.7e-4,
where the 12-step run has annealed out by EM 12 (`sbatch/lr_schedule_ablation/README.md`).
These flags are all opt-in, and the defaults reproduce the original runs:

- `--lr_schedule {cosine,constant,cosine_per_mstep}` and `--lr_horizon_scsi_steps K`: the
  cosine reaches `--eta_min` at EM step K and then **holds** there, unlike
  `CosineAnnealingLR`, which climbs back up past `T_max`.
- `--sample_with_ema`: the E-step and panels use the EMA weights, which were otherwise tracked
  but never used.
- `--save_warmup_ckpt PATH` / `--load_warmup_ckpt PATH`: train the warm start once and share it
  across runs. The load refuses a checkpoint with different dataset / model args.
  `--em_seed` reseeds after it.
- `--ckpt_dir DIR`: rolling `latest.pt` every EM step. Add `--resume` to continue from it
  (same wandb run) when it exists: chain the same SBATCH file with `--dependency=afterany` to
  run past the 48h walltime.
- `eval/{raw,ema}/corr_own|corr_other|corr_gap`: rotation-searched correlation vs GT, logged
  every EM step (`eval_metrics.py`). `corr_gap` is 0 for any y-independent (collapsed) output.
  Multi-class runs also log `corr_gap_xcls` (vs a different digit) and `by_class/<d>/...`.
  Keep `--eval_n` at least 2x the number of classes. `--eval_n 0` turns it off.

All ten digits: `sbatch/tenclass_sweep/README.md`.

## Digit width

`--digit_scale S` (default `1.0`) isotropically scales the extruded digit inside the cube:
in-plane footprint `round(vol_size * 0.65 * S)`, depth band `round(vol_size * 0.25 * S)`.
`>1` widens/thickens, `<1` shrinks. `--inplane_size` / `--depth_extent` still override either
axis with an absolute voxel count. Large `S` makes the digit clip the cube corners under the
SO(3) mount (`data._digit_geometry` warns, then `ValueError` past `vol_size`) — check the
mass-invariance line the geometry viewer prints before committing to a value. wandb point-cloud
panels size their voxel budget from the GT ink count, so they follow `--digit_scale` with no
extra flag.

## Quick data / geometry check

```bash
uv run python data.py --n_images_per_class 2 --num_tilts 16   # interactive 3D window: GT vs
                                                             #  build_warmup recon (rotatable,
                                                             #  linked isosurfaces) + SO(3) mass check
                                                             #  --level tunes the isosurface; --save
                                                             #  PATH / --no_show for headless
uv run python data.py --n_images_per_class 2 --digit_scale 1.4   # same viewer at a wider digit;
                                                             #  read the printed mass-check % first
uv run python pseudoinverse.py --n_images_per_class 2         # known-pose WBP smoke test + Pearson r
```
