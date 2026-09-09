# cryoet_mnist3d — supervised baseline, LIFTED target: backbone × interpolant 2×2

Same 2×2 as [`../supervised_arch_interp/`](../supervised_arch_interp/README.md) — 4 SBATCH jobs
running `experiments/cryoet_mnist3d/main_supervised.py` on **EMNIST digit 3 only**
(`--digit_classes 3 --n_images_per_class 23000`), backbone `{dit, unet}` × interpolant
`{gvp, linear}`, from the frozen `{(target, F(x))}` set under the 3D→2D CryoET tilt-series
channel. The only difference from the canonical grid: **`--lift`** — the target is `R·X`, not
canonical `X`.

## What `--lift` changes

The canonical grid pairs `(X, F(X))` with `X` the upright extruded digit. But `y = F(X)` is
**pose-blind** — the SO(3) mount is drawn inside the channel and discarded — so `y` pins the
clean volume only up to a global rotation (`Q·X` gives an equally valid `y` for any `Q`).
Training toward canonical `X` asks the net for an orientation `y` does not determine.

`--lift` (already the `main_supervised.py` default; passed explicitly in every cell here) pairs
`(R·X, F(X))` with **R a fresh independent Haar-uniform SO(3) rotation per volume**, drawn with
nothing correlated to the channel — `corruption.build_pair_sample`. `F(X)` is unchanged. This
symmetrises the training target over SO(3), so the learned `b_t(·|y)` is rotation-invariant:
`y` fixes the shape, orientation is free. It is the pose-agnostic problem the unsupervised SCSI
loop in `main.py` actually solves, so **this grid, not the canonical one, is its true upper
bound**.

Consequence for the wandb recon panels: `x_hat` comes out in a random orientation and will not
line up with the canonical `x_gt` drawn beside it. The `F(x_hat)`-vs-`y` row and the
digit-plausibility read stay valid; the GT-overlay row now answers "right shape?", not
"aligned?". (Viz code is unchanged.)

## The 2×2

| file | `--arch` | `--interpolant_style` | wandb run |
|---|---|---|---|
| `submit_sup3d_dit_gvp_lift.SBATCH`     | `dit`  | `gvp`    | `dit_gvp_lift`     |
| `submit_sup3d_dit_linear_lift.SBATCH`  | `dit`  | `linear` | `dit_linear_lift`  |
| `submit_sup3d_unet_gvp_lift.SBATCH`    | `unet` | `gvp`    | `unet_gvp_lift`    |
| `submit_sup3d_unet_linear_lift.SBATCH` | `unet` | `linear` | `unet_linear_lift` |

- wandb project **`scsi-cryoet-mnist3d-supervised-arch-interp-lift`** — separate from the
  canonical grid's `scsi-cryoet-mnist3d-supervised-arch-interp`, so the two are filterable
  side by side.
- Checkpoints: `checkpoints/cryoet_mnist3d_supervised/<run>_lift/step_<n>.pt` (gitignored as
  `checkpoints*`). No collision with the canonical grid's `<run>/`.
- Every other knob — 300k optimiser steps, `--batch_size 16`, `--log_every 20000`,
  `--checkpoint_steps 20000…300000`, `--viz_ema`, ODE sweep
  `--viz_n_steps_sampling 16 32 64 128 256 512 1024` — is identical to the canonical grid.
  `--lift` adds one `grid_sample` per volume when the frozen pool is built: a one-time cost,
  nothing per optimiser step.

## Caveats

All caveats in [`../supervised_arch_interp/README.md`](../supervised_arch_interp/README.md)
carry over unchanged:

- **`--time=48:00:00` is a flat budget, not a measurement.** No `--resume_from` in
  `main_supervised.py`; a walltime kill leaves an inspectable `step_<n>.pt` and nothing else.
  Read `train/` step-rate off wandb after the first checkpoint and resubmit if 300k won't finish.
- **`--gres=gpu:1` is generic — the two `unet` cells can OOM on a <40 GB card** (batch 16
  through the video-UNet peaked ~37 GB on a 48 GB L40). Use a typed `--gres` or `--batch_size 8`
  in **both** `unet` cells.
- **Viz cost** is ~10 % wall-clock for the UNet, less for the DiT; lever is `--log_every 40000`
  or a shorter `--viz_n_steps_sampling`.
- The `dit` backbone has had only shape / gradient / adaLN-Zero-at-init checks and a short CPU
  smoke — watch the first `dit_*_lift` checkpoint's `train/loss`.
- Clean comparisons are *within* a backbone (GVP vs Linear) and *within* an interpolant (which
  backbone); cross-arch is apples-to-oranges (same ~33M/37M param budget, different biases).

## Submitting

```bash
cd experiments/cryoet_mnist3d/sbatch/supervised_arch_interp_lift
for f in submit_sup3d_*_lift.SBATCH; do sbatch "$f"; done
```

`experiments/cryoet_mnist3d/data/EMNIST` already exists — no first-run download race.

## Quick local check

```bash
cd experiments/cryoet_mnist3d
WANDB_MODE=disabled uv run python main_supervised.py --arch dit --device cpu --lift \
  --digit_classes 3 --n_images_per_class 4 --batch_size 2 --n_steps_train 4 \
  --log_every 2 --checkpoint_steps 2 4 --viz_n_steps_sampling 8 --checkpoint_dir /tmp/ckpt_smoke_lift
```
