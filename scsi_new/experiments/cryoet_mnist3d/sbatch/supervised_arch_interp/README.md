# cryoet_mnist3d — supervised baseline: backbone × interpolant 2×2

4 SBATCH jobs running `experiments/cryoet_mnist3d/main_supervised.py` (the paired-data
upper bound the SCSI EM loop in `main.py` is chasing). Every job reconstructs **only
EMNIST digit 3** (`--digit_classes 3`, `--n_images_per_class 23000` — nearly all of
EMNIST's 24k threes) from the frozen `{(x, F(x))}` set, under the 3D→2D CryoET tilt-series
channel. No EM loop, no warm start.

## The 2×2

| file | `--arch` | `--interpolant_style` | wandb run |
|---|---|---|---|
| `submit_sup3d_dit_gvp.SBATCH`     | `dit`  | `gvp`    | `dit_gvp`     |
| `submit_sup3d_dit_linear.SBATCH`  | `dit`  | `linear` | `dit_linear`  |
| `submit_sup3d_unet_gvp.SBATCH`    | `unet` | `gvp`    | `unet_gvp`    |
| `submit_sup3d_unet_linear.SBATCH` | `unet` | `linear` | `unet_linear` |

All four log to wandb project **`scsi-cryoet-mnist3d-supervised-arch-interp`** so the grid
is filterable in one place. Checkpoints go to
`checkpoints/cryoet_mnist3d_supervised/<run>/step_<n>.pt` (gitignored as `checkpoints*`).

Fixed in every cell:

| knob | value | why |
|---|---|---|
| `--n_steps_train` | 300000 | as requested |
| `--log_every` | 20000 | viz-panel dump every 20k steps — **16 dumps total**: one at step 0, then steps 20k…300k |
| `--checkpoint_steps` | 20000, 40000, …, 300000 | `torch.save` model+EMA+optim+sched at every 20k |
| `--viz_n_steps_sampling` | 16 32 64 128 256 512 1024 | ODE-step sweep for the recon/trajectory panels; one panel set per value under `viz/{fixed,random}/ode<N>/`. (You wrote `516`; that's read as **512** — the powers-of-two run.) |
| `--viz_ema` | on | panels rendered from the EMA weights — the representative artifact for an "upper bound" run |
| `--batch_size` | 16 | matches the batch profiled in `../warmup_mstep_grid/` |
| `--digit_classes` | 3 | single class |
| `--n_images_per_class` | 23000 | ~3 GB volume pool + ~1.5 GB paired tilt-series tensor; fits `--mem=48G` with headroom (measured peak RSS ~9 GB in the sibling grid) |

## The two backbones

- **`--arch unet` → `ConditionalVelocityCryoET3D`** (`model.py`, unchanged): the diffusers
  `UNet3DConditionModel` video-UNet — `main.py`'s default. FACTORISED (2+1)D: per-slice 2D
  convs sharing weights across depth, cross-depth mixing only in the temporal blocks. ~37M
  params. The `T` tilt projections enter as `T` extra input channels (`stack_tilt_series`).
- **`--arch dit` → `ConditionalDiTCryoET3D`** (`model.py`, **new for this grid**): a
  genuinely volumetric DiT. 3D-patchifies the `(1 + num_tilts, 32, 32, 32)` input into
  `(32/4)**3 = 512` tokens and runs **full 3D self-attention** — every token attends to
  every other, so depth is on equal footing with the in-plane axes. adaLN-Zero timestep
  conditioning (each block starts at the identity → an untrained net predicts a ~0
  velocity). ~33M params at the `--dit_hidden 384 --dit_depth 12 --dit_heads 6
  --patch_size 4` default. Same tilt conditioning as the UNet.
  - *Not* `diffusers.Transformer3DModel` — that is also the (2+1)D ModelScope block, the
    same factorisation as the UNet (the abandoned `simple_3d/model.py::ConditionalDiT3D`
    tried it). Nothing in `diffusers` 0.37 ships a true 3D-patch DiT, so this one is a
    self-contained ~230-line implementation in `model.py`.
  - **This backbone has not been trained end-to-end** — only shape / gradient-plumbing /
    adaLN-Zero-at-init unit checks and a 4-step CPU smoke run. Watch the first `dit_*`
    checkpoint's `train/loss` before trusting the full run.

### Timestep quantisation — deliberate, and relevant to the ODE-step axis

Both backbones quantise `t` to **1000 integer bins** (`(t * INTEGRATION_SCALE).long()`,
`INTEGRATION_SCALE = 999`). The DiT could embed float `t` and get finer time resolution,
but then `ode512` / `ode1024` panels would be measuring *different things* on the two
archs — precisely where you're asking "does more integration help, and does it help the
archs differently?". Matching keeps the ODE-step axis clean. Consequence: past ~256 ODE
steps the *time-dependence* of the field is bin-limited on both archs; the extra steps
still refine the spatial Euler integration, just not the temporal resolution.

## Submitting

```bash
cd experiments/cryoet_mnist3d/sbatch/supervised_arch_interp
for f in submit_sup3d_*.SBATCH; do sbatch "$f"; done
```

`experiments/cryoet_mnist3d/data/EMNIST` already exists, so there is no first-run download
race.

## Caveats (read before trusting a 300k-step run)

- **`--time=48:00:00` is a flat budget, not a measurement.** No CUDA was available to
  profile. 300k optimiser steps + the viz sweep is roughly 2× the biggest cell of
  `../warmup_mstep_grid/` (which used a flat 24h for ~160k steps + E-step passes). If your
  partition rejects 48h, drop to `24:00:00` — but then 300k steps will **not** finish.
- **No `--resume_from`.** The 20k checkpoints save optimiser + scheduler + EMA, so they are
  resumable *in principle*, but `main_supervised.py` has no flag to continue from one. A
  walltime kill leaves an inspectable `step_<n>.pt` and nothing else. After the first
  checkpoint, read `train/` step-rate off wandb and, if needed, resubmit with a real
  `--time` or a smaller `--n_steps_train`.
- **Viz cost is real.** Each of the 16 panel dumps runs `2 panels × Σ_N (6·N + 3·N)` model
  calls over the ODE sweep ≈ 8.1k small-batch (3–6) forwards; the `512` and `1024` renders
  dominate. Rough order: ~130k extra forwards against 300k batch-16 fwd+bwd — order 10%
  wall-clock overhead for the UNet, less for the DiT. Lever if it hurts:
  `--log_every 40000`, or trim `--viz_n_steps_sampling`.
- **`--gres=gpu:1` is generic — the two `unet` cells can OOM on a small card.** Batch 16
  through the video-UNet peaked ~37 GB on a 48 GB L40 in `../warmup_mstep_grid/`. A job
  landing on a <40 GB card (V100 32 GB) will OOM. Fix: a typed `--gres` (check
  `sinfo -o "%n %G"`) or `--batch_size 8` — change it in **both** `unet` cells so the pair
  stays comparable. The `dit` cells are far under this (512 tokens × 384 hidden).
- **Cross-arch comparison is inherently apples-to-oranges.** Same param budget (~33M vs
  ~37M) and same `--batch_size`, but the DiT under-utilises the GPU at batch 16 and its
  inductive biases are entirely different. The clean comparisons are *within* a backbone
  (GVP vs Linear) and *within* an interpolant (which backbone).
- **`--eta_min 1e-5`** cosine floor is the `main_supervised.py` default; the schedule spans
  all 300k steps and is unaffected by where the log/checkpoint pauses fall.

## Smaller / faster variants

```bash
# quick local check (both archs), ~minutes on CPU:
cd experiments/cryoet_mnist3d
WANDB_MODE=disabled uv run python main_supervised.py --arch dit --device cpu \
  --digit_classes 3 --n_images_per_class 4 --batch_size 2 --n_steps_train 4 \
  --log_every 2 --checkpoint_steps 2 4 --viz_n_steps_sampling 8 --checkpoint_dir /tmp/ckpt_smoke
```
