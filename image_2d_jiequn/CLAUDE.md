# CLAUDE.md — `/scsi/imagae_2d_jiequn/` (minimal CryoET SCSI on MNIST)

Guidance for Claude Code when working in `src_clean/`. This folder is a **clean, minimal implementation** of the CryoET tilt-series SCSI algorithm acting on MNIST, distilled from the tangled production code in `../src/`. It does exactly
one thing — learn a clean-digit prior from limited-angle tilt-series projections
— and strips every optional axis (ODE/Euler only, no SDE/score net, no
canonicalization, no DDP/AMP/SLURM, single dataset, single corruption).

## What it is

The algorithm (Self-Consistent Stochastic Interpolant): a clean image is observed
only through `K` 1-D Radon projections at a **known relative tilt schedule** but an
**unknown global rotation** θ₀. We bootstrap a generative prior from those
projections by alternating (a) transporting an observation to a pseudo-clean
estimate through a frozen copy of the drift network and (b) re-corrupting that
estimate and regressing the interpolant velocity.

## File map


| File                     | Role                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `forward.py`             | CryoET tilt-series forward model: rotate by −θ → project (mean over width) → AWGN. `radon_tilt_series(...)` returns `fwd(x) -> (z_out, cond)`.                     |
| `backwards.py`           | FBP pseudo-inverse (ramp + Hamming) + `warmup_target` (FBP → per-image `[-1,1]`).                                                                                  |
| `model.py`               | `build_model(...)` — reuses `ConditionalDhariwalUNet` from `../src/networks.py` (adds `../src` to `sys.path`). Conditions on the raw `[K,32,32]` projection stack. |
| `data.py`                | `load_mnist_pm1` + `CorruptedTiltDataset` (tied per-image RNG so each image's θ₀ is fixed across epochs).                                                          |
| `scsi.py`                | `SCSInterpolant`: Euler `transport` + velocity `loss_fn`.                                                                                                          |
| `bootstrap.py`           | `Trainer`: warmup→bootstrap phase machine, frozen-transport refresh, EMA, checkpointing.                                                                           |
| `cli.py`                 | argparse → `Config` dataclass.                                                                                                                                     |
| `main.py`                | wires dataset → model → interpolant → trainer and runs.                                                                                                            |
| `sample.py`              | loads a checkpoint, reconstructs, saves a `[clean                                                                                                                  |
| `reconstruct_grid.ipynb` | interactive notebook version of `sample.py` (auto-loads `args.json` next to the checkpoint).                                                                       |


> **Not standalone:** `model.py` imports the network from `../src/`. Everything
> else is a self-contained port.

## Environment

Run with `uv`:

```bash
uv run main.py ...
```

Run commands **from inside `image_2d_jiequn/`** so the bare imports (`from forward import ...`) resolve.

## How to run

**Train:**

```bash
python main.py --train_steps 50000 --warmup_steps 6000 --results_folder ./results_clean/run1
```

Writes `args.json`, `model-best.pt`, `model-latest.pt`, `losses.npy` into the run
folder. `model-best.pt` tracks the lowest training loss; `model-latest.pt` is
written every 500 steps (the `Trainer`'s `save_every`, not CLI-exposed).

**Visualize a checkpoint:**

```bash
python sample.py --ckpt ./results_clean/run1/model-best.pt --n 8 --ode_steps 64 --model_channels 32
# or open reconstruct_grid.ipynb and Run All (it reads args.json automatically)
```

## CLI flags (`cli.py`)

`--K` `--tilt_span_deg` `--epsilon` (forward model) ·
`--train_steps` `--warmup_steps` `--transport_steps` `--batch_size` `--lr`
`--ode_steps` `--alpha` `--ema_decay` `--model_channels` (training) ·
`--data_root` `--results_folder` `--max_images` `--num_workers` `--seed` (io).

---

## Recommended settings (from Jiequn's tilt-series sweep, 2026-06-05)

Source of truth: `../private_docs/best_launch_recipes.md` (`radon_tilt_series`
section). Jiequn's **best is FID_f6 = 0.61** at ε=0 on `radon_tilt_series`,
K=16, ±60°. Below are his production values mapped onto `src_clean`'s flags. The
directly-transferable knobs are identical; **four ingredients that produced the
0.61 result are not in `src_clean`** — see "Known differences" so you calibrate
expectations.

**Recommended command (GPU):**

```bash
uv run main.py \
  --K 16 --tilt_span_deg 60 --epsilon 0.0 \
  --train_steps 50000 --warmup_steps 6000 --transport_steps 100 \
  --ode_steps 64 --alpha 1.0 --lr 3e-4 \
  --batch_size 256 --model_channels 32 --ema_decay 0.995 \
  --results_folder ./results_clean/tilt_K16_eps0
```

**Quick local/CPU smoke profile** (this machine is CPU-only; 256-batch DiT-scale
runs are impractical):

```bash
python main.py --max_images 512 --batch_size 32 \
  --train_steps 3000 --warmup_steps 800 --transport_steps 100 \
  --ode_steps 32 --lr 3e-4 --model_channels 32 --ema_decay 0.995
```

**Flag mapping (Jiequn production → `src_clean`):**


| Setting           | Jiequn (production)     | `src_clean` flag        | Notes                                              |
| ----------------- | ----------------------- | ----------------------- | -------------------------------------------------- |
| projections K     | `NUM_VIEWS=16`          | `--K 16`                | known-relative tilt stack                          |
| tilt span         | ±60°                    | `--tilt_span_deg 60`    | limited-angle / missing wedge                      |
| noise ε           | `LEVELS=0.0` (best)     | `--epsilon 0.0`         | sweep 0.1 / 0.2 below                              |
| train budget      | `STEPS=50000`           | `--train_steps 50000`   | curves still descending at 50k                     |
| warmup            | `WARMUP_STEPS=6000`     | `--warmup_steps 6000`   | FBP-seeded honest warmup                           |
| transport refresh | `TRANSPORT_STEPS=100`   | `--transport_steps 100` | refresh frozen teacher every 100                   |
| ODE steps         | `--ode_steps 64`        | `--ode_steps 64`        | Euler transport steps                              |
| alpha             | `--alpha 1.0`           | `--alpha 1.0`           | always use freshly re-corrupted data               |
| learning rate     | `3e-4`                  | `--lr 3e-4`             | Adam                                               |
| batch             | `256`                   | `--batch_size 256`      | drop to 32–64 if memory-limited                    |
| base channels     | `--channels 32`         | `--model_channels 32`   |                                                    |
| EMA decay         | `0.995` (EMBED default) | `--ema_decay 0.995`     | **src_clean default is 0.999 — override to 0.995** |


**Noise sweep (Jiequn, canon mode, 50k steps) — what to expect as ε rises:**


| ε   | Best FID_f6 (canon) | Best FID_f6 (default) |
| --- | ------------------- | --------------------- |
| 0.0 | **0.61**            | 5.91                  |
| 0.1 | **2.13**            | 7.44                  |
| 0.2 | **7.88**            | 14.13                 |


**Insights worth carrying into any run:**

- **Canonicalization is the single biggest lever** — canon beats default by ≥5
FID points at every noise level (projecting out the SO(2) rotation orbit).
- **FID degrades non-linearly in ε** (0.0→0.1 adds +1.5; 0.1→0.2 adds +5.7) — a
regime change near ε ≈ 0.15 as noise approaches the missing-wedge streak
amplitude.
- **EMA teacher ≫ raw teacher** (firstrun: canon-EMA 0.81 vs canon-raw 2.66, ~3×).
- **Longer training keeps helping** — canon curves still descending at 50k; 75–100k
likely pushes ε=0 below 0.5.

---

## Known differences vs the production `src/` recipe

`image_2d_jiequn` faithfully implements the **core** SCSI loop, but deliberately omits
four ingredients that Jiequn's 0.61-FID run relied on. Expect `src_clean` to land
nearer the much weaker "default / raw-teacher" rows than the 0.61 best, for these
reasons (roughly in order of FID impact):

1. **No canonicalization.** `src_clean` is "default" mode only — there is no
  `--canonicalize`. This is the most load-bearing missing piece (≥5 FID points).
   Production canon lives in `../src/canonicalize.py`.
2. **Raw transport teacher, not EMA.** `bootstrap.py` refreshes the frozen
  transport model via `copy.deepcopy(self.model)` (raw/live), whereas Jiequn
   used `transport_source=ema` (mean-teacher), worth ~3×. `src_clean` already
   maintains an EMA (`self.ema`), so this is a ~2-line change — refresh from
   `self.ema.ema_model` instead of the live model.
3. **UNet backbone, not DiT.** `src_clean` reuses `ConditionalDhariwalUNet`;
  production best used `--network dit`. (`../src/` supports both.)
4. **Plain FBP warmup, not the I+K recipe.** `backwards.warmup_target` is
  ramp+Hamming FBP + min-max. Production adds sinogram-angle smoothing +
   aggressive TV + inscribed-disk mask + L0 sparsity
   (`../src/forward_maps.py::_radon_tilt_warmup_target`).

Also: `**src_clean` does not compute FID.** Jiequn's numbers come from an
in-training LeNet-MNIST FID callback (`mnist_fid.jsonl`); `src_clean` only reports
MSE / MSE-up-to-rotation via `sample.py`. The FID values above are for calibrating
the recipe and direction, not directly measurable here without porting the callback.

> If asked to "match Jiequn's result," the highest-leverage additions are (1)
> canonicalization and (2) the EMA transport teacher — port those before DiT or
> the warmup polish.

## Conventions

- Reconstructions are correct only **up to a global rotation** (θ₀ is unobservable);
`sample.py` / the notebook report MSE-up-to-rotation for the honest comparison.
- `bg=-1.0` in `forward.rotate_image` (the pm1 shift-trick) and tied per-index RNG
in `data.CorruptedTiltDataset` are load-bearing — do not "simplify" them away.
- Forward `−θ` / FBP `+θ` sign convention must stay consistent (the
forward→FBP roundtrip correlation check is how you catch a flip).

