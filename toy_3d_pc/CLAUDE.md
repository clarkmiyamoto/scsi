# toy_3d_pc — navigation guide for Claude

Self-contained package: **lifted SCSI (Self-Consistent Stochastic Interpolant) for CryoET in
the point-cloud representation.** A 3D object is a *set* of `N` points `X ∈ ℝ^{N×3}`. The
CryoET channel `F` renders a noisy tilt series of 2D projections under one unknown global
SO(3) pose; SCSI recovers a generative prior over clean clouds from only those projections.

Run everything via `uv run python -m toy_3d_pc scsi …` (CPU on darwin, CUDA on the cluster).

## What makes this package distinct

This is the **literal-pseudocode** clean-room build. There are three sibling packages with
near-identical code — `toy_3d_pointcloud` (base, single SO(3) projection),
`toy_3d_pointcloud_so3`, and `toy_3d_pointcloud_et` (CryoET, the mature reference). `**et` uses
a pool-based EM**; this package deliberately implements the literal algorithm instead:

1. **Fused on-the-fly transport** — each inner training step runs the EMA transport ODE
  `x̂ = Φ_EMA^(k-1)(z'|y)` to generate the clean target (no frozen M-step pool).
2. `**α_y` observation coupling** — `ŷ ← y` w.p. `α_y`, else `F(x̂)`.
3. **EMA over the outer EM loop** — `Θ_EMA^(k) ← γ·Θ_EMA^(k-1) + (1-γ)·Θ^(k)`, frozen during
  the inner loop.

Use `et` as a reference for alternative (pool-based) choices, but keep these three properties
when editing here.

## File map


| File                                   | Role                                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `corruption.py`                        | **Forward model `F`** (`forward_channel`) + **pseudo-inverse `F†`** (`pseudo_inverse`/`backproject_tomo`) + rotation helpers (`tilt_rotations`, `random_rotations`, `rotate_clouds`).                           |
| `si.py`                                | Stochastic interpolant: `linear`/`gvp` schedules, `interpolant(z,x,t,style)→(I_t, İ_t)`, `transport_sample` (Euler or Heun ODE, configurable interval `[eps_start, 1-eps_final]`).                              |
| `model.py`                             | `ConditionalPointCloudVelocity` (permutation-equivariant set-transformer + image cross-attn, `in_channels=K`), `ConditionalModelConfig`, `build_conditional_model`, EMA helpers `clone_ema`/`ema_update_outer`. |
| `warmstart.py`                         | **Algorithm 1** — `find_initialization`: train `b̂^(0)` on `(g·F†(y), y)`.                                                                                                                                      |
| `scsi.py`                              | **Algorithm 2** — `scsi_train` (the literal EM loop, the core deliverable) + `log_em_step`/`log_bootstrap` + checkpoint I/O (`save_checkpoint`, `save_train_state`/`load_train_state` for resume).               |
| `supervised.py`                        | `train_supervised` debug oracle: train directly on `(x, F(x))` with fresh GT (no EM).                                                                                                                           |
| `data.py`                              | Shapes as solids: SDF + bbox per shape (`torus`, `dumbbell`, `trefoil`, `l_shape`, `t_shape`), generic `_sample_solid` rejection sampler, `make_mixture_sampler`, `sample_perturbed_dataset` (template/subtomogram dataset), `mixture_volume_residual` diagnostic (reuses each shape's SDF). |
| `device.py`                            | CUDA>MPS>CPU autodetect, `autocast`, `needs_grad_scaler`, `configure_backends`.                                                                                                                                 |
| `tracking.py`                          | W&B `Tracker`, **enabled by default** (graceful no-op if wandb missing/unconfigured).                                                                                                                           |
| `plot.py`, `balls.py`                  | 3D scatter PNG; optional union-of-balls `.obj` mesh export for W&B.                                                                                                                                             |
| `cli.py`, `__main__.py`, `__init__.py` | argparse `scsi` subcommand; entry point; public exports.                                                                                                                                                        |


## Forward model `F` (the centerpiece, `corruption.py`)

`F(X) = { P · R_tilt(n·Δθ) · R(θ) · (G ∘ (X + W)) + Z }_{n=1..K}` → `(B, K, P, P)`:

- `G` = `_gaussian_splat` (separable, default) **or** `_ball_splat` (analytic projection of a
**solid/filled** ball = a filled disk, not a shell; non-separable, chunked over N). `radius`
is σ (gaussian) / ball radius. Projection `P` (drop-z) is baked into the 2D splat.
- `R(θ)` = one Haar-uniform global SO(3) pose per cloud (the unknown nuisance).
- `R_tilt` = `K` known single-axis tilts (`tilt_step°` apart, centered at 0, `tilt_axis` x/y).
- `W` = `coord_noise_std` on coords before rotation; `Z` = `noise_std` per-tilt image AWGN.
- `F†` (`pseudo_inverse`) = soft space-carving back-projection of the K tilts → occupancy grid
→ sampled point cloud, using **only** the known tilt geometry (residual global pose left for
EM). Used by the warm-start and to seed π(0).

## Algorithm (`scsi.py::scsi_train`)

```
y_obs = F(x_gt)                          # μ; GT used ONLY to make observations + eval
x_boot = F†(y_obs)                        # pseudo-inverse bootstrap (π(0))
model ← find_initialization(...)          # Algorithm 1 warm-start  →  Θ^(0)
model_ema ← clone_ema(model)              # Θ_EMA^(0) ← Θ^(0)
opt = AdamW(model.parameters())           # persistent across outer iters
for k in 1..em_steps:                     # EMA frozen during this inner loop
    for i in 1..training_steps:           # T_tr inner SGD steps
        y = minibatch(y_obs)
        z' ~ N(0,I);  x̂ = transport_sample(model_ema, z', y, sample_steps)   # Φ_EMA(z'|y)
        z  = z' w.p. α_z else N(0,I)                                          # noise coupling
        ŷ  = F(x̂);  ŷ = y w.p. α_y else ŷ                                     # obs coupling
        I_t, İ_t = interpolant(z, x̂, t);  loss = ‖model(I_t,t,ŷ) − İ_t‖²; opt.step()
    ema_update_outer(model_ema, model, γ)                                     # outer EMA
    log_em_step(...)                       # sample π(k) with model_ema; PNG panel + residual
```

## How to run

```bash
uv run python -m toy_3d_pc scsi --debug --no-wandb          # ~seconds smoke test
uv run python -m toy_3d_pc scsi --shape torus --alpha-z 0.5 --alpha-y 0.5
uv run python -m toy_3d_pc scsi --shape dumbbell torus --dataset template --dataset-eps 0.05
uv run python -m toy_3d_pc scsi --supervised --shape torus  # oracle upper bound

# Resume an interrupted run (pass the same flags as the original run):
uv run python -m toy_3d_pc scsi --resume toy_3d_pc_checkpoints/model_em0042.pt --em-steps 100 [... original flags ...]

# Heun integrator with trimmed endpoints:
uv run python -m toy_3d_pc scsi --integrator heun --eps-start 0.01 --eps-final 0.01
```

Key flags (see `cli.py` for all + defaults): `--em-steps`(K) `--training-steps`(T_tr)
`--sample-steps`(ODE steps) `--alpha-z` `--alpha-y` `--ema-decay`(γ) `--pretrain-steps`
`--n-tilts`(K) `--tilt-step` `--tilt-axis` `--splat {gaussian,ball}` `--radius` `--noise-std`
`--coord-noise-std` `--interpolant-style {linear,gvp}` `--shape {torus,dumbbell,trefoil,l_shape,t_shape}`
`--dataset {iid,template}` `--integrator {euler,heun}` `--eps-start` `--eps-final`
`--resume CKPT`. W&B is **on by default**; pass `--no-wandb` to disable.

## Conventions & gotchas (read before editing)

- **No `[-1,1]` normalization.** Clouds live in world coordinates (~`[-1.6, 1.6]`);` extent`maps world half-width to the image;`z ~ N(0,I)`matches the cloud scale. (Unlike the voxel experiments in`simple_3d`/`toy_3d`.)
- **Time convention `t:0→1` = noise→data** (standard flow matching). The pseudocode's
`X_{t=1}=z'` is the reversed convention — same dynamics, integrated `0→1` here. There is **no**
`INTEGRATION_SCALE`: time enters via the model's own sinusoidal `timestep_embedding`.
- `**training_steps` ≠ `sample_steps`.** `training_steps` (T_tr) = inner SGD steps per EM
iteration; `sample_steps` = Euler steps in the transport ODE. The literal loop solves an ODE
*per training step*, so `sample_steps` directly scales cost.
- **Namespaced outputs — do not revert.** Defaults are `toy_3d_pc_checkpoint.pt`,
`toy_3d_pc_eval/`, `toy_3d_pc_checkpoints/`. They are deliberately distinct
from `toy_3d_pointcloud_et`'s `scsi_checkpoint.pt` / `toy3d_pc_eval/` /
`toy3d_pc_scsi_checkpoints/` to avoid clobbering that package's artifacts.
- **Per-EM checkpoint format** (`toy_3d_pc_checkpoints/model_em{k:04d}.pt`): saved by
`save_train_state` — contains `model`, `model_ema`, `optimizer`, `em_step`, `global_step`,
`cfg`. Load with `load_train_state` for resume or `load_checkpoint` (model-only). The
separate `*_ema.pt` files no longer exist; both nets are in the single per-step file.
- **Resume requires the same `--seed` and data flags** as the original run so `y_obs` is
reproduced identically (bootstrap + warmstart are skipped; the EM loop picks up from
`em_step + 1`).
- The model uses only LayerNorm (no BN/dropout), so `transport_sample` needs no train/eval
toggle for correctness.
- `_ball_splat` is non-separable and builds an `(…, N, P, P)` work tensor (chunked over N) — far
costlier than the default Gaussian; keep N/P modest when using `--splat ball`.
- Adding a shape: write an SDF (negative == inside; compose primitives like `_sphere_sd`,
`_capsule_sd`, `_box_sd` with `min`/`max` for unions/intersections) and register a `_Solid(sd,
bbox, oversample)` entry in `data.py::_SHAPE_SOLIDS` — sampling and the
`mixture_volume_residual` diagnostic both derive from it automatically. Thin/sparse solids
(tubes, sparse unions) need a higher `oversample` so `_sample_solid` converges without hitting
`max_rounds`.
- Gitignored ephemera: `*.pt`, `*.png`, checkpoint/eval dirs. Verify with `--debug --no-wandb`.

