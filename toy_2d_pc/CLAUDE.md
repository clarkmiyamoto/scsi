# toy_2d_pc — navigation guide for Claude

Self-contained package: **lifted SCSI (Self-Consistent Stochastic Interpolant) for a
CryoEM-style channel in the 2D point-cloud representation.** A 2D object is a *set* of `N`
points `X ∈ ℝ^{N×2}` — a point-cloud representation of an MNIST digit. The CryoEM channel `F`
renders it as a `(P, P)` density image, then projects a noisy tilt series of **1D**
sinograms under one unknown global SO(2) pose; SCSI recovers a generative prior over clean
clouds from only those sinograms.

This is the **dimension-down analogue** of `toy_3d_pc` (2D↔1D instead of 3D↔2D): the same
literal-pseudocode SCSI algorithm (fused on-the-fly EMA transport, `α_y`/`α_z` coupling, outer
EMA), but cheap enough to iterate on in seconds. Run everything via
`uv run python -m toy_2d_pc scsi …` (CPU on darwin, CUDA/MPS elsewhere).

## What's different from `toy_3d_pc`

- **Rendering is a separate stage from the channel.** `renderer.py` turns the point cloud
  into an actual `(P, P)` image in its own canonical (unrotated) frame — gaussian / disk /
  histogram. `corruption.py`'s `forward_channel` then rotates *that image* (bilinear,
  `affine_grid`/`grid_sample`) and sums along one axis per tilt — a literal rotate-then-sum
  Radon transform. `toy_3d_pc` doesn't have this split: its `G` splats directly onto the
  already-projected 1D detector coordinate, no intermediate image is ever built.
- **SO(2) is abelian.** The unknown global pose and each known tilt angle add into a *single*
  combined rotation per tilt (`theta_global + Δθ_k`), so `forward_channel` does one image
  rotation per tilt — not two composed rotation stages like `toy_3d_pc`'s non-commuting SO(3).
- **`--splat` has no default** (`gaussian` / `disk` / `histogram`) — must be passed explicitly.
  `histogram` (hard lattice binning) has no `toy_3d_pc` analogue.
- **Data is real MNIST, not a synthetic SDF shape.** `data.py` pools the first `--n-images`
  training examples of one `--digit` class (dataset order, not shuffled) and samples point
  clouds from pixel-intensity density (inverse-CDF `multinomial` + sub-pixel jitter). No
  `available_shapes()`/SDF registry, no `mixture_volume_residual` diagnostic (no ground-truth
  volume for a real image).
- **No `--tilt-axis`.** 2D has only one rotation plane.
- **No `pca_canonicalize`.** `toy_3d_pc`'s own docs record that the PCA/moment-axis variant is
  reference-free and its frame gets picked by per-sample sampling noise under
  symmetry/near-degenerate eigenvalues, which hurt results there — `canonicalize.py` here
  implements only `reference_canonicalize` (the one actually used by `--canonicalize`),
  deliberately not reintroducing that superseded path.
- **`log_em_step`'s visualization collapses two `toy_3d_pc` panels into one.** In 3D, a single
  object's row showed one central-tilt image, plus a separate all-tilts grid panel. In 2D a
  lone 1D signal is uninformative alone, so the natural visualization is the full **sinogram**
  (`K` tilts × `P` detector positions) as a single grayscale heatmap — richer than either 3D
  panel and needs no second panel.

## File map

| File | Role |
| --- | --- |
| `renderer.py` | **`G`**: point cloud → `(P, P)` canonical-frame image. `gaussian_splat`, `disk_splat` (filled-disk chord-length profile), `histogram_splat` (hard lattice binning), `render(..., kind=...)` dispatcher (`kind` required, no default). |
| `corruption.py` | **Forward model `F`** (`forward_channel`): rotates the rendered image by each combined `theta_global + Δθ_k` angle (`_rotate_images`, `affine_grid`/`grid_sample`) and sums rows → `(B, K, P)` sinogram + AWGN. Plus pseudo-inverse `F†` (`pseudo_inverse`/`backproject_tomo`, 2D space-carving) + rotation helpers (`tilt_angles`, `random_rotations`, `rotate_clouds` — the last still point-level, used by `warmstart`). |
| `data.py` | `load_mnist_digit_pool` (first N images of one digit class), `image_to_pointcloud`/`_images_to_pointclouds` (density → points via `multinomial` + jitter), `make_mnist_sampler` (pool → `(batch, n_points, device) -> (B,N,2)` sampler). |
| `si.py` | Stochastic interpolant: `linear`/`gvp` schedules, `interpolant(z,x,t,style)→(I_t, İ_t)`, `transport_sample` (Euler or Heun ODE). Dimension-agnostic, near-verbatim port of `toy_3d_pc/si.py`. |
| `model.py` | `ConditionalPointCloudVelocity` (permutation-equivariant set-transformer + 1D signal cross-attn via `SignalPatchEncoder`, `Conv1d` over the `(B,K,P)` sinogram), `ConditionalModelConfig`, `build_conditional_model`, EMA helpers `clone_ema`/`ema_update_outer`. |
| `warmstart.py` | **Algorithm 1** — `find_initialization`: train `b̂^(0)` on `(g·F†(y), y)`. |
| `canonicalize.py` | Canonicalization operator `C` — **`reference_canonicalize` only** (2D ICP: `kabsch` with 2×2 SVD, `icp_align`, `seed_reference`, `update_reference`). No PCA variant (see above). |
| `scsi.py` | **Algorithm 2** — `scsi_train` (the literal EM loop) + `log_em_step`/`log_bootstrap` (2D scatter + sinogram heatmap panels) + checkpoint I/O. |
| `supervised.py` | `train_supervised` debug oracle: train directly on `(x, F(x))` with fresh GT (no EM). |
| `device.py` | CUDA>MPS>CPU autodetect, `autocast`, `needs_grad_scaler`, `configure_backends`. Verbatim port. |
| `tracking.py` | W&B `Tracker`, enabled by default (graceful no-op if wandb missing/unconfigured). No `log_clouds`/`log_meshes` (2D panels are PNGs, logged via `log_image`). |
| `plot.py` | 2D scatter PNG (`save_scatter`). |
| `cli.py`, `__main__.py`, `__init__.py` | argparse `scsi` subcommand; entry point; public exports. |

## Forward model `F` (`corruption.py` + `renderer.py`)

```
img = G(X)                                       # (P, P) canonical-frame density (renderer.py)
F(X) = { sum_rows( Rot(theta + n·Δθ) . img ) + Z }_{n=1..K}   ->  (B, K, P)
```

- `G` = `render(..., kind={"gaussian","disk","histogram"})`, required, no default.
- `theta` = one Haar-uniform SO(2) pose per cloud (the unknown nuisance).
- `Δθ` = `K` known single-axis tilt increments (`tilt_step°` apart, centered at 0).
- Rotation is applied to the **image** (bilinear `affine_grid`/`grid_sample`), not the points —
  and since SO(2) is abelian, `theta + Δθ_k` is one combined angle, one rotation per tilt.
- `F†` (`pseudo_inverse`) = soft space-carving back-projection of the K tilts → 2D occupancy
  grid → sampled point cloud, using **only** the known tilt geometry (residual global pose left
  for EM). Used by the warm-start and to seed π(0).

## Algorithm (`scsi.py::scsi_train`)

```
pool = MNIST digit pool (first --n-images of --digit)
gt   = point clouds sampled from the pool                # mu-source
y_obs = F(gt)                                             # mu; GT used ONLY for observations + eval
x_boot = F†(y_obs)                                        # pseudo-inverse bootstrap (π(0))
model ← find_initialization(...)                          # Algorithm 1 warm-start  →  Θ^(0)
model_ema ← clone_ema(model)                               # Θ_EMA^(0) ← Θ^(0)
opt = AdamW(model.parameters())                            # persistent across outer iters
for k in 1..em_steps:                                      # EMA frozen during this inner loop
    for i in 1..training_steps:                            # T_tr inner SGD steps
        y = minibatch(y_obs)
        z' ~ N(0,I);  x̂ = transport_sample(model_ema, z', y, sample_steps)   # Φ_EMA(z'|y)
        x̂_C = C(x̂)  if --canonicalize else x̂                                 # ICP-align x̂ to shared reference frame
        z  = z' w.p. α_z else N(0,I)                                          # noise coupling
        ŷ  = F(x̂);  ŷ = y w.p. α_y else ŷ                                     # obs coupling (always uses x̂, not x̂_C)
        I_t, İ_t = interpolant(z, x̂_C, t);  loss = ‖model(I_t,t,ŷ) − İ_t‖²; opt.step()
    ema_update_outer(model_ema, model, γ)                                     # outer EMA
    log_em_step(...)                       # sample π(k) with model_ema; 4-row panel + sinograms
```

## How to run

```bash
uv run python -m toy_2d_pc scsi --splat gaussian --debug --no-wandb   # ~seconds smoke test
uv run python -m toy_2d_pc scsi --splat disk --digit 3 --n-images 128
uv run python -m toy_2d_pc scsi --splat histogram --canonicalize
uv run python -m toy_2d_pc scsi --splat gaussian --supervised --debug --no-wandb  # oracle

# Resume an interrupted run (pass the same flags as the original run):
uv run python -m toy_2d_pc scsi --splat gaussian --resume toy_2d_pc_checkpoints/<out-stem>/model_em0042.pt --em-steps 100 [... original flags ...]
```

Key flags (see `cli.py` for all + defaults): `--splat {gaussian,disk,histogram}` (**required**)
`--digit` `--n-images` `--mnist-size` `--em-steps`(K) `--training-steps`(T_tr)
`--sample-steps`(ODE steps) `--alpha-z` `--alpha-y` `--ema-decay`(γ) `--pretrain-steps`
`--n-tilts`(K) `--tilt-step` `--radius` `--noise-std` `--coord-noise-std`
`--interpolant-style {linear,gvp}` `--integrator {euler,heun}` `--eps-start` `--eps-final`
`--resume CKPT` `--canonicalize` (off by default; `--canon-icp-iters` `--canon-icp-restarts`
`--canon-ref-decay` tune it). W&B is **on by default**; pass `--no-wandb` to disable.

## Conventions & gotchas (read before editing)

- **No `[-1,1]` normalization.** Clouds live in world coordinates (extent-scaled, default
  `[-2, 2]`); `z ~ N(0,I)` matches the cloud scale, same convention as `toy_3d_pc`.
- **Time convention `t:0→1` = noise→data**, no `INTEGRATION_SCALE` — same as `toy_3d_pc`.
- `training_steps` (T_tr, inner SGD steps) ≠ `sample_steps` (Euler/Heun steps in the transport
  ODE, solved *per training step* — directly scales cost).
- **Namespaced outputs — do not revert.** Defaults are `toy_2d_pc_checkpoint.pt`,
  `toy_2d_pc_eval/`, `toy_2d_pc_checkpoints/`, distinct from every sibling package's
  artifacts (`toy_3d_pc_*`, `toy3d_pc_*`, ...) so runs never clobber each other.
- **Per-EM checkpoint format** (`toy_2d_pc_checkpoints/<out-stem>/model_em{k:04d}.pt`): saved
  by `save_train_state` — `model`, `model_ema`, `optimizer`, `em_step`, `global_step`, `cfg`,
  `reference`. Load with `load_train_state` for resume (7-tuple ending in `reference`) or
  `load_checkpoint` (model-only). `--resume` requires the same `--seed`/`--digit`/`--n-images`
  flags as the original run so `y_obs` is reproduced identically.
- The model uses only LayerNorm (no BN/dropout), so `transport_sample` needs no train/eval
  toggle for correctness.
- `disk_splat` is non-separable and builds a `(…, N, P, P)` work tensor (chunked over N) —
  costlier than `gaussian`; keep N/P modest when using `--splat disk`.
- **`--canonicalize` only reframes the interpolant target, nothing else** — same contract as
  `toy_3d_pc`: `x̂_C = C(x̂)` replaces `x̂` in `I_t`/`İ_t` only; `ŷ = F(x̂)` and the `α_z` noise
  coupling still use the uncanonicalized `x̂` and `F`'s own fresh random pose.
- **Reference lifecycle** (same as `toy_3d_pc`): `seed_reference` picks one representative
  cloud (canonical frame defined only up to a global rotation); `update_reference` EMA-blends
  via ICP correspondences (`--canon-ref-decay`, 1.0=freeze); persisted in the per-EM checkpoint,
  GT-free.
- `kabsch` forbids reflections (det-fix) so chiral point sets are never mirrored;
  `torch.linalg.svd` has no MPS kernel, so the tiny `(…,2,2)` SVD routes through CPU **only on
  MPS** (native on CUDA, no per-step sync) — same pattern as `toy_3d_pc`.
- `--mnist-size` controls the resolution MNIST images are resized to *before* point-cloud
  sampling (default 28, native resolution) — distinct from `--image-size` (`P`, the sinogram
  detector length / rendered-image resolution used by the forward channel).
- Gitignored ephemera: `*.pt`, `*.png`, checkpoint/eval dirs. Verify with `--debug --no-wandb`.
