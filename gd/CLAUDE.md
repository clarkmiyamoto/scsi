# gd — direct gradient-descent CryoET reconstruction

Standalone experiment, self-contained (no imports from sibling `scsi/` folders, per
top-level `CLAUDE.md` convention). Given one known 3D shape, render it into a finite
CryoET tilt-series dataset, then recover it by running Adam **directly** on the
shape's own parameters (point positions or voxel occupancy) — no neural network, no
EM prior update. A baseline to test whether raw gradient descent alone can undo the
corruption channel.

## Run it

```bash
uv run python -m gd --debug                      # tiny smoke config, ~seconds
uv run python -m gd --shape torus --param pointcloud --method distributional
uv run python -m gd --shape torus --param voxel --method joint_pose
```

All flags are in `cli.py`. W&B logging is **always on** here (unlike sibling
packages, where it's optional) — every run needs `wandb login` first. `--debug`
overrides sizes/steps to a config that runs in seconds; use it to sanity-check any
change before a real run.

## Module map

| File | Role |
|---|---|
| `corruption.py` | The one shared forward channel, `render_projection`: rotates weighted 3D points through known tilts + a random global pose, splats to images, adds AWGN. Also `carve_occupancy`/`sample_points_from_occupancy` (space-carving back-projection init). |
| `shapes.py` | Ground-truth generators (`sphere`/`torus`/`cylinder`/`helix`), as point clouds or voxel occupancy grids. |
| `parameterize.py` | The optimizable `X`: `PointCloudParam` (learnable positions, fixed weights) and `VoxelParam` (fixed grid coords, learnable `sigmoid(logits)` weights). Both expose `.render_args() -> (positions, weights)`. |
| `dataset.py` | Builds the fixed observed dataset `D = {y_i}` once (N noisy re-renders of one ground-truth template). |
| `recon.py` | Pluggable loss registry, `RECON_METHODS = {"distributional", "joint_pose"}`. Add a method by subclassing `ReconMethod` and registering it here. |
| `canonicalize.py` | `kabsch`/`chamfer`/`icp_align` — blind rigid alignment (X is only ever recovered up to an unknown global rotation). |
| `eval.py` | `evaluate_reconstruction`: ICP-aligns `X` to ground truth, reports Chamfer distance. |
| `optimize.py` | The training loop: build dataset → build `X` → Adam → periodic eval + viz + W&B log. |
| `plot.py` / `tracking.py` | 3D scatter / tilt montage PNGs; always-on W&B wrapper (`log_cloud` uses `wandb.Object3D` — rotatable in the UI). |
| `cli.py` | argparse entry point (`python -m gd`). |

## Key design points

- **Point cloud and voxel share one renderer.** Both reduce to "weighted 3D
  points" fed through `render_projection`; a voxel isn't rendered by a separate
  Radon/resample path. New parameterizations should follow the same
  `(positions, weights)` interface.
- **Splat kernels** (`--splat-kernel gaussian|ball|cube`, `--splat-size`): gaussian
  and cube are separable (`O(N·P)`, fast); ball is not (`O(N·P²)`) — avoid large
  `--vol-size` with `--splat-kernel ball`.
- **Voxel point count scales as `vol_size³`.** `--gt-resolution` defaults to
  exactly `vol_size` for the voxel param (not multiplied) — a naive per-axis
  multiplier here previously caused a ~150 GiB OOM. Render cost is
  `O(n_objects * n_tilts * resolution * image_size)`; be conservative bumping any
  of those for `--param voxel`.
- **Both recon methods share `render_projection`** but call it differently:
  `distributional` renders a fresh batch with random poses + noise (matching the
  real corruption process) and compares distributions (SWD/MMD); `joint_pose`
  renders with each real particle's *learned* pose (`noise_std=0`, since it's a
  direct least-squares fit to noisy targets, not a distribution match).
- **Eval requires ICP**, not raw Chamfer — the global pose is never observed, only
  resolved up to rotation.

## Known behavior, not a bug

Under `distributional`, Chamfer distance typically improves fast early (shape
topology resolves — e.g. torus ring forms by ~step 300) then plateaus or wobbles
rather than keeps improving. This looks like point-clumping: nothing in the loss
penalizes multiple points landing near each other, since overlapping splats barely
change the rendered image. Not currently fixed; a repulsion term would be the
natural next step if reconstruction fidelity (not just topology) matters.

With `--param voxel` and a high `--noise-std`, Chamfer can similarly *increase*
over long runs even though the true-shape voxels are already well resolved: once
those voxels' logits saturate (sigmoid gradient ≈ 0 near 0/1), Adam's
adaptive step size keeps nudging background voxels sitting in the sigmoid's
sensitive middle region, and noisy gradients slowly inflate them into
salt-and-pepper background mass that `voxel_to_pointcloud`'s weighted sampling
turns into stray points. `--lr-schedule {cosine,linear,exponential}` (with
`--lr-min-frac`) anneals the LR to stop this late-training drift, and
`--l1-weight` adds a sparsity penalty on voxel weights to suppress it at the
source; both are configured in `optimize.py`/`cli.py`.

## Before changing the renderer/loss math

Run all four smoke combos and eyeball the output PNGs — cheap and catches shape/
broadcast bugs immediately:

```bash
for param in pointcloud voxel; do
  for method in distributional joint_pose; do
    uv run python -m gd --debug --param $param --method $method
  done
done
```
