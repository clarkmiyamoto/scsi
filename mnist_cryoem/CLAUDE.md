# mnist_cryoem — navigation guide for Claude

Self-contained package: **SCSI (Self-Consistent Stochastic Interpolant) on MNIST under a
CryoEM-style 2D→1D channel.** A ground-truth digit is randomly rotated in-plane (SO(2)),
projected to a 1D signal (sum along one axis), and corrupted with AWGN. SCSI recovers a
generative prior over clean digits *and* the pose from only those 1D observations — a joint
(image, SO(2) pose) product state, with one `ConditionalVelocityCryoEM` predicting both
velocities.

Unlike every other experiment directory in this repo, **the file layout here is deliberately
split to mirror the EM pseudocode 1:1** — each named piece of the algorithm (E-step, M-step,
integrator, interpolant, outer loop) has exactly one file. If you're modifying the algorithm
itself rather than the model/data/channel, `scsi.py` and `em.py` are almost certainly where the
change belongs.

**This package also contains a 3D/CryoET generalization**, living alongside the 2D pipeline in
the SAME files (see "3D/CryoET generalization" below): extruded-MNIST volumes in R^{p^3} under
an SO(3) → 2D-projection → AWGN channel with a physically-motivated mount+fixed-axis tilt
series, reconstructed via a `ConditionalVelocityCryoET3D` (3D UNet image branch + SO(3)
pose-search head). Every 3D-specific symbol is suffixed `_3d` (or, for the CryoET-specific
rotation-sampling functions, `_so3`) and lives in the SAME file as its 2D counterpart — `si.py`
gained rotation math, not a new file — except two wholly new entry points, `pretrain_3d.py` and
`main_3d.py`, mirroring `pretrain.py`/`main.py`.

## Algorithm

```
for k in 1..K:
    # E-step: propose clean data using teacher Phi^(k-1)         [scsi.py::propose_estep]
    idx = sample_pool_indices(N_obs, pool_size)                  # y_i ~ mu
    x_hat, R_hat = ode.sample_joint(model, z'~N(0,I), y_obs[idx])  # Phi_Theta^(k-1)(z'|y)

    # M-step: update student against the teacher-proposed pool   [scsi.py::train_mstep]
    for j in 1..T_tr:
        (x_hat, R_hat) ~ pool
        y_hat = corruption.forward_channel(x_hat, theta=R_hat)   # F(x_hat; R_hat)
        z ~ N(0,I); t ~ U(0,1)
        I_t, I_dot_t = si.interpolant(z, x_hat, t)                # + si.pose_interpolant for R
        loss = || model(I_t, R_t, t, y_hat) - (I_dot_t, R_dot_t) ||^2
        opt.step()

    checkpoint + wandb_logging.log_reconstruction_grid(...)               # [em.py::run_em_loop]
             + wandb_logging.log_em_pool_diagnostics(...)
```

`em.py::run_em_loop` is the `for k in 1..K` line; everything inside one iteration is
`scsi.py`.

`pretrain.py` is an optional stage that runs *before* this loop: ordinary supervised
stochastic-interpolant training on known GT digits, producing a `Theta^(0)` checkpoint that
`main.py --init_ckpt` can load instead of starting the EM loop from a random init. See its
own section below — it deliberately does not reuse `scsi.py::train_mstep`.

## File map

| File | Role |
| --- | --- |
| `si.py` | **Stochastic interpolants only** — pure math, no model/optimizer coupling. `alpha/beta_{linear,gvp}` schedules, `interpolant(x0,x1,t,style)` (image branch, ALSO reused by the 3D pose branch — see below), `wrap_to_pi`, `pose_interpolant(theta_z,theta_hat,t)` (SO(2) geodesic branch, always constant-angular-velocity regardless of `--interpolant_style`). **3D:** `gram_schmidt_to_matrix`/`matrix_to_gram_schmidt` (the 6D continuous rotation representation, Zhou et al. 2019 — `(B,6)` <-> `(B,3,3)`), `rotation_geodesic_angle` (trace-formula geodesic distance, SO(3) analogue of `wrap_to_pi`-based circular error), `axis_angle_to_matrix` (Rodrigues' formula, used by `corruption.py`'s tilt-series construction). No `pose_interpolant_3d` exists — the 3D pose branch reuses `interpolant()` directly, see "3D/CryoET generalization" below. |
| `ode.py` | **Φ, the E-step's integrator.** `sample_joint(model, z_image, y, n_steps, method)` — jointly integrates the image (Euclidean) and pose (SO(2), `wrap_to_pi` after each step) branches from noise to data. This is the seam for tuning the integrator: add a new `method` here without touching `scsi.py`/`si.py`. **Currently Euler-only** (raises on anything else) — matches every other experiment dir in this repo. A future Heun/RK4 needs its own geodesic-aware step for the SO(2) pose branch; it can't just reuse a generic Euclidean implementation for both branches. **3D:** `sample_joint_3d` — unlike the SO(2) case, BOTH branches integrate via plain Euler here (no wrap/exp-map step for pose), since the 3D pose state is the flat 6D representation. |
| `scsi.py` | **The E-step and M-step**, named to match the pseudocode exactly. `sample_pool_indices` + `propose_estep` (E-step, calls `ode.sample_joint`); `loss_func_joint` + `train_mstep` (M-step, calls `corruption.forward_channel` then the loss, logs scalars via `wandb_logging.log_train_step`). **3D:** `propose_estep_3d`, `loss_func_joint_3d` (calls `si.interpolant` TWICE — once for the volume, once for the 6D pose, same function, same `style`), `train_mstep_3d` (`--steps_per_em 1` is impractical at 3D scale — see its docstring). `sample_pool_indices` is reused unmodified by both. |
| `em.py` | **The outer `for k in 1..K` loop.** `run_em_loop(...)` — one E-step then one M-step per `k` so `Phi^(k-1)`/`Theta^(k)` line up, checkpointing, calls `wandb_logging.log_reconstruction_grid` and `wandb_logging.log_em_pool_diagnostics`. **3D:** `run_em_loop_3d`, mirrors it line-for-line with the `_3d` E/M-step and diagnostic calls swapped in. |
| `model.py` | `ConditionalDiT` (image branch, ported from `image_2d/model.py`), `PoseHead` (SO(2) branch, new to this codebase), `ConditionalVelocityCryoEM` (joint wrapper — owns the `y`-broadcast and `t_frac -> t_int` conversion). `IMAGE_SIZE`, `INTEGRATION_SCALE`. **3D:** `VOL_SIZE`, `ConditionalUNet3D` (wraps `diffusers.UNet3DConditionModel`, the only working 3D backbone in this repo — a DiT3D analogue is dead code in `simple_3d/model.py`), `PoseHead3D` (SO(3) pose-search head — Conv3d trunk + MLP taking the RAW `(B,6)` pose state, no matrix conversion needed since the 6D representation has no double-cover sign ambiguity to disambiguate, unlike quaternions), `ConditionalVelocityCryoET3D` (joint wrapper, `arch="unet3d"` only). |
| `corruption.py` | Forward channel `F`: `sample_uniform_angle` (Haar-uniform SO(2), literal `z~N(0,1)` direction), `sample_tilt_series_angles` (T evenly-spaced angles from a random per-acquisition start offset — the tilt-series generalization of `sample_uniform_angle`), `rotate_2d` (black-fill in-plane rotation), `project_1d`, `forward_channel` (rotate → project → AWGN). **3D:** `sample_uniform_rotation_so3` (Haar-uniform SO(3) via `scipy.spatial.transform.Rotation`), `sample_tilt_series_rotations_so3` (mount+fixed-axis tilt series — see "3D/CryoET generalization" below for the composition-order rule), `rotate_3d` (same black-fill shift-trick as `rotate_2d`), `project_2d`, `forward_channel_3d` (public pose type is `(B,6)`, converted to a matrix only internally). |
| `data.py` | `load_mnist_subset`, `build_observations` (applies `corruption.forward_channel` `corruptions_per_object` times per object to build the observation set `mu`; each application is one tilt-series ACQUISITION of `n_tilts` projections via `corruption.sample_tilt_series_angles`; keeps `theta_star` as a diagnostic-only ground truth, never fed to the model; also returns `acq_idx` — which acquisition each observation belongs to, contiguous and in tilt order, so `y_obs[acq_idx == a]` is one full tilt series — used only for grouped visualization in `wandb_logging.py::log_reconstruction_grid`). **3D:** `load_mnist_volumes_3d` (extrudes a shrunk-in-plane MNIST digit into a `p^3` volume with margin on ALL THREE axes — see gotchas below for why depth-only margin isn't enough), `build_observations_3d` (mirrors `build_observations`; its diagnostic-only ground truth `R_star` is kept as `(N,3,3)` MATRICES, not the 6D representation, since it's never fed to the model). |
| `wandb_logging.py` | `log_train_step` (per-SGD-step scalars, extracted from `train_mstep`'s inner loop), `log_reconstruction_grid` (ONE 4-row panel where each COLUMN is a different acquisition/"problem" — column 0 is `fixed_acq_id`, held constant the whole run, the rest freshly randomized each EM step; per column: GT digit / that acquisition's real tilts stacked into a (T,W) sinogram / one generated x_hat conditioned on the middle tilt / that same x_hat reprojected at every tilt's TRUE angle, stacked into a second sinogram directly comparable to the first — runs its own small `ode.sample_joint` call rather than reusing `scsi.py::propose_estep`, since `scsi.py` already imports this module and importing back would be circular), `log_em_pool_diagnostics` (scalar-only mean circular-error health check over the whole training pool), `log_overfit_step` (per-step loss/loss_image/loss_pose/grad_norm for `overfit.py`, no images). Degrades gracefully without wandb (`_WANDB_AVAILABLE`), same pattern as everywhere else in the repo. **3D:** `log_pretrain_reconstruction_3d`, `log_em_pool_diagnostics_3d` (`si.rotation_geodesic_angle` replaces `wrap_to_pi`-based error), `log_reconstruction_grid_3d` (each observation is a full `(H,W)` image here, not a 1D signal, so the sinogram-stacking trick is replaced by `_projection_mosaic` — xy\|xz\|yz sum-projection mosaics — for GT/generated volumes, and horizontal strips of a few preview tilts for observations/recorruptions). Both `log_pretrain_reconstruction_3d` and `log_reconstruction_grid_3d` ALSO log an interactive companion, reusing the SAME `x_hat` the static mosaic panel already computed (no extra `sample_joint_3d` call): `_volume_to_pointcloud` converts a `(D,H,W)` volume into an `(N,6)` `[x y z r g b]` point cloud (per-volume top-`keep_frac` quantile threshold, NOT a fixed cutoff — `x_hat` has no guaranteed range, and an untrained model legitimately renders as noise under this scheme, that's the diagnostic not a bug), logged as a list of `wandb.Object3D` under one key (`pretrain/generations_3d`, `em/generations_3d`) so wandb's media panel renders a steppable, individually-orbitable 3D gallery. GT volumes are deliberately NOT logged this way (the static mosaic already carries that comparison; GT never changes call to call, so repeating the upload every `--plot_every`/EM-step would be pure waste). |
| `main.py` | CLI (`argparse`), device autodetect, `--debug` tiny-run defaults, dataset load, model/optimizer construction, optional `--init_ckpt` load, `wandb.init`/`finish`, one call to `em.run_em_loop(...)`. Also owns `--overfit` dispatch (see `overfit.py`). |
| `overfit.py` | **`--overfit` sanity check, not part of the SCSI algorithm.** `overfit_single_batch(model, x_batch, theta_batch, y_batch, ..., loss_fn=loss_func_joint)` — draws one fixed `(x_hat, theta_hat, y)` batch and runs many SGD steps of `loss_fn` against it with a throwaway optimizer, logging every step via `wandb_logging.log_overfit_step`. Only the interpolant's own randomness (`t`, `z_image`/`z_vol`, `theta_z`/`pose_z`) varies step to step, so the loss floor is the irreducible conditional variance, not zero — see the module docstring before reading a plateau as a bug. Dispatched from `main.py`/`main_3d.py` right after model construction; exits before the EM loop is ever called. The `loss_fn` parameter is the ONE place this file was extended in place rather than getting a `_3d` copy — `main_3d.py` passes `loss_fn=scsi.loss_func_joint_3d`; every existing 2D caller is unaffected since it's a keyword default. |
| `pretrain.py` | **Supervised warm-start for `Theta^(0)`.** Flat stochastic-interpolant SGD loop (no outer/inner loop) directly on `scsi.py::loss_func_joint` — draws a fresh random rotation `R` per step, keeps the GT digit canonical, computes `y = corruption.forward_channel(x_i, theta=R)`. Pluggable pool selection via `SELECTION_STRATEGIES` (currently one entry, `per_class`, drawing `--n_pretrain_images_per_class` images from each of `--digit_classes`). Saves a `model.state_dict()` checkpoint that `main.py --init_ckpt` can load directly. |
| `pretrain_3d.py` | **NEW FILE, not an extension of `pretrain.py`.** 3D/CryoET analogue — same flat-SGD structure, `select_per_class` calls `data.load_mnist_volumes_3d`, trains `scsi.loss_func_joint_3d`, saves to `mnist_cryoet_checkpoints/pretrain_theta0.pt` (separate namespace from the 2D pipeline's `mnist_cryoem_checkpoints/`). |
| `main_3d.py` | **NEW FILE, not an extension of `main.py`.** 3D/CryoET analogue — same CLI/dispatch structure, loads via `data.load_mnist_volumes_3d`/`build_observations_3d`, builds `ConditionalVelocityCryoET3D`, runs `em.run_em_loop_3d`. Defaults differ from `main.py` for compute-cost reasons: `--batch_size 8` (not 256), `--sample_steps 20` (not 50) — 3D UNet passes are much heavier than the 2D DiT/UNet2D. |
| `test_so3_math.py` | **NEW FILE, validation gate for the 3D generalization — run to green BEFORE trusting `pretrain_3d.py`/`main_3d.py` output.** Plain-assert script (`uv run python test_so3_math.py`), pytest-compatible. Checks: Gram-Schmidt output is always a proper rotation; the 6D<->matrix round-trip is exact; the tilt-series composition order shares one physical axis per acquisition (see gotcha below — this is the single highest-risk correctness surface in the 3D generalization); rotating a volume doesn't change its projected mass (catches clipping/margin bugs); the pose-branch interpolant/integrator pairing is exact for the `linear` schedule. |

## How to run

```bash
uv run python main.py --debug --no_wandb    # ~seconds smoke test, no wandb needed
uv run python main.py                       # full run, default hyperparameters
uv run python main.py --steps_per_em 1      # literal pseudocode: fresh Phi^(k-1) draw every SGD step
uv run python main.py --overfit --no_wandb  # sanity check only: overfit one fixed batch, exit (no EM loop)

uv run python pretrain.py --debug --no_wandb                     # ~seconds smoke test
uv run python pretrain.py --digit_classes 3 7 --n_pretrain_images_per_class 4
uv run python main.py --init_ckpt mnist_cryoem_checkpoints/pretrain_theta0.pt   # warm-started EM
```

Key `main.py` flags (see `main.py::parse_args` for the full list): `--n_em_steps`(K)
`--steps_per_em`(T_tr) `--steps_first_em` `--sample_steps` (Euler steps for Φ)
`--interpolant_style {linear,gvp}` (image branch only) `--pose_loss_weight`
`--digit_classes` (list, default: all 10) `--n_images_per_class` (PER class, not a total)
`--init_ckpt` `--corruptions_per_object` (independent tilt-series acquisitions per object)
`--n_tilts` (T, tilts per acquisition; default 1 = old single-random-rotation behavior)
`--tilt_increment_deg` (angular step between tilts within one acquisition; unused if
`--n_tilts 1`) `--noise_std` `--batch_size` `--lr` `--no_wandb`
`--overfit` (mode switch: run `overfit.py`'s single-batch sanity check and exit, skipping the
EM loop entirely — batch is the first `min(--batch_size, len(x_gt))` GT digits) `--overfit_steps`.

Key `pretrain.py` flags (see `pretrain.py::parse_args`): `--digit_classes`
`--n_pretrain_images_per_class` `--selection_strategy` `--n_steps` `--checkpoint_every`
`--out_ckpt`, plus the same `--interpolant_style`/`--pose_loss_weight`/`--noise_std`/
`--batch_size`/`--lr`/`--no_wandb` as `main.py`. `--digit_classes`/`--n_images_per_class` on
`main.py` and `--digit_classes`/`--n_pretrain_images_per_class` on `pretrain.py` are
independent — the pretraining pool is meant to be tiny, the EM problem's dataset is not.

## 3D/CryoET generalization

```bash
uv run python test_so3_math.py                    # validation gate — run this FIRST, always

uv run python main_3d.py --debug --no_wandb       # ~seconds smoke test
uv run python main_3d.py                          # full run, default hyperparameters
uv run python main_3d.py --overfit --no_wandb     # sanity check only, no EM loop

uv run python pretrain_3d.py --debug --no_wandb                          # ~seconds smoke test
uv run python pretrain_3d.py --digit_classes 3 7 --n_pretrain_images_per_class 4
uv run python main_3d.py --init_ckpt mnist_cryoet_checkpoints/pretrain_theta0.pt
```

The extra `main_3d.py`/`pretrain_3d.py` flags beyond the 2D set: `--vol_size` (p, default 32),
`--inplane_size`/`--depth_extent` (volume-construction margins, see gotcha below),
`--tilt_axis` (3 floats, the FIXED lab-frame tilt axis).

**Pose representation: the 6D continuous rotation representation (Zhou et al., CVPR 2019;
Gram-Schmidt orthogonalization, `si.gram_schmidt_to_matrix`/`matrix_to_gram_schmidt`), NOT
quaternions.** This was a deliberate design choice: representing pose as a flat, unconstrained
`(B,6)` vector rather than a point on the SO(3)/quaternion manifold means the pose branch needs
**no geodesic interpolant and no manifold-aware ODE step at all** — it reuses `si.interpolant()`
verbatim (the SAME function the image branch uses, just called a second time with a `(B,1)`
time-broadcast instead of `(B,1,1,1,1)`) and integrates via plain Euler
(`pose = pose + v_pose * dt`), exactly like the image/volume branch. A rotation matrix is only
ever materialized (via `gram_schmidt_to_matrix`) at the two places that actually need one:
`corruption.rotate_3d` and `si.rotation_geodesic_angle`. If you're tempted to add quaternion
machinery here, don't — that was an earlier draft, deliberately abandoned because it needed a
slerp interpolant plus a body-frame-vs-space-frame sign convention that's exactly the kind of
silent, hard-to-debug correctness bug this repo already has one instance of (see the
canonical-target gotcha below).

## Conventions & gotchas (read before editing)

- **`logging.py` is not a valid filename here — use `wandb_logging.py`.** `main.py` runs as a
  script, so its directory is `sys.path[0]` and is searched *before* site-packages: a local
  `logging.py` would shadow the stdlib `logging` module for every transitive import in the
  process (torch, diffusers, tqdm, and wandb all do `import logging` internally). Verified
  empirically — `import logging` from this directory with a `logging.py` present raises
  `AttributeError: module 'logging' has no attribute 'getLogger'`. Don't rename it back.
- **E-step/M-step follow the textbook EM convention here** (`propose_estep` proposes data,
  `train_mstep` trains the network) — the opposite of `image_2d/`, `simple_3d/`, and `toy_3d/`,
  which name the training call `train_estep`. See the top-level `CLAUDE.md` Gotchas for the
  full explanation; this directory is the reference example of the textbook naming.
- `ode.sample_joint` and `scsi.loss_func_joint`/`train_mstep` both operate on the **joint**
  `(image, pose)` state — there's no way to train/sample the image branch alone. If you add a
  new branch to the state (e.g. a scale factor), it needs updates in all three of `si.py`
  (interpolant), `ode.py` (integrator step), and `model.py` (velocity head), not just one.
- `train_mstep`'s optimizer is created once in `main.py` and passed in, persisting across every
  EM outer iteration — required for `--steps_per_em` as low as 1 to be a fair test of the
  literal pseudocode rather than being crippled by constantly-reset Adam moment estimates.
- `theta_star` (the true rotation) is threaded through `data.py` → `main.py` → `em.py` →
  `wandb_logging.py` purely as a diagnostic (circular-error metric, GT-digit panel row) — never
  passed to the model or used in `scsi.py`'s loss.
- Gitignored ephemera: `mnist_cryoem_checkpoints/`, `mnist_cryoem_eval/`, `wandb/`,
  `__pycache__/`. Verify changes with `--debug --no_wandb` before a full run.
- **The image branch's target is always the canonical (unrotated) digit — never a rotated
  one.** This applies everywhere `x_hat`/`x1` is used as an interpolant target
  (`scsi.py::train_mstep`, `pretrain.py`): the *pose* branch carries the rotation, and
  `corruption.forward_channel(x_hat, theta=R_hat)` RE-APPLIES `R_hat` to reconstruct `y_hat`.
  If the image target were already rotated, that reconstruction would double-rotate it —
  silently wrong, no shape error, so it's easy to get backwards. `pretrain.py` in particular:
  its GT images from `data.py` are left untouched; only a freshly-drawn `theta` is passed to
  `forward_channel`, never `rotate_2d` applied to the image target itself.
- **`--n_images_per_class`/`--n_pretrain_images_per_class` are PER CLASS, not totals**, and
  `--digit_classes` defaults to all 10 MNIST classes on both `main.py` and `pretrain.py`. A
  bare `uv run python main.py` therefore loads `2 * 10 = 20` GT digits by default, not 2 —
  narrow with `--digit_classes 3` (etc.) to get a single-class run.
- **3D: the canonical-target invariant applies exactly as in 2D, regardless of pose
  representation.** The volume branch's target is always the canonical (unrotated) digit —
  never a rotated one — everywhere `x_hat`/`x1` is used as an interpolant target
  (`scsi.py::train_mstep_3d`, `pretrain_3d.py`): the pose branch carries the rotation, and
  `corruption.forward_channel_3d(x_hat, pose6=pose6_hat)` RE-APPLIES it to reconstruct `y_hat`.
  Same silent-bug shape as the 2D gotcha above (no shape error, no exception — just
  qualitatively wrong reconstructions) — switching from quaternions to the 6D representation
  during design eliminated the SO(3) slerp/frame-convention risk entirely, but did NOT touch
  this one, since it has nothing to do with how pose is represented.
- **3D: `sample_tilt_series_rotations_so3`'s composition order is `R_total = R_tilt @
  R_mount`** — tilt OUTER (applied in the lab frame, second), mount INNER (applied in the
  specimen frame, first). Swapping this breaks the single-tilt-axis property (each acquisition
  would tilt about a different, mount-dependent axis — no longer a real tilt series), silently:
  no shape error, just physically wrong data generation. `test_so3_math.py`'s
  `test_tilt_composition_order` guards this directly; it checks `R_total^{-1} @ tilt_axis`
  (NOT `R_total @ tilt_axis`, which is not the invariant quantity) stays constant across one
  acquisition's tilts.
- **3D: volume construction needs margin on ALL THREE axes, not just depth.** A naive
  "shrink nothing, extrude with only a depth margin" scheme lets in-plane ink reach the cube's
  corners, which a generic SO(3) rotation (unlike 2D's SO(2), confined to one plane) can then
  clip. `data.load_mnist_volumes_3d` shrinks the digit in-plane (`--inplane_size`) AND limits
  the extruded depth band (`--depth_extent`) — both are needed together. `test_so3_math.py`'s
  `test_mass_invariance_no_clipping` is the empirical gate on whatever margins you pick — trust
  that over arithmetic, since anti-aliased resizing interacts with the exact margin needed.
- **3D gitignored ephemera** (separate namespace from the 2D pipeline, so both can be run from
  this directory without collisions): `mnist_cryoet_checkpoints/`, `mnist_cryoet_eval/` — same
  `*.pt`/`*.png` gitignore patterns as the 2D pipeline's `mnist_cryoem_checkpoints/`/
  `mnist_cryoem_eval/` already cover these.
