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

**This package also contains a rotational-MRA generalization**, living alongside the 2D CryoEM
pipeline in the SAME files (see "Rotational MRA generalization" below): the SAME in-plane SO(2)
rotation as the CryoEM channel, but with the 1D projection step removed — `F(x) = R∘x + W` stays
a full 2D image, reconstructed via a `ConditionalVelocityMRA` that reuses the CryoEM pipeline's
`ConditionalDiT`/`ConditionalUNet2D`/`PoseHead` completely unchanged. Every MRA-specific symbol is
suffixed `_mra` and lives in the SAME file as its 2D CryoEM counterpart, mirroring the `_3d`
convention above — except one wholly new entry point, `main_mra_rotation.py`, mirroring `main.py`.

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
| `ode.py` | **Φ, the E-step's integrator.** `sample_joint(model, z_image, y, n_steps, method)` — jointly integrates the image (Euclidean) and pose (SO(2), `wrap_to_pi` after each step) branches from noise to data. This is the seam for tuning the integrator: add a new `method` here without touching `scsi.py`/`si.py`. **Currently Euler-only** (raises on anything else) — matches every other experiment dir in this repo. A future Heun/RK4 needs its own geodesic-aware step for the SO(2) pose branch; it can't just reuse a generic Euclidean implementation for both branches. **The pose/latent branch is OPTIONAL**: `sample_joint` checks `model.pose_branch is not None` (the single source of truth also used by `scsi.propose_estep`/`loss_func_joint`/`train_mstep_mra`) and, if absent, never initializes/updates `theta`, returning `theta=None` — see `model.ConditionalVelocityMRA`'s `use_pose_head` flag and the R-optional gotcha below. **3D:** `sample_joint_3d` — unlike the SO(2) case, BOTH branches integrate via plain Euler here (no wrap/exp-map step for pose), since the 3D pose state is the flat 6D representation; pose is NOT optional here (no `use_pose_head` on `ConditionalVelocityCryoET3D`). **MRA:** `sample_joint` is reused unmodified — it never inspects `y`'s shape, just forwards it opaquely to `model.forward`. |
| `scsi.py` | **The E-step and M-step**, named to match the pseudocode exactly. `sample_pool_indices` + `propose_estep` (E-step, calls `ode.sample_joint`); `loss_func_joint` + `train_mstep` (M-step, calls `corruption.forward_channel` then the loss, logs scalars via `wandb_logging.log_train_step`). **3D:** `propose_estep_3d`, `loss_func_joint_3d` (calls `si.interpolant` TWICE — once for the volume, once for the 6D pose, same function, same `style`), `train_mstep_3d` (`--steps_per_em 1` is impractical at 3D scale — see its docstring). `sample_pool_indices` is reused unmodified by both. **MRA:** `train_mstep_mra` (identical to `train_mstep` except it calls `corruption.forward_channel_mra` instead of `forward_channel`) is the ONLY MRA-specific symbol here — `propose_estep` and `loss_func_joint` are reused unmodified, since neither ever hardcodes a channel call or inspects `y`'s shape. **`propose_estep`, `loss_func_joint`, and `train_mstep_mra` all treat the pose/latent `R` as OPTIONAL**, gated on `model.pose_branch is not None` (never on whether a caller-supplied `theta_hat` happens to be non-None — see the R-optional gotcha below): `propose_estep` returns `theta_pool=None`; `loss_func_joint` skips `pose_interpolant` and returns `loss_pose=0`; `train_mstep_mra` builds `theta_batch=None`, which sends `forward_channel_mra` into its own fresh-random-rotation branch instead of re-applying a recovered one — i.e. the M-step marginalizes over `R` instead of estimating it. `--no_pose_head` on `main_mra_rotation.py` is the only call site that currently constructs a model with `use_pose_head=False`; `train_mstep` (CryoEM) is NOT similarly augmented (not requested, `ConditionalVelocityCryoEM` has no `use_pose_head` flag). |
| `em.py` | **The outer `for k in 1..K` loop.** `run_em_loop(...)` — one E-step then one M-step per `k` so `Phi^(k-1)`/`Theta^(k)` line up, checkpointing, calls `wandb_logging.log_reconstruction_grid` and `wandb_logging.log_em_pool_diagnostics`. **3D:** `run_em_loop_3d`, mirrors it line-for-line with the `_3d` E/M-step and diagnostic calls swapped in. **MRA:** `run_em_loop_mra`, mirrors it too, except there's no `acq_idx`/`n_acq` (no tilt-series grouping) — a single `fixed_obs_id` (observation index) replaces `fixed_acq_id`; the E-step (`propose_estep`) and `log_em_pool_diagnostics` calls are reused unmodified, only `log_reconstruction_grid_mra`/`train_mstep_mra` are MRA-specific. `run_em_loop_mra` needed ZERO changes to support `--no_pose_head`: it already threads `theta_pool` opaquely through `train_mstep_mra`/`log_em_pool_diagnostics`, both of which now tolerate `None`. |
| `model.py` | `ConditionalDiT` (image branch, ported from `image_2d/model.py`), `PoseHead` (SO(2) branch, new to this codebase), `ConditionalVelocityCryoEM` (joint wrapper — owns the `y`-broadcast and `t_frac -> t_int` conversion). `IMAGE_SIZE`, `INTEGRATION_SCALE`. **3D:** `VOL_SIZE`, `ConditionalUNet3D` (wraps `diffusers.UNet3DConditionModel`, the only working 3D backbone in this repo — a DiT3D analogue is dead code in `simple_3d/model.py`), `PoseHead3D` (SO(3) pose-search head — Conv3d trunk + MLP taking the RAW `(B,6)` pose state, no matrix conversion needed since the 6D representation has no double-cover sign ambiguity to disambiguate, unlike quaternions), `ConditionalVelocityCryoET3D` (joint wrapper, `arch="unet3d"` only). **MRA:** `ConditionalVelocityMRA` (joint wrapper, reuses `ConditionalDiT`/`ConditionalUNet2D`/`PoseHead` completely unchanged — `y` is already a native `(B,1,H,W)` image here, so `forward` skips the `y`-broadcast step `ConditionalVelocityCryoEM.forward` does). `ConditionalVelocityMRA` takes `use_pose_head: bool = True`; when `False`, `self.pose_branch = None` and `forward` returns `(v_x, None)`, never touching `theta_t` — this is THE single source of truth every pose-optional function in `ode.py`/`scsi.py`/`wandb_logging.py` reads via `getattr(model, "pose_branch", None) is not None`. `ConditionalVelocityCryoEM`/`ConditionalVelocityCryoET3D` have no such flag — their pose branches are unconditional. |
| `corruption.py` | Forward channel `F`: `sample_uniform_angle` (Haar-uniform SO(2), literal `z~N(0,1)` direction), `sample_tilt_series_angles` (T evenly-spaced angles from a random per-acquisition start offset — the tilt-series generalization of `sample_uniform_angle`), `rotate_2d` (black-fill in-plane rotation), `project_1d`, `forward_channel` (rotate → project → AWGN). **3D:** `sample_uniform_rotation_so3` (Haar-uniform SO(3) via `scipy.spatial.transform.Rotation`), `sample_tilt_series_rotations_so3` (mount+fixed-axis tilt series — see "3D/CryoET generalization" below for the composition-order rule), `rotate_3d` (same black-fill shift-trick as `rotate_2d`), `project_2d`, `forward_channel_3d` (public pose type is `(B,6)`, converted to a matrix only internally). **MRA:** `mask_to_disk` (masks GT to its inscribed disk — a precondition for `forward_channel_mra`, see the gotcha below), `forward_channel_mra` (`rotate_2d` reused verbatim, then full-image AWGN directly — no `project_1d`, so `y` stays `(B,1,H,W)`). |
| `data.py` | `load_mnist_subset` (takes `train: bool = True` — `pretrain*.py` always passes `train=True` (MNIST train split), `main*.py` always passes `train=False` (MNIST test split), keeping the pretraining pool and the EM dataset disjoint by construction; see the dataset-split gotcha below), `build_observations` (applies `corruption.forward_channel` `corruptions_per_object` times per object to build the observation set `mu`; each application is one tilt-series ACQUISITION of `n_tilts` projections via `corruption.sample_tilt_series_angles`; keeps `theta_star` as a diagnostic-only ground truth, never fed to the model; also returns `acq_idx` — which acquisition each observation belongs to, contiguous and in tilt order, so `y_obs[acq_idx == a]` is one full tilt series — used only for grouped visualization in `wandb_logging.py::log_reconstruction_grid`). **3D:** `load_mnist_volumes_3d` (extrudes a shrunk-in-plane MNIST digit into a `p^3` volume with margin on ALL THREE axes — see gotchas below for why depth-only margin isn't enough), `build_observations_3d` (mirrors `build_observations`; its diagnostic-only ground truth `R_star` is kept as `(N,3,3)` MATRICES, not the 6D representation, since it's never fed to the model). **MRA:** `load_mnist_subset_mra` (`load_mnist_subset` + `corruption.mask_to_disk`, see the gotcha below), `build_observations_mra` (simpler than `build_observations` — every observation is an independent `corruption.forward_channel_mra` draw, no tilt-series/`acq_idx` structure, so it returns a 3-tuple `(y_obs, theta_star, image_idx)` not a 4-tuple). |
| `wandb_logging.py` | `log_train_step` (per-SGD-step scalars, extracted from `train_mstep`'s inner loop), `log_reconstruction_grid` (ONE 4-row panel where each COLUMN is a different acquisition/"problem" — column 0 is `fixed_acq_id`, held constant the whole run, the rest freshly randomized each EM step; per column: GT digit / that acquisition's real tilts stacked into a (T,W) sinogram / one generated x_hat conditioned on the middle tilt / that same x_hat reprojected at every tilt's TRUE angle, stacked into a second sinogram directly comparable to the first — runs its own small `ode.sample_joint` call rather than reusing `scsi.py::propose_estep`, since `scsi.py` already imports this module and importing back would be circular), `log_em_pool_diagnostics` (scalar-only mean circular-error health check over the whole training pool), `log_overfit_step` (per-step loss/loss_image/loss_pose/grad_norm for `overfit.py`, no images). Degrades gracefully without wandb (`_WANDB_AVAILABLE`), same pattern as everywhere else in the repo. **3D:** `log_pretrain_reconstruction_3d`, `log_em_pool_diagnostics_3d` (`si.rotation_geodesic_angle` replaces `wrap_to_pi`-based error), `log_reconstruction_grid_3d` (each observation is a full `(H,W)` image here, not a 1D signal, so the sinogram-stacking trick is replaced by `_projection_mosaic` — xy\|xz\|yz sum-projection mosaics — for GT/generated volumes, and horizontal strips of a few preview tilts for observations/recorruptions). Both `log_pretrain_reconstruction_3d` and `log_reconstruction_grid_3d` ALSO log an interactive companion, reusing the SAME `x_hat` the static mosaic panel already computed (no extra `sample_joint_3d` call): `_volume_to_pointcloud` converts a `(D,H,W)` volume into an `(N,6)` `[x y z r g b]` point cloud (per-volume top-`keep_frac` quantile threshold, NOT a fixed cutoff — `x_hat` has no guaranteed range, and an untrained model legitimately renders as noise under this scheme, that's the diagnostic not a bug), logged as a list of `wandb.Object3D` under one key (`pretrain/generations_3d`, `em/generations_3d`) so wandb's media panel renders a steppable, individually-orbitable 3D gallery. GT volumes are deliberately NOT logged this way (the static mosaic already carries that comparison; GT never changes call to call, so repeating the upload every `--plot_every`/EM-step would be pure waste). **MRA:** `log_reconstruction_grid_mra` — simpler than `log_reconstruction_grid`, since there's no tilt series to stack into a sinogram: one column per observation (not per acquisition), every row already a plain `(H,W)` image. `log_pretrain_reconstruction_mra` — MRA analogue of `log_pretrain_reconstruction` for `pretrain_mra_rotation.py`, same no-strip simplification; when `model.pose_branch is not None` it recorrupts at the RECOVERED angle theta_hat (not the true one), matching `log_pretrain_reconstruction`'s self-consistency convention since `pretrain_mra_rotation.py` has no separate pool-level pose diagnostic to lean on. When pose-free (`theta_hat is None`, `pretrain_mra_rotation.py --no_pose_head`): row 3 falls back to recorrupting at `theta_true` instead (same convention `log_reconstruction_grid_mra` already uses), the row label changes to `"F(x_hat; true θ)"`, and `circ_err`/`pretrain/circular_error` are skipped entirely rather than computed against `None`. `log_em_pool_diagnostics` is reused unmodified (it only ever touches `theta_pool`/`theta_star`, zero image tensors) except for one added guard: it no-ops when `theta_pool is None` (pose-free model) rather than crashing on `wrap_to_pi(None - theta_star[...])`. `log_reconstruction_grid_mra`'s column titles similarly drop "rec θ" when `theta_hat is None` — its recorrupt row is unaffected either way, since it already recorrupts at `theta_star` (true angle), never `theta_hat`. |
| `main.py` | CLI (`argparse`), device autodetect, `--debug` tiny-run defaults, dataset load, model/optimizer construction, optional `--init_ckpt` load, `wandb.init`/`finish`, one call to `em.run_em_loop(...)`. Also owns `--overfit` dispatch (see `overfit.py`). |
| `overfit.py` | **`--overfit` sanity check, not part of the SCSI algorithm.** `overfit_single_batch(model, x_batch, theta_batch, y_batch, ..., loss_fn=loss_func_joint)` — draws one fixed `(x_hat, theta_hat, y)` batch and runs many SGD steps of `loss_fn` against it with a throwaway optimizer, logging every step via `wandb_logging.log_overfit_step`. Only the interpolant's own randomness (`t`, `z_image`/`z_vol`, `theta_z`/`pose_z`) varies step to step, so the loss floor is the irreducible conditional variance, not zero — see the module docstring before reading a plateau as a bug. Dispatched from `main.py`/`main_3d.py` right after model construction; exits before the EM loop is ever called. The `loss_fn` parameter is the ONE place this file was extended in place rather than getting a `_3d` copy — `main_3d.py` passes `loss_fn=scsi.loss_func_joint_3d`; every existing 2D caller is unaffected since it's a keyword default. |
| `pretrain.py` | **Supervised warm-start for `Theta^(0)`.** Flat stochastic-interpolant SGD loop (no outer/inner loop) directly on `scsi.py::loss_func_joint`. Two mutually exclusive target modes via `--warmstart_target`: `gt` (default) — draws a fresh random rotation `R` per step, keeps the GT digit canonical, computes `y = corruption.forward_channel(x_i, theta=R)`, exactly as before. `classical_recon` — `build_classical_recon_pool` builds a FIXED, finite pool once up front: `data.build_observations` draws `--corruptions_per_object` tilt-series acquisitions of `--n_tilts` observations per GT object (SAME flag names/defaults as `main.py`'s own EM-pool geometry, so this can be sized to match it exactly — but a much smaller `--corruptions_per_object` default, since each acquisition costs one `classical_recon.backproject` call at pool-build time), backprojects each acquisition into ONE reconstruction `x_hat` (filtered FBP at the TRUE angles — a legitimate oracle since this is synthetic data), then globally rescales the WHOLE reconstruction pool to the GT pool's mean/std and clamps to `[-1,1]` (`--recon_calibration`; raw `backproject()` output is not on MNIST's `[-1,1]` scale — see the calibration gotcha below). Every observation in an acquisition is paired with that acquisition's OWN `x_hat` and its OWN real (never resynthesized) observation `y` — `x1 = x_hat`, conditioning `y` = the real observation, unlike `train_mstep`'s fresh-`F(x_pool)` resynthesis pattern. Pluggable GT-pool selection via `SELECTION_STRATEGIES` (currently one entry, `per_class`, drawing `--n_pretrain_images_per_class` images from each of `--digit_classes`) is orthogonal to `--warmstart_target` — it picks which GT objects exist, not what the interpolant target is. Saves a `model.state_dict()` checkpoint that `main.py --init_ckpt` can load directly. |
| `pretrain_3d.py` | **NEW FILE, not an extension of `pretrain.py`.** 3D/CryoET analogue — same flat-SGD structure, `select_per_class` calls `data.load_mnist_volumes_3d`, trains `scsi.loss_func_joint_3d`, saves to `mnist_cryoet_checkpoints/pretrain_theta0.pt` (separate namespace from the 2D pipeline's `mnist_cryoem_checkpoints/`). Same `--warmstart_target classical_recon` mode as `pretrain.py`, via `build_classical_recon_pool_3d` (`data.build_observations_3d` + `classical_recon_3d.backproject`, plus `--tilt_axis`, matching `main_3d.py`'s flag) — one representation-specific wrinkle: `build_observations_3d` returns rotations as `(N,3,3)` matrices (what `backproject` needs), converted to the `(N,6)` Gram-Schmidt pose only at the very end, once, for `loss_func_joint_3d`'s `pose_hat`. |
| `main_3d.py` | **NEW FILE, not an extension of `main.py`.** 3D/CryoET analogue — same CLI/dispatch structure, loads via `data.load_mnist_volumes_3d`/`build_observations_3d`, builds `ConditionalVelocityCryoET3D`, runs `em.run_em_loop_3d`. Defaults differ from `main.py` for compute-cost reasons: `--batch_size 8` (not 256), `--sample_steps 20` (not 50) — 3D UNet passes are much heavier than the 2D DiT/UNet2D. |
| `main_mra_rotation.py` | **NEW FILE, not an extension of `main.py`.** Rotational-MRA analogue — same CLI/dispatch structure, but loads via `data.load_mnist_subset_mra`/`build_observations_mra`, builds a `ConditionalVelocityMRA`, runs `em.run_em_loop_mra`. No `--n_tilts`/`--tilt_increment_deg` (no tilt series — `--corruptions_per_object` directly sets `N_obs`, default `500`); `--noise_std` defaults to `0.3` (full-image AWGN scale), not `main.py`'s `3.0` (projected-signal scale). `--overfit` needs no `loss_fn=` override (the default `loss_func_joint` is already correct, since the pose branch is still plain SO(2)). Checkpoints/eval images go to `mnist_mra_rotation_checkpoints/`/`mnist_mra_rotation_eval/`. `--no_pose_head` (`action="store_true"`) passes `use_pose_head=not args.no_pose_head` to `ConditionalVelocityMRA` — the ONLY code this file needed; `--overfit` dispatch and the `run_em_loop_mra` call are byte-for-byte unchanged, since they already thread `theta_batch`/`theta_pool` opaquely through functions that now tolerate `None`. See "Rotational MRA generalization" below. |
| `pretrain_mra_rotation.py` | **NEW FILE, not an extension of `pretrain.py`.** Rotational-MRA analogue — same flat-SGD structure directly on `scsi.py::loss_func_joint` (reused unmodified), `select_per_class` calls `data.load_mnist_subset_mra` (disk-masked pool — required, see the MRA gotcha below), draws a fresh random rotation per step via `corruption.forward_channel_mra`, logs qualitative panels via `wandb_logging.log_pretrain_reconstruction_mra`, saves to `mnist_mra_rotation_checkpoints/pretrain_theta0.pt` (same namespace `main_mra_rotation.py` writes to, so `--init_ckpt` chains directly). `--noise_std` defaults to `0.3`, matching `main_mra_rotation.py` (not `pretrain.py`'s `1.0`). `--no_pose_head` mirrors `main_mra_rotation.py`'s flag of the same name (`use_pose_head=not args.no_pose_head` into `ConditionalVelocityMRA`) — the training loop itself needed no other changes, only `wandb_logging.log_pretrain_reconstruction_mra` needed a `theta_hat is None` guard (see its row above). A `--no_pose_head` pretrain checkpoint's state_dict has no `pose_branch.*` keys, so `main_mra_rotation.py --init_ckpt` must ALSO be run with `--no_pose_head` to load it (mismatched flags fail loudly on `load_state_dict`'s `strict=True`). |
| `test_so3_math.py` | **NEW FILE, validation gate for the 3D generalization — run to green BEFORE trusting `pretrain_3d.py`/`main_3d.py` output.** Plain-assert script (`uv run python test_so3_math.py`), pytest-compatible. Checks: Gram-Schmidt output is always a proper rotation; the 6D<->matrix round-trip is exact; the tilt-series composition order shares one physical axis per acquisition (see gotcha below — this is the single highest-risk correctness surface in the 3D generalization); rotating a volume doesn't change its projected mass (catches clipping/margin bugs); the pose-branch interpolant/integrator pairing is exact for the `linear` schedule. |
| `classical_recon.py` / `classical_recon_3d.py` | **NEW FILES, originally standalone diagnostics — don't touch `si.py`/`scsi.py`/`em.py`/`model.py`, but their `backproject()` function is now ALSO consumed by `pretrain.py`/`pretrain_3d.py`'s `--warmstart_target classical_recon` (lazy-imported, so the default `--warmstart_target gt` path never pays for `skimage`/`matplotlib`).** So "not wired into the SCSI algorithm" no longer describes this pair of files as a whole — only `reconstruct_examples`/`compare`/`compare_mosaic`/the CLI/plotting remain diagnostic-only; `backproject` itself is a real dependency of pretraining now. Filtered backprojection (2D) / weighted backprojection (3D): the classical, non-learned tomographic inversion of the CryoEM/CryoET channel, evaluated as a feasibility check for a possible classical-recon pretraining scheme (an alternative to `pretrain.py`/`pretrain_3d.py`'s GT-supervised `x_hat`, where the pseudo-target would come from inverting a real tilt series instead of a known MNIST label) — a feasibility check since acted upon: see `--warmstart_target classical_recon` above. Both re-derive `corruption.py`'s exact rotate+project geometry as an adjoint operator (ramp-filter each projection — 1D Ram-Lak / 2D radial `\|k\|`, optionally Hann-tapered — smear it back across the summed axis, undo the KNOWN rotation, average) rather than reaching for `skimage.transform.iradon`, which has its own padding/rotation-direction/sinogram-width conventions unrelated to this codebase's channel. Each script sweeps `--n_views` and both `--mode {random,tilt_series}` (`random` = full-coverage upper bound; `tilt_series` = the SAME limited-angle acquisition geometry `main.py`/`main_3d.py`'s EM pool actually produces, real missing-wedge artifacts included), reports Pearson r + SSIM, and saves gallery/sweep PNGs to `classical_recon_eval/` (no wandb — always-local matplotlib output, unlike the rest of this codebase's `--no_wandb` convention). **Every reconstruction uses the TRUE (`theta_star`/`R_star`) pose — these are upper-bound numbers conditional on known pose, not evidence a full pretraining scheme (which would also need pose estimation) is viable; `--mode tilt_series` additionally reconstructs at an ARBITRARY absolute rotation (random per-acquisition start offset/mount), not the canonical frame `pretrain*.py` targets need — see both scripts' module docstrings. `build_classical_recon_pool`/`build_classical_recon_pool_3d` sidestep this SPECIFIC caveat by calling `backproject` directly with `data.build_observations`'s ABSOLUTE `theta_star`/`R_star` (not a tilt-series-relative offset), never going through `reconstruct_examples`'s `--mode tilt_series` gallery path — so pretraining's reconstructions land in the canonical frame correctly; the "raw output isn't on `[-1,1]` scale" caveat two sentences below still applies to them, and is handled by `--recon_calibration`, see the gotcha below.** Empirically (`--debug` aside): filtered recon clearly recovers digit identity by `n_views≈64` at each script's channel-realistic `--noise_std` default (`3.0` 2D / `0.5` 3D, matching `main.py`/`main_3d.py`); the 3D script also surfaces a real (not a bug) depth-axis softening — WBP with a finite view count blurs the GT's sharp extrusion-depth edges into a gradual ramp — which tanks a voxelwise SSIM while the xy\|xz\|yz projection mosaic (what the gallery panel actually shows) still looks sharp; `compare_mosaic` reports that second, more representative number alongside the strict one specifically so this doesn't misread as failure. DC-offset handling (both channels' background is `-1`, not `0`, so every projection carries a constant `-image_size`/`-vol_size` pedestal) and why backprojection uses a fresh zero-padded rotation helper instead of `corruption.rotate_2d`/`rotate_3d` directly (their internal shift-to-`-1` trick would double-shift already-shifted projections) are both documented in `classical_recon.py`'s module docstring — `classical_recon_3d.py` reuses that reasoning by reference rather than repeating it. |

| `tilt_series_difficulty.py` / `tilt_series_difficulty_3d.py` | **NEW FILES, standalone diagnostics building on `classical_recon.py`/`classical_recon_3d.py` — import `backproject`/`compare` from them rather than duplicating.** Answers a narrower follow-up question than raw reconstruction fidelity: which tilt-series geometries (n_tilts, angular span) are "poor but not hopeless" as a seed for a possible classical-recon pretraining scheme. Three measurements per script: (A) a span x n_tilts quality grid (`tilt_increment_deg = span/n_tilts` per cell) that separates missing-wedge failure (flat rows — span-limited, more tilts don't help) from aliasing (flat columns — density-limited); (B) a multi-acquisition averaging test — reconstructs the SAME digit from K independent random-offset acquisitions and checks whether averaging the K reconstructions recovers most of the per-acquisition quality loss (large gap = acquisition-specific VARIANCE, since each acquisition's missing wedge sits at a different random orientation by construction of `sample_tilt_series_angles`/`sample_tilt_series_rotations_so3` — exactly what SCSI's cross-acquisition pooling is built to exploit; small gap = shared BIAS pooling can't fix); (C) a class-identity nearest-neighbor check — does a reconstruction still best-match its own digit class (by Pearson r) among a labeled pool, independent of raw pixel fidelity. The 3D script runs at deliberately reduced scope (fewer classes, smaller `--vol_size`, coarser grid) since a single-axis SO(3) tilt series leaves a genuinely worse missing region (a whole cone, not a wedge) and 3D reconstructions are heavier — it exists to check whether the 2D conclusion transfers, not to assume it does. **Empirical finding (both channels, `uv run python tilt_series_difficulty.py` / `_3d.py` with defaults): even down to very narrow spans (2°-30°, per-acquisition Pearson r≈0.32-0.50, visually just a directional streak with no recognizable digit) the degradation stays strongly variance-dominated (averaging gap +0.3 to +0.4) and class-identity accuracy stays well above chance (50-100% vs 10-20%) — no sharp cliff into a "hopeless/bias-dominated" regime was found in the tested range; span is the dominant driver of quality, n_tilts/density matters much less at narrow spans.** Same upper-bound-via-known-pose caveat as `classical_recon.py`/`classical_recon_3d.py`; test (B) is additionally an optimistic PROXY — it averages reconstructions of the SAME digit across acquisitions, which real pretraining never gets to do (each SGD step sees exactly one acquisition's reconstruction as its target) — a large averaging gap is evidence the per-acquisition error is non-systematic, a necessary but not sufficient condition for a network trained on many such noisy examples to plausibly learn past it. |

## How to run

```bash
uv run python main.py --debug --no_wandb    # ~seconds smoke test, no wandb needed
uv run python main.py                       # full run, default hyperparameters
uv run python main.py --steps_per_em 1      # literal pseudocode: fresh Phi^(k-1) draw every SGD step
uv run python main.py --overfit --no_wandb  # sanity check only: overfit one fixed batch, exit (no EM loop)

uv run python pretrain.py --debug --no_wandb                     # ~seconds smoke test
uv run python pretrain.py --digit_classes 3 7 --n_pretrain_images_per_class 4
uv run python main.py --init_ckpt mnist_cryoem_checkpoints/pretrain_theta0.pt   # warm-started EM

# Classical-recon warm start (x1 = a classical reconstruction, not the literal GT digit)
uv run python pretrain.py --warmstart_target classical_recon --debug --no_wandb
uv run python pretrain.py --warmstart_target classical_recon \
    --n_tilts 16 --tilt_increment_deg 7.5 --noise_std 3.0   # matches main.py's real EM geometry
uv run python pretrain_3d.py --warmstart_target classical_recon --debug --no_wandb
uv run python pretrain_3d.py --warmstart_target classical_recon \
    --n_tilts 16 --tilt_increment_deg 7.5 --noise_std 0.5   # matches main_3d.py's real EM geometry

# Classical (non-learned) reconstruction feasibility check -- see file map above
uv run python classical_recon.py --debug                         # ~seconds smoke test
uv run python classical_recon.py --digit_classes 0 3 8 --n_images_per_class 1
uv run python classical_recon.py --mode tilt_series               # realistic missing-wedge geometry
uv run python classical_recon_3d.py --debug                       # ~seconds smoke test
uv run python classical_recon_3d.py --digit_classes 0 3 8 --n_images_per_class 1

# Which tilt-series geometries are "poor but not hopeless" for that pretraining scheme
uv run python tilt_series_difficulty.py --debug                   # ~seconds smoke test
uv run python tilt_series_difficulty.py                           # full span x n_tilts grid
uv run python tilt_series_difficulty_3d.py --debug                # ~seconds smoke test
uv run python tilt_series_difficulty_3d.py                        # reduced-scope 3D grid
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
They're also now provably **disjoint** regardless of `--digit_classes` overlap: `pretrain.py`
always draws from MNIST's 60k-image TRAIN split, `main.py` from its 10k-image TEST split (see
`data.py`'s `train` parameter and the dataset-split gotcha below). This does not make either
draw reproducible — both remain unseeded shuffles.

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

## Rotational MRA generalization

```bash
uv run python main_mra_rotation.py --debug --no_wandb    # ~seconds smoke test, no wandb needed
uv run python main_mra_rotation.py                       # full run, default hyperparameters
uv run python main_mra_rotation.py --overfit --no_wandb  # sanity check only: overfit one fixed batch, exit (no EM loop)

uv run python pretrain_mra_rotation.py --debug --no_wandb                     # ~seconds smoke test
uv run python pretrain_mra_rotation.py --digit_classes 3 7 --n_pretrain_images_per_class 4
uv run python main_mra_rotation.py --init_ckpt mnist_mra_rotation_checkpoints/pretrain_theta0.pt  # warm-started EM

uv run python main_mra_rotation.py --debug --no_wandb --no_pose_head              # pose-free ablation
uv run python main_mra_rotation.py --debug --no_wandb --no_pose_head --overfit   # sanity check only

uv run python pretrain_mra_rotation.py --debug --no_wandb --no_pose_head         # pose-free Theta^(0)
uv run python main_mra_rotation.py --no_pose_head \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_theta0.pt    # MUST match --no_pose_head
```

A third corruption channel alongside the 2D CryoEM (rotate → project → AWGN) and 3D CryoET
(rotate → project → AWGN) channels: classical **rotational multi-reference alignment**,
`F(x) = R∘x + W` — the SAME in-plane SO(2) rotation as the CryoEM channel
(`corruption.rotate_2d`/`sample_uniform_angle` reused verbatim), but with **no projection step**
— `y` stays a full `(B,1,H,W)` image, corrupted with full-image AWGN directly. The pose branch is
still plain SO(2) (`si.pose_interpolant`, `ode.sample_joint`), so no new interpolant/integrator
math was needed at all — only the pieces that either hardcode `corruption.forward_channel` or
assume `y` needs a 1D→2D broadcast got `_mra` copies: `corruption.forward_channel_mra`,
`model.ConditionalVelocityMRA`, `data.build_observations_mra`, `scsi.train_mstep_mra`,
`em.run_em_loop_mra`, `wandb_logging.log_reconstruction_grid_mra`. Everything else —
`ode.sample_joint`, `scsi.propose_estep`, `scsi.loss_func_joint`, `scsi.sample_pool_indices`,
`wandb_logging.log_em_pool_diagnostics`, `overfit.overfit_single_batch` — is reused completely
unmodified, since none of them ever hardcode a channel call or inspect `y`'s trailing shape.

There is no tilt-series structure in this channel (classical MRA has no shared-axis/mount
concept), so `main_mra_rotation.py` has no `--n_tilts`/`--tilt_increment_deg` flags —
`--corruptions_per_object` directly sets `N_obs = n_images * corruptions_per_object`, unlike
`main.py`/`main_3d.py` where it's additionally multiplied by `--n_tilts`. `--noise_std` defaults
to `0.3` (full-image, `[-1,1]`-scale AWGN std, matching `image_2d/main.py`'s own AWGN/MRA
default) — NOT `main.py`'s `noise_std=3.0`, which is calibrated for the 1D-projected channel's
much larger summed numeric range.

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
- **`pretrain.py`/`pretrain_3d.py --warmstart_target classical_recon`: raw `classical_recon(_3d).backproject()` output is NOT on MNIST's `[-1,1]` scale — measured empirically (not assumed) before this mode was built: at `main.py`'s own default tilt-series geometry (`n_tilts=16`, 120° missing-wedge span), filtered (hann) backprojection's pool std sits at ~0.4-0.7x the GT pool's std, with per-image min/max routinely overshooting to roughly `[-2.0, +0.9]` instead of `[-1, 1]`; unfiltered backprojection is far worse (mean in the single digits, not near 0). Fed straight into `si.interpolant(z, x1, t)` as `x1`, this would badly miscalibrate the learned velocity field's output magnitude, and would be inconsistent with `main.py --init_ckpt`'s downstream assumption that `x_pool` lives in `[-1,1]` with background `-1`. Fix (`--recon_calibration affine_clamp`, the default): ONE affine map, fit ONCE from the WHOLE reconstruction pool's AGGREGATE mean/std to the WHOLE GT pool's AGGREGATE mean/std (`a = std(GT)/std(recon)`, `b = mean(GT) - a*mean(recon)`), applied to every `x_hat`, then hard-clamped to `[-1,1]`. Deliberately POOL-level, not per-acquisition against that acquisition's own source image's individual mean/std: a per-acquisition fit would hide the reconstruction's actual error (each x_hat would be individually re-centered onto its own GT) instead of correcting only the aggregate scale. This still reads `x_gt`'s two aggregate numbers (not zero GT information) — porting this to real, non-synthetic data would need to replace THOSE TWO NUMBERS (e.g. a known prior on the target distribution's scale), not assume the calibration step disappears entirely. Verified visually (not just via the printed mean/std) that this produces digit-identifiable targets even at `n_tilts=16`, matching `tilt_series_difficulty.py`'s prior finding that "poor" tilt geometries stay variance-dominated, not bias-dominated. `--recon_calibration none` is an escape hatch to inspect raw output; expected to train poorly.
- **`--n_images_per_class`/`--n_pretrain_images_per_class` are PER CLASS, not totals**, and
  `--digit_classes` defaults to all 10 MNIST classes on both `main.py` and `pretrain.py`. A
  bare `uv run python main.py` therefore loads `2 * 10 = 20` GT digits by default, not 2 —
  narrow with `--digit_classes 3` (etc.) to get a single-class run.
- **`pretrain*.py` and `main*.py` draw from disjoint, hardcoded MNIST splits — not a CLI
  flag.** All three `data.py` loaders (`load_mnist_subset`, `load_mnist_subset_mra`,
  `load_mnist_volumes_3d`) take `train: bool = True`; the three `pretrain*.py` scripts'
  `select_per_class` always pass `train=True` (60k-image train split), the three `main*.py`
  scripts always pass `train=False` (10k-image test split, both the `--overfit` branch and
  the EM branch). WHY: previously both pools drew unseeded random shuffles from the same
  hardcoded `train=True` split, so overlap between the pretraining pool and the EM dataset was
  possible and unrecorded even with non-overlapping `--digit_classes`. WHAT changed: the split
  is a hardcoded architectural constant per script family, deliberately not exposed as a
  `--split` flag — a flag would let a caller reopen the exact hole this closes; revisit only
  if `--n_images_per_class` starts approaching the test split's smallest class (~892, digit
  5). CAVEAT: the test split is much smaller per class (892–1135) than train (5421–6742);
  `load_mnist_subset`'s internal `DataLoader(..., batch_size=n_images_per_class,
  shuffle=True)` has `drop_last=False`, so requesting more than a class's count returns a
  **silent short batch**, not an error — current defaults (2, up to 16 in `--debug`) are far
  below either cap. This does NOT make the draws reproducible (`seed` remains unused by every
  caller — it's threaded through every loader and `select_per_class` but no call site ever
  passes it). Scope: this guarantee is local to `mnist_cryoem/`; `image_2d/` and `toy_2d_pc/`
  load MNIST independently via their own `data.py` and are unaffected.
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
- **MRA: GT digits MUST be disk-masked (`corruption.mask_to_disk`, applied once by
  `data.load_mnist_subset_mra`) before entering `forward_channel_mra` — otherwise `rotate_2d`'s
  square-canvas corner geometry leaks a second, content-independent cue to θ straight into `y`.**
  `rotate_2d` rotates a SQUARE canvas inside a SQUARE `F.grid_sample` sampling grid; rotation
  preserves radius, but the square boundary itself isn't circularly symmetric (radius 1 at the
  edge midpoints, √2 at the corners), so whether an output pixel's rotated source coordinate
  falls inside vs. outside the square (triggering zero-padding, exactly `-1` after the shifted-
  frame trick `rotate_2d` already uses) is a deterministic function of `θ mod 90°`, independent
  of image content. The 2D CryoEM/3D CryoET channels never see this: `project_1d`/`project_2d`
  sum it away into a smooth per-column/per-pixel bias. `forward_channel_mra` has no projection
  step, so without masking this artifact would reach `y` directly and give the pose branch a
  free, geometry-only signal unrelated to the actual rotated digit — silently, no shape error,
  same "easy to get backwards" flavor as the canonical-target gotcha above. Masking GT to its
  inscribed disk (radius 1, same normalized coords as `rotate_2d`'s `affine_grid`) fixes this:
  every rotation of a disk-masked image keeps all content within that same disk, so the annulus
  between radius 1 and √2 is uniformly exactly `-1` in the source for every θ, matching
  grid_sample's zero-pad value exactly — no leak.
- **MRA: `em/circular_error` (reused unmodified from the CryoEM pipeline's
  `log_em_pool_diagnostics`) is gauge-ambiguous, same as in the 2D CryoEM/3D CryoET pipelines** —
  the learned `(image, pose)` prior is only identifiable up to a *global* rotation (rotate the
  whole learned prior by φ, shift every recovered angle by −φ, and `y = F(x;θ)` is reproduced
  exactly). A plateau at a nonzero, arbitrary constant does not by itself mean training failed —
  check `loss_image`/the reconstruction panel's visual quality instead.
- **MRA gitignored ephemera** (separate namespace, so it can be run from this directory without
  colliding with the 2D/3D pipelines): `mnist_mra_rotation_checkpoints/`,
  `mnist_mra_rotation_eval/` — same `*.pt`/`*.png` gitignore patterns as the other pipelines'
  ephemera dirs already cover these.
- **MRA: the pose/latent branch `R` is OPTIONAL, via `--no_pose_head`
  (`ConditionalVelocityMRA(use_pose_head=False)`) — implemented as `None`-propagation through
  the EXISTING functions, not a parallel `_nopose` copy of `ode.py`/`scsi.py`.** The general
  pattern, for the next latent branch someone adds: `forward_channel : X × R → Y` with `R`
  optional means `ode.sample_joint`, `scsi.propose_estep`, `scsi.loss_func_joint`, and
  `scsi.train_mstep_mra` all read ONE single source of truth —
  `getattr(model, "pose_branch", None) is not None` — never a caller-supplied value's
  `is not None`-ness (a real `theta_hat`/`theta_batch` can still legitimately flow into a
  pose-free model, e.g. `main_mra_rotation.py --overfit --no_pose_head` still draws a real
  angle to render `y`; the model just ignores it and returns `v_theta=None`, which is what
  `loss_func_joint` actually keys off for the loss). When `--no_pose_head` is set:
  `train_mstep_mra`'s `forward_channel_mra(x_batch, theta=theta_batch)` call is UNCHANGED
  syntactically, but `theta_batch=None` now falls into its fresh-Haar-draw branch instead of
  re-applying a recovered rotation — the M-step MARGINALIZES over `R` instead of estimating
  it, a real semantic shift hiding behind a line that looks untouched. Without a pose branch
  pinning an orientation, nothing forces `x_hat` into one particular canonical gauge — expect
  blurrier/rotationally-hedged samples relative to the pose-having model, especially early in
  training or at high `--noise_std`; that is the finding this ablation is FOR, not a bug (same
  spirit as `em/circular_error`'s gauge-ambiguity note above). `--pose_loss_weight` becomes
  inert under `--no_pose_head` (multiplies a constant-zero `loss_pose`).
  `pretrain_mra_rotation.py` has the matching `--no_pose_head` flag too (see its row above) —
  a checkpoint trained with it can ONLY be loaded by `main_mra_rotation.py --init_ckpt` if
  `--no_pose_head` is passed there as well (mismatched `pose_branch.*` keys fail loudly on
  `load_state_dict`'s `strict=True` otherwise). Left out of scope: `train_mstep` (CryoEM,
  `main.py`) is the same shape and could get the identical treatment, but wasn't requested.
