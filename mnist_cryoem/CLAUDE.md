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

    checkpoint + wandb_logging.log_em_step(...)                  # [em.py::run_em_loop]
```

`em.py::run_em_loop` is the `for k in 1..K` line; everything inside one iteration is
`scsi.py`.

## File map

| File | Role |
| --- | --- |
| `si.py` | **Stochastic interpolants only** — pure math, no model/optimizer coupling. `alpha/beta_{linear,gvp}` schedules, `interpolant(x0,x1,t,style)` (image branch), `wrap_to_pi`, `pose_interpolant(theta_z,theta_hat,t)` (SO(2) geodesic branch, always constant-angular-velocity regardless of `--interpolant_style`). |
| `ode.py` | **Φ, the E-step's integrator.** `sample_joint(model, z_image, y, n_steps, method)` — jointly integrates the image (Euclidean) and pose (SO(2), `wrap_to_pi` after each step) branches from noise to data. This is the seam for tuning the integrator: add a new `method` here without touching `scsi.py`/`si.py`. **Currently Euler-only** (raises on anything else) — matches every other experiment dir in this repo. A future Heun/RK4 needs its own geodesic-aware step for the pose branch; it can't just reuse a generic Euclidean implementation for both branches. |
| `scsi.py` | **The E-step and M-step**, named to match the pseudocode exactly. `sample_pool_indices` + `propose_estep` (E-step, calls `ode.sample_joint`); `loss_func_joint` + `train_mstep` (M-step, calls `corruption.forward_channel` then the loss, logs scalars via `wandb_logging.log_train_step`). |
| `em.py` | **The outer `for k in 1..K` loop.** `run_em_loop(...)` — one E-step then one M-step per `k` so `Phi^(k-1)`/`Theta^(k)` line up, checkpointing, calls `wandb_logging.log_em_step`. |
| `model.py` | `ConditionalDiT` (image branch, ported from `image_2d/model.py`), `PoseHead` (SO(2) branch, new to this codebase), `ConditionalVelocityCryoEM` (joint wrapper — owns the `y`-broadcast and `t_frac -> t_int` conversion). `IMAGE_SIZE`, `INTEGRATION_SCALE`. |
| `corruption.py` | Forward channel `F`: `sample_uniform_angle` (Haar-uniform SO(2), literal `z~N(0,1)` direction), `rotate_2d` (black-fill in-plane rotation), `project_1d`, `forward_channel` (rotate → project → AWGN). |
| `data.py` | `load_mnist_subset`, `build_observations` (applies `corruption.forward_channel` `corruptions_per_image` times per digit to build the observation set `mu`; keeps `theta_star` as a diagnostic-only ground truth, never fed to the model). |
| `wandb_logging.py` | `log_train_step` (per-SGD-step scalars, extracted from `train_mstep`'s inner loop), `log_em_step` (4-row reconstruction panel + mean circular-error diagnostic). Degrades gracefully without wandb (`_WANDB_AVAILABLE`), same pattern as everywhere else in the repo. |
| `main.py` | CLI (`argparse`), device autodetect, `--debug` tiny-run defaults, dataset load, model/optimizer construction, `wandb.init`/`finish`, one call to `em.run_em_loop(...)`. |

## How to run

```bash
uv run python main.py --debug --no_wandb    # ~seconds smoke test, no wandb needed
uv run python main.py                       # full run, default hyperparameters
uv run python main.py --steps_per_em 1      # literal pseudocode: fresh Phi^(k-1) draw every SGD step
```

Key flags (see `main.py::parse_args` for the full list): `--n_em_steps`(K)
`--steps_per_em`(T_tr) `--steps_first_em` `--sample_steps` (Euler steps for Φ)
`--interpolant_style {linear,gvp}` (image branch only) `--pose_loss_weight`
`--n_images` `--corruptions_per_image` `--noise_std` `--batch_size` `--lr` `--no_wandb`.

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
