"""Lifted SCSI (Self-Consistent Stochastic Interpolant) for CryoET point clouds.

Recovers a generative prior over clean 3D point clouds from only their corrupted
CryoET tilt-series projections, via:

  1. GT generation   sample X; freeze observations y_obs = F(X)  (mu = the observation dist.).
  2. Warm-start       F-dagger(y_obs) -> x_boot; train b^(0) on (g . x_boot, y)  (Algorithm 1).
  3. SCSI EM loop     the **literal self-consistent** loop (Algorithm 2): the EMA transport map
                      is sampled on the fly inside each training step.

This is the literal pseudocode (not the pool-based EM): per outer iteration k, with the
EMA params FROZEN, each of the ``training_steps`` inner SGD steps draws z', runs the EMA
transport ODE to get x-hat = Phi_EMA^(k-1)(z'|y), forms y-hat = F(x-hat) (with alpha_z /
alpha_y coupling), and updates Theta on || b_t(I_t | y-hat) - dI_t ||^2. The EMA is then
updated once: Theta_EMA^(k) <- gamma Theta_EMA^(k-1) + (1-gamma) Theta^(k).

Clean ground-truth clouds are used ONLY to synthesize y_obs and for visualization; the
model never trains on them.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .corruption import forward_channel, pseudo_inverse
from .data import make_mixture_sampler, mixture_surface_residual, sample_perturbed_dataset
from .device import autocast, configure_backends, describe, needs_grad_scaler, synchronize
from .model import (
    ConditionalModelConfig,
    build_conditional_model,
    clone_ema,
    ema_update_outer,
)
from .si import interpolant, transport_sample
from .warmstart import find_initialization


# ── Checkpoints ────────────────────────────────────────────────────────────────


def save_checkpoint(path: str, model: nn.Module, cfg: ConditionalModelConfig) -> None:
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ConditionalModelConfig(**ckpt["cfg"])
    model = build_conditional_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ── Visualization ───────────────────────────────────────────────────────────────


def _set3d(ax, lim: float) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def log_em_step(
    gt_eval: torch.Tensor,       # (n, N, 3) clean clouds (reference only)
    y_eval: torch.Tensor,        # (n, K, P, P) fixed observations
    pi_eval: torch.Tensor,       # (n, N, 3) current prior samples pi(k)
    *,
    radius: float,
    noise_std: float,
    image_size: int,
    extent: float,
    em_step: int,
    tracker,
    out_dir: Path,
    global_step: list,
    viz_ball_radius: float,
    tag: str = "EM step",
    shapes: list[str] | None = None,
    coord_noise_std: float = 0.0,
    n_tilts: int = 11,
    tilt_step: float = 12.0,
    tilt_axis: str = "y",
    splat: str = "gaussian",
) -> None:
    """4-row panel: GT cloud | y_obs | pi(k) sample | F(pi(k)) consistency check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = gt_eval.size(0)
    with torch.no_grad():
        y_pi = forward_channel(
            pi_eval, radius=radius, noise_std=noise_std,
            image_size=image_size, extent=extent, coord_noise_std=coord_noise_std,
            n_tilts=n_tilts, tilt_step=tilt_step, tilt_axis=tilt_axis, splat=splat,
        )

    ch = y_eval.shape[1] // 2  # central tilt
    gt_np = gt_eval.cpu().numpy()
    pi_np = pi_eval.cpu().numpy()
    yobs_np = y_eval[:, ch].cpu().numpy()
    ypi_np = y_pi[:, ch].cpu().numpy()
    lim = max(1.6, extent * 0.8)

    fig = plt.figure(figsize=(2.6 * n, 10.4))
    row_titles = ["GT cloud", "obs y", f"pi({em_step})", "F(pi)"]
    for j in range(n):
        ax0 = fig.add_subplot(4, n, j + 1, projection="3d")
        ax0.scatter(gt_np[j, :, 0], gt_np[j, :, 1], gt_np[j, :, 2], s=2, alpha=0.6)
        _set3d(ax0, lim)
        ax1 = fig.add_subplot(4, n, n + j + 1)
        ax1.imshow(yobs_np[j], cmap="gray")
        ax1.axis("off")
        ax2 = fig.add_subplot(4, n, 2 * n + j + 1, projection="3d")
        ax2.scatter(pi_np[j, :, 0], pi_np[j, :, 1], pi_np[j, :, 2], s=2, alpha=0.6, color="C1")
        _set3d(ax2, lim)
        ax3 = fig.add_subplot(4, n, 3 * n + j + 1)
        ax3.imshow(ypi_np[j], cmap="gray")
        ax3.axis("off")
        if j == 0:
            for ax, title in zip((ax0, ax1, ax2, ax3), row_titles):
                ax.set_title(title, fontsize=9)

    fig.suptitle(f"{tag} {em_step}", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    slug = tag.lower().replace(" ", "_")
    path = out_dir / f"{slug}_{em_step:04d}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    resid = mixture_surface_residual(pi_eval, shapes or ["torus"]).item()
    print(f"  [eval] wrote {path}  pi surface_residual={resid:.4f} (0 = on a target surface)")

    if tracker is not None and tracker.enabled:
        tracker.log_image("eval/panel", str(path), step=global_step[0])
        tracker.log_clouds("eval/gt", gt_eval, step=global_step[0])
        tracker.log_clouds("eval/pi", pi_eval, step=global_step[0])
        if viz_ball_radius > 0:
            from .balls import save_balls_obj

            with tempfile.TemporaryDirectory(prefix="pcscsi_balls_") as tmp:
                paths = []
                for j in range(n):
                    p = os.path.join(tmp, f"pi_{em_step:04d}_{j}.obj")
                    save_balls_obj(pi_np[j], p, viz_ball_radius, 1)
                    paths.append(p)
                tracker.log_meshes("eval/pi_balls", paths, step=global_step[0])
        tracker.log({"eval/surface_residual": resid}, step=global_step[0])


def log_bootstrap(
    pi0: torch.Tensor,            # (n, N, 3) bootstrap reconstruction pi(0)
    tracker,
    out_dir: Path,
    global_step: list,
    viz_ball_radius: float,
    gt: torch.Tensor | None = None,
    shapes: list[str] | None = None,
) -> None:
    """Visualize the F-dagger bootstrap pi(0) and log it to disk + W&B."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = pi0.size(0)
    pi_np = pi0.cpu().numpy()
    gt_np = gt.cpu().numpy() if gt is not None else None
    lim = 1.8
    rows = 2 if gt_np is not None else 1

    fig = plt.figure(figsize=(2.8 * n, 2.8 * rows))
    for j in range(n):
        ax = fig.add_subplot(rows, n, j + 1, projection="3d")
        ax.scatter(pi_np[j, :, 0], pi_np[j, :, 1], pi_np[j, :, 2], s=2, alpha=0.6, color="C1")
        _set3d(ax, lim)
        if j == 0:
            ax.set_title("pi(0)  [F-dagger]", fontsize=9)
        if gt_np is not None:
            ax2 = fig.add_subplot(rows, n, n + j + 1, projection="3d")
            ax2.scatter(gt_np[j, :, 0], gt_np[j, :, 1], gt_np[j, :, 2], s=2, alpha=0.6)
            _set3d(ax2, lim)
            if j == 0:
                ax2.set_title("GT cloud", fontsize=9)

    fig.suptitle("bootstrap pi(0): F-dagger (tomo)", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "bootstrap_pi0.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    resid = mixture_surface_residual(pi0, shapes or ["torus"]).item()
    print(f"  [bootstrap] wrote {path}  pi(0) surface_residual={resid:.4f}")

    if tracker is not None and tracker.enabled:
        tracker.log_image("bootstrap/pi0_panel", str(path), step=global_step[0])
        tracker.log_clouds("bootstrap/pi0", pi0, step=global_step[0])
        tracker.log({"bootstrap/surface_residual": resid}, step=global_step[0])


# ── EM driver ─────────────────────────────────────────────────────────────────


def scsi_train(
    device: torch.device,
    cfg: ConditionalModelConfig,
    *,
    n_objects: int = 128,
    em_steps: int = 30,
    training_steps: int = 200,
    batch: int = 32,
    lr: float = 2e-4,
    radius: float = 0.08,
    noise_std: float = 0.1,
    extent: float = 2.0,
    sample_steps: int = 50,
    alpha_z: float = 0.0,
    alpha_y: float = 0.0,
    ema_decay: float = 0.999,
    pretrain_steps: int = 2000,
    style: str = "linear",
    shapes: list[str] | None = None,
    n_eval: int = 4,
    use_amp: bool = True,
    seed: int = 0,
    tracker=None,
    out: str = "toy_3d_pc_checkpoint.pt",
    eval_dir: str = "toy_3d_pc_eval",
    viz_ball_radius: float = 0.05,
    coord_noise_std: float = 0.0,
    n_tilts: int = 11,
    tilt_step: float = 12.0,
    tilt_axis: str = "y",
    splat: str = "gaussian",
    tomo_vol: int = 48,
    tomo_quantile: float = 0.15,
    dataset: str = "iid",
    dataset_eps: float = 0.0,
) -> nn.Module:
    torch.manual_seed(seed)
    configure_backends(device)
    shapes = shapes or ["torus"]
    sample_fn = make_mixture_sampler(shapes)
    print(f"[scsi] device={describe(device)}  amp={use_amp}  shapes={shapes}  splat={splat}")

    model = build_conditional_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scsi] parameters: {n_params:,}")

    fwd = dict(
        radius=radius, noise_std=noise_std, image_size=cfg.image_size, extent=extent,
        coord_noise_std=coord_noise_std, n_tilts=n_tilts, tilt_step=tilt_step,
        tilt_axis=tilt_axis, splat=splat,
    )

    # Phase 1: ground-truth clouds -> fixed observations y_obs = F(x_gt) (mu).
    if dataset == "template":
        gt = sample_perturbed_dataset(shapes, n_objects, cfg.n_points, dataset_eps, device=device)
        print(f"[scsi] dataset=template  eps={dataset_eps}")
    else:
        gt = sample_fn(n_objects, cfg.n_points, device=device)
    with torch.no_grad():
        y_obs = forward_channel(gt, **fwd)
    gt, y_obs = gt.cpu(), y_obs.cpu()
    print(f"[scsi] y_obs {tuple(y_obs.shape)}  range=[{y_obs.min():.2f}, {y_obs.max():.2f}]")

    out_dir = Path(eval_dir)
    ckpt_dir = Path("toy_3d_pc_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    global_step = [0]

    n_eval = min(n_eval, n_objects)
    gt_eval, y_eval = gt[:n_eval], y_obs[:n_eval]

    # Phase 2a: pseudo-inverse bootstrap  x_boot = F-dagger(y_obs).
    x_boot = pseudo_inverse(
        y_obs, cfg.n_points, tilt_step, tilt_axis,
        extent=extent, vol_size=tomo_vol, carve_quantile=tomo_quantile, seed=seed,
    )
    print(f"[scsi] F-dagger bootstrap  pi(0) {tuple(x_boot.shape)}")
    log_bootstrap(x_boot[:n_eval], tracker, out_dir, global_step, viz_ball_radius,
                  gt=gt_eval, shapes=shapes)

    # Phase 2b: warm-start the drift Theta^(0)  (Algorithm 1).
    if pretrain_steps > 0:
        print(f"[scsi] warm-start: {pretrain_steps} steps on g . F-dagger(y)")
        find_initialization(
            model, y_obs, x_boot, style=style, pretrain_steps=pretrain_steps,
            batch=batch, lr=lr, device=device, use_amp=use_amp,
            tracker=tracker, global_step=global_step,
        )
        save_checkpoint(str(ckpt_dir / "model_warmstart.pt"), model, cfg)

    # EMA init: Theta_EMA^(0) <- Theta^(0).
    model_ema = clone_ema(model)
    print(f"[scsi] EMA sampler enabled  gamma={ema_decay}")

    # Persistent optimizer for the inner SGD steps across all outer iterations.
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )
    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    Nobj = y_obs.size(0)

    # Phase 3: literal self-consistent EM loop (Algorithm 2).
    for k in range(1, em_steps + 1):
        print("=" * 60)
        print(f"EM iteration {k} / {em_steps}")
        print("=" * 60)
        model.train()
        running, n_batches = 0.0, 0
        for i in range(1, training_steps + 1):
            idx = torch.randint(Nobj, (batch,))
            y = y_obs[idx].to(device, non_blocking=(device.type == "cuda"))

            # x-hat = Phi_EMA^(k-1)(z' | y): on-the-fly transport with the FROZEN EMA map.
            z_prime = torch.randn(batch, cfg.n_points, 3, device=device)
            x_hat = transport_sample(model_ema, z_prime, y, n_steps=sample_steps)

            # alpha_z noise coupling: z = z' w.p. alpha_z, else fresh N(0, I).
            mask_z = (torch.rand(batch, 1, 1, device=device) < alpha_z)
            z = torch.where(mask_z, z_prime, torch.randn_like(z_prime))

            # y-hat = F(x-hat) (fresh global pose); alpha_y obs coupling: y-hat = y w.p. alpha_y.
            with torch.no_grad():
                y_hat = forward_channel(x_hat, **fwd)
            mask_y = (torch.rand(batch, 1, 1, 1, device=device) < alpha_y)
            y_hat = torch.where(mask_y, y, y_hat)

            with autocast(device, use_amp):
                t = torch.rand(batch, device=device)
                I_t, I_dot = interpolant(z, x_hat, t, style)
                pred = model(I_t, t, y_hat)
                loss = (pred - I_dot).pow(2).mean()

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            running += loss.item()
            n_batches += 1
            global_step[0] += 1
            if tracker is not None:
                tracker.log({"train/loss": loss.item()}, step=global_step[0])

        print(f"  inner loss={running / max(n_batches, 1):.5f}")

        # Theta^(k) <- Theta ; EMA over the outer loop.
        ema_update_outer(model_ema, model, ema_decay)
        save_checkpoint(str(ckpt_dir / f"model_em{k:04d}.pt"), model, cfg)
        save_checkpoint(str(ckpt_dir / f"model_em{k:04d}_ema.pt"), model_ema, cfg)

        # Eval: sample pi(k) with the EMA transport map and log.
        with torch.no_grad():
            z_eval = torch.randn(n_eval, cfg.n_points, 3, device=device)
            pi_eval = transport_sample(
                model_ema, z_eval, y_eval.to(device), n_steps=sample_steps
            ).cpu()
        synchronize(device)
        log_em_step(
            gt_eval, y_eval, pi_eval,
            radius=radius, noise_std=noise_std, image_size=cfg.image_size, extent=extent,
            em_step=k, tracker=tracker, out_dir=out_dir, global_step=global_step,
            viz_ball_radius=viz_ball_radius, shapes=shapes, coord_noise_std=coord_noise_std,
            n_tilts=n_tilts, tilt_step=tilt_step, tilt_axis=tilt_axis, splat=splat,
        )

    save_checkpoint(out, model, cfg)
    ema_out = out.replace(".pt", "_ema.pt") if out.endswith(".pt") else out + "_ema"
    save_checkpoint(ema_out, model_ema, cfg)
    print(f"[scsi] saved final checkpoints -> {out} , {ema_out}")
    return model_ema
