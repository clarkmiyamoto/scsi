"""Lifted SCSI (Self-Consistent Stochastic Interpolant) for point clouds.

Recovers a generative prior over clean 3D point clouds from only their corrupted
2D projections, via EM -- the point-cloud analogue of ``toy_3d/scsi_train.py``.

Per EM iteration k:
  E-step  train_estep   train b_t on (x ~ pi(k), y_hat = F(x)) pairs:
                            I_t = (1-t) z + t x,  target = x - z,
                            loss = || b_t(I_t | y_hat) - (x - z) ||^2
  M-step  update_prior  sample pi(k+1) = Phi(z' | y_obs) with the conditional ODE.

Clean ground-truth clouds are used ONLY to synthesize the fixed observations
y_obs = F(x_gt) and for visualization; the model never trains on them.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .balls import save_balls_obj
from .corruption import forward_channel, random_so2, random_so3
from .data import make_mixture_sampler, mixture_surface_residual
from .device import autocast, configure_backends, describe, needs_grad_scaler, synchronize
from .model import ConditionalPointCloudVelocity
from .prior import BootstrapContext, make_bootstrap
from .tracking import Tracker


@dataclass
class ConditionalModelConfig:
    dim: int = 128
    depth: int = 6
    heads: int = 4
    n_points: int = 512
    image_size: int = 32
    patch_size: int = 4


def build_conditional_model(
    cfg: ConditionalModelConfig, device: torch.device
) -> ConditionalPointCloudVelocity:
    return ConditionalPointCloudVelocity(
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
    ).to(device)


def save_checkpoint(path: str, model: nn.Module, cfg: ConditionalModelConfig) -> None:
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ConditionalModelConfig(**ckpt["cfg"])
    model = build_conditional_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ── E-step ──────────────────────────────────────────────────────────────────


def train_estep(
    model: nn.Module,
    x_pool: torch.Tensor,                 # (Nobj, N, 3) current prior — CPU
    z_pool: torch.Tensor | None,          # (Nobj, N, 3) paired noise — CPU
    coupled_fraction: float,
    radius: float,
    noise_std: float,
    image_size: int,
    extent: float,
    epochs: int,
    batch: int,
    lr: float,
    device: torch.device,
    use_amp: bool,
    global_step: list,
    tracker: Tracker | None = None,
    channel: str = "so3",
    so2_axis: str = "z",
) -> None:
    has_z = z_pool is not None and coupled_fraction > 0.0
    dataset = TensorDataset(x_pool, z_pool) if has_z else TensorDataset(x_pool)
    loader = DataLoader(
        dataset, batch_size=batch, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for batch_data in loader:
            if has_z:
                x1, z_coupled = batch_data
                x1 = x1.to(device, non_blocking=(device.type == "cuda"))
                z_coupled = z_coupled.to(device, non_blocking=(device.type == "cuda"))
                B = x1.size(0)
                n_coupled = int(round(coupled_fraction * B))
                if n_coupled >= B:
                    z = z_coupled
                elif n_coupled <= 0:
                    z = torch.randn_like(x1)
                else:
                    z_rand = torch.randn(B - n_coupled, *x1.shape[1:], device=device)
                    z = torch.cat([z_coupled[:n_coupled], z_rand], dim=0)
            else:
                (x1,) = batch_data
                x1 = x1.to(device, non_blocking=(device.type == "cuda"))
                z = torch.randn_like(x1)

            # y_hat = F(x): fresh random pose each batch; no gradient needed.
            with torch.no_grad():
                y_hat = forward_channel(
                    x1, radius=radius, noise_std=noise_std,
                    image_size=image_size, extent=extent, channel=channel, so2_axis=so2_axis,
                )

            with autocast(device, use_amp):
                t = torch.rand(x1.size(0), device=device)
                tt = t[:, None, None]
                xt = (1.0 - tt) * z + tt * x1        # I_t  (linear interpolant)
                target = x1 - z                      # dI_t/dt
                pred = model(xt, t, y_hat)
                loss = (pred - target).pow(2).mean()

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
            if tracker is not None:
                tracker.log({"train/loss": loss.item()}, step=global_step[0])
            global_step[0] += 1

        sched.step()
        print(f"      epoch {epoch:3d}  loss={running / max(n_batches, 1):.5f}")


# ── M-step ──────────────────────────────────────────────────────────────────


@torch.inference_mode()
def update_prior(
    model: nn.Module,
    y_obs: torch.Tensor,        # (Nobj, 1, P, P) fixed observations — CPU
    n_points: int,
    n_steps: int,
    batch: int,
    device: torch.device,
    shapes: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pi(k+1) = Phi(z' | y_obs) via Euler ODE. Returns (x_pool, z_pool) on CPU."""
    model.eval()
    Nobj = y_obs.size(0)
    y_gpu = y_obs.to(device)
    dt = 1.0 / n_steps
    x_chunks, z_chunks = [], []

    for start in range(0, Nobj, batch):
        end = min(start + batch, Nobj)
        b = end - start
        z = torch.randn(b, n_points, 3, device=device)   # X(0) ~ N(0, I)
        y_ch = y_gpu[start:end]
        x = z.clone()
        for k in range(n_steps):
            t = torch.full((b,), k * dt, device=device)
            x = x + model(x, t, y_ch) * dt
        x_chunks.append(x.cpu())
        z_chunks.append(z.cpu())

    x_pool = torch.cat(x_chunks, dim=0)
    z_pool = torch.cat(z_chunks, dim=0)
    print(
        f"    prior  range=[{x_pool.min():.3f}, {x_pool.max():.3f}]  "
        f"std={x_pool.std():.3f}  "
        f"surface_residual={mixture_surface_residual(x_pool, shapes or ['torus']):.4f}"
    )
    return x_pool, z_pool


# ── Visualization ─────────────────────────────────────────────────────────────


def _set3d(ax, lim: float) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def log_em_step(
    gt_eval: torch.Tensor,       # (n, N, 3) clean clouds (reference only)
    y_eval: torch.Tensor,        # (n, 1, P, P) fixed observations
    pi_eval: torch.Tensor,       # (n, N, 3) current prior samples
    radius: float,
    noise_std: float,
    image_size: int,
    extent: float,
    em_step: int,
    tracker: Tracker | None,
    out_dir: Path,
    global_step: list,
    viz_ball_radius: float,
    tag: str = "EM step",
    shapes: list[str] | None = None,
    channel: str = "so3",
    so2_axis: str = "z",
) -> None:
    """4-row panel: GT cloud | y_obs | pi(k) sample | F(pi(k)) consistency check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = gt_eval.size(0)
    with torch.no_grad():
        y_pi = forward_channel(
            pi_eval, radius=radius, noise_std=noise_std,
            image_size=image_size, extent=extent, channel=channel, so2_axis=so2_axis,
        )

    gt_np = gt_eval.cpu().numpy()
    pi_np = pi_eval.cpu().numpy()
    yobs_np = y_eval[:, 0].cpu().numpy()
    ypi_np = y_pi[:, 0].cpu().numpy()
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
            with tempfile.TemporaryDirectory(prefix="pcscsi_balls_") as tmp:
                paths = []
                for j in range(n):
                    p = os.path.join(tmp, f"pi_{em_step:04d}_{j}.obj")
                    save_balls_obj(pi_np[j], p, viz_ball_radius, 1)
                    paths.append(p)
                tracker.log_meshes("eval/pi_balls", paths, step=global_step[0])
        tracker.log({"eval/surface_residual": resid}, step=global_step[0])


# ── Bootstrap: pre-trained rotation prior ──────────────────────────────────────


def pretrain_rotation_prior(
    model: nn.Module,
    y_obs: torch.Tensor,                 # (Nobj, 1, P, P) real observations — CPU
    *,
    shape_fn: Callable[..., torch.Tensor],
    n_points: int,
    radius: float,
    noise_std: float,
    image_size: int,
    extent: float,
    steps: int,
    batch: int,
    lr: float,
    sample_steps: int,
    device: torch.device,
    use_amp: bool,
    seed: int,
    perturb_std: float = 0.0,
    shapes: list[str] | None = None,
    tracker: Tracker | None = None,
    global_step: list | None = None,
    channel: str = "so3",
    so2_axis: str = "z",
) -> torch.Tensor:
    """Warmstart pi(0): pre-train ``model`` to generate random rotations of the
    base object conditioned on F(x), then sample pi(0) on the real observations.

    Each step draws fresh canonical clouds from ``shape_fn``, rotates each by an
    independent random rotation matching the ``channel`` (full SO(3); SO(2) about
    ``so2_axis``; or no rotation for ``"awgn_proj"``), renders ``y = F(rotated)`` with
    F's own fresh random pose / coordinate noise (the
    marginalize-over-pose channel), and takes a flow-matching step on the pair
    ``(rotated, y)``. Rotating the *target* is what lets the recovered prior cover all
    orientations instead of collapsing to the canonical frame. The trained weights are
    kept -- they warm-start the EM model. ``perturb_std`` adds optional 3D jitter to
    the targets (0 = clean). Returns x_pool (Nobj, N, 3) CPU.
    """
    torch.manual_seed(seed)
    print(f"[scsi] bootstrap pre-training: {steps} flow-matching steps on random "
          f"rotations of the base object")

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )
    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    model.train()
    for step in range(1, steps + 1):
        x1 = shape_fn(batch, n_points, device=device)        # canonical clouds
        if channel == "so2":
            R = random_so2(batch, device, axis=so2_axis)
        elif channel == "so3":
            R = random_so3(batch, device)
        else:                                                # awgn_proj: no rotation
            R = None
        if R is not None:
            x1 = torch.matmul(x1, R.transpose(-1, -2))       # random rotation per cloud
        if perturb_std > 0:
            x1 = x1 + perturb_std * torch.randn_like(x1)
        with torch.no_grad():
            y = forward_channel(
                x1, radius=radius, noise_std=noise_std,
                image_size=image_size, extent=extent, channel=channel, so2_axis=so2_axis,
            )                                                # fresh random pose in F
        z = torch.randn_like(x1)

        with autocast(device, use_amp):
            t = torch.rand(batch, device=device)
            tt = t[:, None, None]
            xt = (1.0 - tt) * z + tt * x1                    # I_t  (linear interpolant)
            target = x1 - z                                  # dI_t/dt
            pred = model(xt, t, y)
            loss = (pred - target).pow(2).mean()

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

        if step % 100 == 0 or step == 1:
            print(f"    [pretrain] step {step:6d}/{steps}  loss={loss.item():.5f}")
        if tracker is not None and global_step is not None:
            tracker.log({"bootstrap/loss": loss.item()}, step=global_step[0])
        if global_step is not None:
            global_step[0] += 1

    # Sample pi(0) from the pre-trained model, conditioned on the real observations.
    x_pool, _ = update_prior(model, y_obs, n_points, sample_steps, batch, device, shapes=shapes)
    return x_pool


# ── EM driver ─────────────────────────────────────────────────────────────────


def scsi_train(
    device: torch.device,
    cfg: ConditionalModelConfig,
    n_objects: int = 128,
    em_steps: int = 30,
    epochs_per_em: int = 2,
    batch: int = 32,
    lr: float = 2e-4,
    radius: float = 0.08,
    noise_std: float = 0.1,
    extent: float = 2.0,
    sample_steps: int = 50,
    coupled_fraction: float = 0.0,
    bootstrap: str = "backproject",
    perturb_std: float = 0.0,
    shapes: list[str] | None = None,
    pretrain_steps: int = 2000,
    n_eval: int = 4,
    use_amp: bool = True,
    seed: int = 0,
    tracker: Tracker | None = None,
    out: str = "scsi_checkpoint.pt",
    eval_dir: str = "toy3d_pc_eval",
    viz_ball_radius: float = 0.05,
    channel: str = "so3",
    so2_axis: str = "z",
) -> ConditionalPointCloudVelocity:
    torch.manual_seed(seed)
    configure_backends(device)
    shapes = shapes or ["torus"]
    sample_fn = make_mixture_sampler(shapes)
    print(f"[scsi] device={describe(device)}  amp={use_amp}  shapes={shapes}")

    model = build_conditional_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scsi] parameters: {n_params:,}")

    # Clean ground-truth clouds: used ONLY to make y_obs and for visualization.
    gt = sample_fn(n_objects, cfg.n_points, device=device)           # (Nobj, N, 3)
    with torch.no_grad():
        y_obs = forward_channel(
            gt, radius=radius, noise_std=noise_std,
            image_size=cfg.image_size, extent=extent, channel=channel, so2_axis=so2_axis,
        )                                                            # frozen observations
    gt, y_obs = gt.cpu(), y_obs.cpu()
    print(f"[scsi] y_obs {tuple(y_obs.shape)}  range=[{y_obs.min():.2f}, {y_obs.max():.2f}]")

    out_dir = Path(eval_dir)
    ckpt_dir = Path("toy3d_pc_scsi_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    global_step = [0]  # shared wandb step; the "perturbed" bootstrap advances it too

    # Bootstrap pi(0). The "perturbed" bootstrap pre-trains `model` (fields below);
    # the others ignore the model / pre-training fields and just build a pool.
    ctx = BootstrapContext(
        y_obs=y_obs, n_objects=n_objects, n_points=cfg.n_points,
        extent=extent, seed=seed,
        shapes=shapes, perturb_std=perturb_std,
        model=model, device=device,
        radius=radius, noise_std=noise_std, image_size=cfg.image_size,
        pretrain_steps=pretrain_steps, batch=batch, lr=lr,
        sample_steps=sample_steps, use_amp=use_amp,
        tracker=tracker, global_step=global_step,
        channel=channel, so2_axis=so2_axis,
    )
    x_pool = make_bootstrap(bootstrap, ctx)
    z_pool = None
    print(f"[scsi] bootstrap={bootstrap}  pi(0) {tuple(x_pool.shape)}")

    n_eval = min(n_eval, n_objects)
    gt_eval, y_eval = gt[:n_eval], y_obs[:n_eval]

    for k in range(em_steps):
        print("=" * 60)
        print(f"EM iteration {k} / {em_steps}")
        print("=" * 60)

        train_estep(
            model, x_pool, z_pool, coupled_fraction,
            radius=radius, noise_std=noise_std,
            image_size=cfg.image_size, extent=extent,
            epochs=epochs_per_em, batch=batch, lr=lr,
            device=device, use_amp=use_amp,
            global_step=global_step, tracker=tracker, channel=channel, so2_axis=so2_axis,
        )
        save_checkpoint(str(ckpt_dir / f"model_em{k:04d}.pt"), model, cfg)

        print(f"  M-step: sampling pi({k + 1}) ...")
        x_pool, z_pool = update_prior(
            model, y_obs, cfg.n_points, sample_steps, batch * 3, device, shapes=shapes
        )
        synchronize(device)

        log_em_step(
            gt_eval, y_eval, x_pool[:n_eval],
            radius=radius, noise_std=noise_std,
            image_size=cfg.image_size, extent=extent,
            em_step=k, tracker=tracker, out_dir=out_dir,
            global_step=global_step, viz_ball_radius=viz_ball_radius,
            shapes=shapes, channel=channel, so2_axis=so2_axis,
        )

    save_checkpoint(out, model, cfg)
    print(f"[scsi] saved final checkpoint -> {out}")
    return model


# ── Supervised baseline (debug) ───────────────────────────────────────────────


def train_supervised(
    device: torch.device,
    cfg: ConditionalModelConfig,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 2e-4,
    radius: float = 0.08,
    noise_std: float = 0.1,
    extent: float = 2.0,
    sample_steps: int = 50,
    n_eval: int = 4,
    eval_every: int = 500,
    use_amp: bool = True,
    seed: int = 0,
    shapes: list[str] | None = None,
    tracker: Tracker | None = None,
    out: str = "scsi_checkpoint.pt",
    eval_dir: str = "toy3d_pc_eval",
    viz_ball_radius: float = 0.05,
    channel: str = "so3",
    so2_axis: str = "z",
) -> ConditionalPointCloudVelocity:
    """Supervised oracle: train b_t directly on the coupling (x, F(x)) with unlimited
    fresh ground truth. No EM, pool, or bootstrap -- the upper bound / sanity check
    for ``scsi_train``. If this learns the shape(s) but SCSI doesn't, the fault is in
    the EM dynamics, not the model / channel / conditioning.
    """
    torch.manual_seed(seed)
    configure_backends(device)
    shapes = shapes or ["torus"]
    sample_fn = make_mixture_sampler(shapes)
    print(f"[supervised] device={describe(device)}  amp={use_amp}  shapes={shapes}")

    model = build_conditional_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[supervised] parameters: {n_params:,}")

    # Fixed held-out observations for the eval panel / residual.
    gt_eval = sample_fn(n_eval, cfg.n_points, device=device)
    with torch.no_grad():
        y_eval = forward_channel(
            gt_eval, radius=radius, noise_std=noise_std,
            image_size=cfg.image_size, extent=extent, channel=channel, so2_axis=so2_axis,
        )
    gt_eval, y_eval = gt_eval.cpu(), y_eval.cpu()

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )
    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    out_dir = Path(eval_dir)
    ckpt_dir = Path("toy3d_pc_scsi_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    global_step = [0]

    model.train()
    for step in range(1, steps + 1):
        x1 = sample_fn(batch, cfg.n_points, device=device)      # fresh GT every step
        with torch.no_grad():
            y = forward_channel(
                x1, radius=radius, noise_std=noise_std,
                image_size=cfg.image_size, extent=extent, channel=channel, so2_axis=so2_axis,
            )                                                    # fresh random pose
        z = torch.randn_like(x1)

        with autocast(device, use_amp):
            t = torch.rand(batch, device=device)
            tt = t[:, None, None]
            xt = (1.0 - tt) * z + tt * x1
            target = x1 - z
            pred = model(xt, t, y)
            loss = (pred - target).pow(2).mean()

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
        global_step[0] += 1

        if step % 100 == 0 or step == 1:
            print(f"  [supervised] step {step:6d}/{steps}  loss={loss.item():.5f}")
            if tracker is not None:
                tracker.log({"train/loss": loss.item()}, step=global_step[0])

        if eval_every and (step % eval_every == 0 or step == steps):
            x_pi, _ = update_prior(
                model, y_eval, cfg.n_points, sample_steps, max(batch, n_eval), device,
                shapes=shapes,
            )
            log_em_step(
                gt_eval, y_eval, x_pi[:n_eval],
                radius=radius, noise_std=noise_std,
                image_size=cfg.image_size, extent=extent,
                em_step=step, tracker=tracker, out_dir=out_dir,
                global_step=global_step, viz_ball_radius=viz_ball_radius,
                tag="supervised", shapes=shapes, channel=channel, so2_axis=so2_axis,
            )
            save_checkpoint(str(ckpt_dir / f"model_sup{step:06d}.pt"), model, cfg)
            model.train()

    save_checkpoint(out, model, cfg)
    print(f"[supervised] saved final checkpoint -> {out}")
    return model
