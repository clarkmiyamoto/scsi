"""Flow matching training + ODE sampling for point clouds.

Linear (optimal-transport) interpolant:
    X_t = (1 - t) * X0 + t * X1            with  X0 ~ N(0, I),  X1 ~ data
    conditional velocity  u = dX_t/dt = X1 - X0

Objective (per-coordinate MSE over the N x 3 entries):
    L = E_{t, X0, X1}  || v_theta(X_t, t) - (X1 - X0) ||^2

Sampling integrates  dX/dt = v_theta(X, t)  from t=0 (noise) to t=1 (data).
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass

import torch

from .balls import save_balls_obj
from .data import sample_torus, torus_surface_residual
from .device import (
    autocast,
    configure_backends,
    describe,
    needs_grad_scaler,
    synchronize,
)
from .model import PointCloudVelocity
from .tracking import Tracker


@dataclass
class ModelConfig:
    dim: int = 128
    depth: int = 6
    heads: int = 4
    n_points: int = 512


def build_model(cfg: ModelConfig, device: torch.device) -> PointCloudVelocity:
    return PointCloudVelocity(cfg.dim, cfg.depth, cfg.heads).to(device)


def train(
    device: torch.device,
    cfg: ModelConfig,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 2e-4,
    use_amp: bool = True,
    compile_model: bool = False,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 200,
    tracker: Tracker | None = None,
    sample_every: int = 0,
    sample_n: int = 4,
    sample_steps: int = 50,
    ball_radius: float = 0.05,
    ball_subdivisions: int = 1,
) -> PointCloudVelocity:
    torch.manual_seed(seed)
    configure_backends(device)
    tracker = tracker or Tracker(enabled=False)
    print(f"[train] device={describe(device)}  amp={use_amp}  compile={compile_model}")

    model = build_model(cfg, device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as exc:  # MPS/older builds can fail; fall back to eager
            print(f"[warn] torch.compile unavailable ({exc}); running eager")

    fused = device.type == "cuda"  # fused AdamW is CUDA-only
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=fused)

    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    model.train()
    synchronize(device)
    t_window = time.perf_counter()
    for step in range(1, steps + 1):
        x1 = sample_torus(batch, cfg.n_points, device=device)  # data   (t=1)
        x0 = torch.randn_like(x1)                              # noise  (t=0)
        t = torch.rand(batch, device=device)                  # (B,)
        tt = t[:, None, None]                                 # broadcast over (N, 3)
        xt = (1.0 - tt) * x0 + tt * x1                         # interpolant
        target = x1 - x0                                       # conditional velocity

        with autocast(device, use_amp):
            pred = model(xt, t)
            loss = (pred - target).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        if step % log_every == 0 or step == 1:
            synchronize(device)
            now = time.perf_counter()
            its = (log_every if step > 1 else 1) / (now - t_window)
            pts_per_s = its * batch * cfg.n_points
            print(
                f"[train] step {step:6d}/{steps}  loss {loss.item():.4f}  "
                f"{its:6.1f} it/s  {pts_per_s/1e6:5.2f} M pts/s"
            )
            tracker.log(
                {"loss": loss.item(), "it_per_s": its, "M_pts_per_s": pts_per_s / 1e6},
                step=step,
            )
            t_window = now

        # Periodically generate clouds and log them as interactive 3D objects:
        # both the raw point cloud AND the solid union-of-balls mesh.
        if sample_every and step % sample_every == 0:
            clouds = sample(model, device, sample_n, cfg.n_points, sample_steps)
            tracker.log_clouds("samples_3d", clouds, step=step)          # points
            if tracker.enabled and ball_radius > 0:
                clouds_np = clouds.cpu().numpy()
                with tempfile.TemporaryDirectory(prefix="pcfm_balls_") as tmp:
                    obj_paths = []
                    for i, c in enumerate(clouds_np):
                        p = os.path.join(tmp, f"balls_{step:06d}_{i}.obj")
                        save_balls_obj(c, p, ball_radius, ball_subdivisions)
                        obj_paths.append(p)
                    tracker.log_meshes("samples_balls", obj_paths, step=step)  # mesh
            tracker.log(
                {"sample_surface_residual": torus_surface_residual(clouds).item()},
                step=step,
            )
            model.train()  # sample() set eval mode; restore for training
            synchronize(device)
            t_window = time.perf_counter()  # don't count sampling in throughput

    return model


@torch.inference_mode()
def sample(
    model: PointCloudVelocity,
    device: torch.device,
    n_clouds: int = 4,
    n_points: int = 512,
    n_steps: int = 100,
    seed: int | None = None,
) -> torch.Tensor:
    """Euler-integrate the probability-flow ODE from noise to data.

    Returns (n_clouds, n_points, 3). Sampling runs in fp32 for accuracy.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    x = torch.randn(n_clouds, n_points, 3, device=device)  # X(0) ~ N(0, I)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((n_clouds,), k * dt, device=device)
        x = x + model(x, t) * dt
    return x


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Get the underlying module if it was wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path: str, model: torch.nn.Module, cfg: ModelConfig) -> None:
    torch.save({"model": unwrap(model).state_dict(), "cfg": asdict(cfg)}, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["cfg"])
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
