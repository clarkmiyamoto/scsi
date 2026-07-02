"""Supervised oracle (debug upper bound) for the CryoEM point-cloud SCSI (2D toy).

Trains the conditional drift directly on ``(x, F(x))`` with unlimited fresh ground truth
-- no EM, EMA, pool, or bootstrap. This is the sanity check for :func:`scsi.scsi_train`:
if the oracle learns the digit but SCSI does not, the fault is in the EM dynamics, not
the model / channel / conditioning.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .corruption import forward_channel
from .data import load_mnist_digit_pool, make_mnist_sampler
from .device import autocast, configure_backends, describe, needs_grad_scaler
from .model import ConditionalModelConfig, build_conditional_model
from .scsi import log_em_step, save_checkpoint
from .si import interpolant, transport_sample


def train_supervised(
    device: torch.device,
    cfg: ConditionalModelConfig,
    *,
    digit: int = 0,
    n_images: int = 64,
    mnist_size: int = 28,
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
    style: str = "linear",
    tracker=None,
    out: str = "toy_2d_pc_checkpoint.pt",
    eval_dir: str = "toy_2d_pc_eval",
    coord_noise_std: float = 0.0,
    n_tilts: int = 11,
    tilt_step: float = 12.0,
    splat: str,
) -> nn.Module:
    torch.manual_seed(seed)
    configure_backends(device)
    print(f"[supervised] device={describe(device)}  amp={use_amp}  digit={digit}  splat={splat}")

    model = build_conditional_model(cfg, device)
    print(f"[supervised] parameters: {sum(p.numel() for p in model.parameters()):,}")

    fwd = dict(
        radius=radius, noise_std=noise_std, image_size=cfg.image_size, extent=extent,
        coord_noise_std=coord_noise_std, n_tilts=n_tilts, tilt_step=tilt_step, splat=splat,
    )

    pool = load_mnist_digit_pool(digit, n_images, mnist_size, device=device)
    sample_fn = make_mnist_sampler(pool, extent)

    # Fixed held-out observations for the eval panel.
    gt_eval = sample_fn(n_eval, cfg.n_points, device=device)
    with torch.no_grad():
        y_eval = forward_channel(gt_eval, **fwd)
    gt_eval, y_eval = gt_eval.cpu(), y_eval.cpu()

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )
    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    out_dir = Path(eval_dir)
    ckpt_dir = Path("toy_2d_pc_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    global_step = [0]

    model.train()
    for step in range(1, steps + 1):
        x1 = sample_fn(batch, cfg.n_points, device=device)
        with torch.no_grad():
            y = forward_channel(x1, **fwd)

        with autocast(device, use_amp):
            z = torch.randn_like(x1)
            t = torch.rand(batch, device=device)
            I_t, I_dot = interpolant(z, x1, t, style)
            pred = model(I_t, t, y)
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
        global_step[0] += 1

        if step % 100 == 0 or step == 1:
            print(f"  [supervised] step {step:6d}/{steps}  loss={loss.item():.5f}")
            if tracker is not None:
                tracker.log({"train/loss": loss.item()}, step=global_step[0])

        if eval_every and (step % eval_every == 0 or step == steps):
            with torch.no_grad():
                z_eval = torch.randn(n_eval, cfg.n_points, 2, device=device)
                pi = transport_sample(model, z_eval, y_eval.to(device), n_steps=sample_steps).cpu()
            log_em_step(
                gt_eval, y_eval, pi,
                radius=radius, noise_std=noise_std, image_size=cfg.image_size, extent=extent,
                em_step=step, tracker=tracker, out_dir=out_dir, global_step=global_step,
                tag="supervised",
                coord_noise_std=coord_noise_std, n_tilts=n_tilts, tilt_step=tilt_step,
                splat=splat,
            )
            save_checkpoint(str(ckpt_dir / f"model_sup{step:06d}.pt"), model, cfg)
            model.train()

    save_checkpoint(out, model, cfg)
    print(f"[supervised] saved final checkpoint -> {out}")
    return model
