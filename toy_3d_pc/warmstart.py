"""Algorithm 1 -- warm-start / initialization of the drift b-hat^(0) for SCSI (CryoET).

Trains the conditional velocity field on the *pseudo-inverse* reconstruction so the EM
loop starts from a transport map that already roughly inverts ``F``:

    for i in 1 .. pretrain_steps:
        y ~ mu                      # a fixed observation (tilt series)
        g ~ SO(3)                   # random global pose (augments the unknown nuisance)
        x-hat = g . F-dagger(y)     # rotated space-carving back-projection
        z ~ N(0, I);  t ~ U(0, 1)
        I_t = alpha_t z + beta_t x-hat
        SGD on  || b_t(I_t | y) - dI_t ||^2

``F-dagger(y_obs)`` is precomputed once by the driver and passed in as ``x_boot`` (so it
can also be visualized as pi(0)). Conditioning is the *real* observation ``y`` -- this is
a supervised pretraining whose clean target is the (rotated) back-projection. Returns the
trained ``Theta^(0)`` in ``model`` (in place).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .corruption import random_rotations, rotate_clouds
from .device import autocast, needs_grad_scaler
from .si import interpolant


def find_initialization(
    model: nn.Module,
    y_obs: torch.Tensor,        # (Nobj, K, P, P) fixed observations -- CPU
    x_boot: torch.Tensor,       # (Nobj, N, 3) F-dagger(y_obs) -- CPU
    *,
    style: str = "linear",
    pretrain_steps: int = 2000,
    batch: int = 32,
    lr: float = 2e-4,
    device: torch.device,
    use_amp: bool = True,
    tracker=None,
    global_step: list | None = None,
) -> None:
    if pretrain_steps <= 0:
        return
    Nobj = y_obs.size(0)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )
    use_scaler = needs_grad_scaler(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    model.train()
    running, n = 0.0, 0
    for step in range(1, pretrain_steps + 1):
        idx = torch.randint(Nobj, (batch,))
        y = y_obs[idx].to(device, non_blocking=(device.type == "cuda"))
        x0 = x_boot[idx].to(device, non_blocking=(device.type == "cuda"))
        x_hat = rotate_clouds(x0, random_rotations(batch, device))   # g . F-dagger(y)

        with autocast(device, use_amp):
            z = torch.randn_like(x_hat)
            t = torch.rand(batch, device=device)
            I_t, I_dot = interpolant(z, x_hat, t, style)
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

        running += loss.item()
        n += 1
        if global_step is not None:
            global_step[0] += 1
        if step % 100 == 0 or step == 1:
            gs = global_step[0] if global_step is not None else step
            print(f"  [warmstart] step {step:5d}/{pretrain_steps}  loss={running / max(n, 1):.5f}")
            if tracker is not None:
                tracker.log({"warmstart/loss": loss.item()}, step=gs)
            running, n = 0.0, 0
