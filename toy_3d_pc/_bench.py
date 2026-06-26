"""Benchmark the EM inner-loop hot sections; compare before/after context-caching fix.

Run with:
    uv run python -m toy_3d_pc._bench
"""
from __future__ import annotations

import time
import torch

from .corruption import forward_channel, tilt_rotations, random_rotations
from .model import ConditionalModelConfig, build_conditional_model, clone_ema
from .si import transport_sample, interpolant
from .device import resolve_device

torch.manual_seed(0)
device = resolve_device("auto")

B = 16
N = 256
sample_steps = 32
n_tilts = 11
image_size = 32
REPS = 5

cfg = ConditionalModelConfig(
    dim=128, depth=6, heads=4, n_points=N,
    image_size=image_size, patch_size=4, in_channels=n_tilts,
)
fwd = dict(
    radius=0.08, noise_std=0.1, image_size=image_size, extent=2.0,
    coord_noise_std=0.0, n_tilts=n_tilts, tilt_step=12.0, tilt_axis="y", splat="gaussian",
)

model = build_conditional_model(cfg, device)
model_ema = clone_ema(model)
model.train()

x  = torch.randn(B, N, 3, device=device)
y  = torch.randn(B, n_tilts, image_size, image_size, device=device)
z  = torch.randn(B, N, 3, device=device)
t  = torch.rand(B, device=device)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timeit(label: str, fn, reps: int = REPS) -> float:
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    elapsed = (time.perf_counter() - t0) / reps
    print(f"  {label:<55s}  {elapsed*1000:7.1f} ms")
    return elapsed


print(f"\n=== Benchmark  device={device}  B={B}  N={N}  sample_steps={sample_steps} ===\n")

# Warmup
for _ in range(3):
    _ = model_ema(x, t, y)
    _ = transport_sample(model_ema, z, y, n_steps=sample_steps)
sync()

# ── Sub-components ────────────────────────────────────────────────────────────
print("── Sub-components ──")
timeit("forward_channel  (incl. random_rotations)", lambda: forward_channel(x.detach().cpu(), **fwd))
timeit("random_rotations  torch-native  B=16", lambda: random_rotations(B, device))
timeit("tilt_rotations  (first call hits scipy)", lambda: tilt_rotations(n_tilts, 12.0, "y", device))

print()
print("── Model forward pass ──")
with torch.no_grad():
    ctx_cached = model_ema.encode_obs(y)

timeit("model_ema.forward  y= (encodes image each call)", lambda: model_ema(x, t, y))
timeit("model_ema.forward  ctx= (pre-encoded, no img enc)", lambda: model_ema(x, t, ctx=ctx_cached))

# ── Integrated transport_sample ───────────────────────────────────────────────
print()
print("── transport_sample (integrated) ──")
t_transport = timeit(f"transport_sample  {sample_steps} steps  (ctx cached once)", lambda: transport_sample(model_ema, z, y, n_steps=sample_steps))

# ── Training step ─────────────────────────────────────────────────────────────
print()
print("── Training step ──")

def fwd_bwd():
    I_t, I_dot = interpolant(z, x, t, "linear")
    pred = model(I_t, t, y)
    loss = (pred - I_dot).pow(2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)

t_fwd_channel = timeit("forward_channel  × 1", lambda: forward_channel(x.detach().cpu(), **fwd))
t_fwd_bwd     = timeit("model fwd+bwd  × 1", fwd_bwd)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("── Per-inner-step cost estimate ──")
total = t_transport + t_fwd_channel + t_fwd_bwd
print(f"  transport_sample  {t_transport*1000:6.1f} ms  ({100*t_transport/total:.0f}%)")
print(f"  forward_channel   {t_fwd_channel*1000:6.1f} ms  ({100*t_fwd_channel/total:.0f}%)")
print(f"  fwd+bwd           {t_fwd_bwd*1000:6.1f} ms  ({100*t_fwd_bwd/total:.0f}%)")
print(f"  TOTAL (estimated) {total*1000:6.1f} ms / inner step")
