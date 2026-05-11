import torch
import torch.nn as nn
import torch.nn.functional as F
from model import INTEGRATION_SCALE, VOL_SIZE


# ── Interpolant building blocks ──────────────────────────────────────────────

def alpha_linear(t): return 1.0 - t
def beta_linear(t):  return t
def alpha_dot_linear(t): return torch.full_like(t, -1.0)
def beta_dot_linear(t):  return torch.full_like(t,  1.0)

def alpha_gvp(t):     return torch.cos(t * torch.pi / 2.0)
def beta_gvp(t):      return torch.sin(t * torch.pi / 2.0)
def alpha_dot_gvp(t): return -torch.pi / 2.0 * torch.sin(t * torch.pi / 2.0)
def beta_dot_gvp(t):  return  torch.pi / 2.0 * torch.cos(t * torch.pi / 2.0)


def interpolant(
    x0: torch.Tensor,    # (B, 1, D, H, W) — noise
    x1: torch.Tensor,    # (B, 1, D, H, W) — data
    t: torch.Tensor,     # (B, 1, 1, 1, 1) — broadcastable time
    style: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (I_t, dI_t/dt) for 5-D volume tensors."""
    if style == "linear":
        I_t     = alpha_linear(t) * x0     + beta_linear(t) * x1
        I_dot_t = alpha_dot_linear(t) * x0 + beta_dot_linear(t) * x1
    elif style == "gvp":
        I_t     = alpha_gvp(t) * x0     + beta_gvp(t) * x1
        I_dot_t = alpha_dot_gvp(t) * x0 + beta_dot_gvp(t) * x1
    else:
        raise ValueError(f"Unknown interpolant style: {style!r}")
    return I_t, I_dot_t


# ── Loss ─────────────────────────────────────────────────────────────────────

def loss_func(
    model: nn.Module,
    x: torch.Tensor,             # (B, 1, D, H, W)  3D volume
    y: torch.Tensor,             # (B, 1, H, W)      2D projection
    style: str,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    if z is None:
        z = torch.randn_like(x)
    t5 = t[:, None, None, None, None]   # broadcast over (B, 1, D, H, W)

    I_t, I_dot_t = interpolant(z, x, t5, style)

    t_int = (t * INTEGRATION_SCALE).long()
    pred = model(I_t, t_int, y)

    return F.mse_loss(pred, I_dot_t)


# ── ODE Integrators ───────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    model: nn.Module,
    initial_state: torch.Tensor,   # (B, 1, D, H, W)
    y: torch.Tensor,                # (B, 1, H, W)
    n_steps: int = 50,
    method: str = "euler",
) -> torch.Tensor:
    model.eval()
    if method == "euler":
        return _sample_euler(model, initial_state, y, n_steps)
    elif method == "midpoint":
        return _sample_midpoint(model, initial_state, y, n_steps)
    else:
        raise ValueError(f"Unknown method: {method!r}")


@torch.no_grad()
def _sample_euler(model, initial_state, y, n_steps):
    B = y.size(0)
    x = initial_state
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t1 = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device, dtype=torch.long)
        v = model(x, t1, y)
        x = x + v * dt
    return x


@torch.no_grad()
def _sample_midpoint(model, initial_state, y, n_steps):
    B = y.size(0)
    x = initial_state
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t1 = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device, dtype=torch.long)
        v1 = model(x, t1, y)
        x_mid = x + v1 * (dt / 2.0)
        t2 = torch.full((B,), (t_val + dt / 2.0) * INTEGRATION_SCALE,
                         device=y.device, dtype=torch.long)
        v2 = model(x_mid, t2, y)
        x = x + v2 * dt
    return x
