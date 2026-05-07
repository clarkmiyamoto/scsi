import torch
import torch.nn as nn
import torch.nn.functional as F
from model import INTEGRATION_SCALE, IMAGE_SIZE, ELL_SCALE

########################################################
# Interpolants
########################################################


##### Construction of interpolant
def interpolant(x0: torch.Tensor, 
                x1: torch.Tensor, 
                t: torch.Tensor, 
                style: str = "linear") -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
        x0: torch.Tensor, shape (B, C, H, W)
        x1: torch.Tensor, shape (B, C, H, W)
        t: torch.Tensor, shape (B, None, None, None)
        style: str, "linear" or "gvp"

    Returns:
        I_t: torch.Tensor, shape (B, C, H, W)
        I_dot_t: torch.Tensor, shape (B, C, H, W)
    '''
    if style == "linear":
        I_t = alpha_linear(t) * x0 + beta_linear(t) * x1
        I_dot_t = alpha_dot_linear(t) * x0 + beta_dot_linear(t) * x1
        return I_t, I_dot_t
    elif style == "gvp":
        I_t = alpha_gvp(t) * x0 + beta_gvp(t) * x1
        I_dot_t = alpha_dot_gvp(t) * x0 + beta_dot_gvp(t) * x1
        return I_t, I_dot_t
    else:
        raise ValueError(f"Unknown style: {style}")

##### Choices of interpolants
# Linear
def alpha_linear(t: torch.Tensor) -> torch.Tensor:
    return 1.0 - t

def beta_linear(t: torch.Tensor) -> torch.Tensor:
    return t

def alpha_dot_linear(t: torch.Tensor) -> torch.Tensor:
    return -1.0

def beta_dot_linear(t: torch.Tensor) -> torch.Tensor:
    return 1.0

# GVP
def alpha_gvp(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * torch.pi / 2.0)

def beta_gvp(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(t * torch.pi / 2.0)

def alpha_dot_gvp(t: torch.Tensor) -> torch.Tensor:
    return -torch.pi / 2.0 * torch.sin(t * torch.pi / 2.0)

def beta_dot_gvp(t: torch.Tensor) -> torch.Tensor:
    return torch.pi / 2.0 * torch.cos(t * torch.pi / 2.0)

########################################################
# Loss
########################################################
def loss_func(model: nn.Module,
         x: torch.Tensor,
         y: torch.Tensor,
         style: str,
         z: torch.Tensor | None = None) -> torch.Tensor:
    """Stochastic interpolant loss with I_t = (1-t)*Z + t*X."""
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    if z is None:
        z = torch.randn_like(x)
    t4 = t[:, None, None, None]
    
    I_t, I_dot_t = interpolant(z, x, t4, style)
    
    t_dit = (t * INTEGRATION_SCALE).long()
    pred = model(I_t, t_dit, y)

    return F.mse_loss(pred, I_dot_t)


########################################################
# ODE Integrators
########################################################
@torch.no_grad()
def sample(model: nn.Module,
           initial_state: torch.Tensor,
           y: torch.Tensor,
           n_steps: int = 50, 
           method: str = "euler") -> torch.Tensor:
    model.eval()
    if method == "euler":
        return _sample_euler(model, initial_state, y, n_steps)
    elif method == "midpoint":
        return _sample_midpoint(model, initial_state, y, n_steps)
    else:
        raise ValueError(f"Unknown method: {method}")

@torch.no_grad()
def _sample_euler(model: nn.Module, initial_state: torch.Tensor, y: torch.Tensor,
                 n_steps: int = 50) -> torch.Tensor:
    model.eval()
    B = y.size(0)
    x = initial_state
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt 
        t1 = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device).long()
        v1 = model(x, t1, y)
        x = x + v1 * dt
    return x

@torch.no_grad()
def _sample_midpoint(model: nn.Module, initial_state: torch.Tensor, y: torch.Tensor,
                    n_steps: int = 50) -> torch.Tensor:
    """Midpoint-rule ODE integration from t=0 (noise) to t=1 (data)."""
    model.eval()
    B = y.size(0)
    x = initial_state
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t1 = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device).long()
        v1 = model(x, t1, y)
        x_mid = x + v1 * (dt / 2.0)
        t2 = torch.full((B,), (t_val + dt / 2.0) * INTEGRATION_SCALE, device=y.device).long()
        v2 = model(x_mid, t2, y)
        x = x + v2 * dt
    return x


########################################################
# Curriculum (ell-conditioned) variants
########################################################
def loss_func_ell(model: nn.Module,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  ell: float,
                  style: str,
                  z: torch.Tensor | None = None) -> torch.Tensor:
    """Stochastic interpolant loss conditioned on curriculum level ell."""
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    if z is None:
        z = torch.randn_like(x)
    t4 = t[:, None, None, None]

    I_t, I_dot_t = interpolant(z, x, t4, style)

    t_dit = (t * INTEGRATION_SCALE).long()
    pred = model(I_t, t_dit, y, ell)

    return F.mse_loss(pred, I_dot_t)


@torch.no_grad()
def sample_ell(model: nn.Module,
               initial_state: torch.Tensor,
               y: torch.Tensor,
               ell: float,
               n_steps: int = 50,
               method: str = "euler") -> torch.Tensor:
    model.eval()
    if method == "euler":
        return _sample_euler_ell(model, initial_state, y, ell, n_steps)
    elif method == "midpoint":
        return _sample_midpoint_ell(model, initial_state, y, ell, n_steps)
    else:
        raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def _sample_euler_ell(model: nn.Module, initial_state: torch.Tensor, y: torch.Tensor,
                      ell: float, n_steps: int = 50) -> torch.Tensor:
    model.eval()
    B = y.size(0)
    x = initial_state
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t1 = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device).long()
        v1 = model(x, t1, y, ell)
        x = x + v1 * dt
    return x


@torch.no_grad()
def _sample_midpoint_ell(model: nn.Module, initial_state: torch.Tensor, y: torch.Tensor,
                         ell: float, n_steps: int = 50) -> torch.Tensor:
    model.eval()
    B = y.size(0)
    x = initial_state
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t1 = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device).long()
        v1 = model(x, t1, y, ell)
        x_mid = x + v1 * (dt / 2.0)
        t2 = torch.full((B,), (t_val + dt / 2.0) * INTEGRATION_SCALE, device=y.device).long()
        v2 = model(x_mid, t2, y, ell)
        x = x + v2 * dt
    return x