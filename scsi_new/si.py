import torch

### Styles of Interpolants

def alpha_linear(t):     return 1.0 - t
def beta_linear(t):      return t
def alpha_dot_linear(t): return -1.0
def beta_dot_linear(t):  return 1.0

def alpha_gvp(t):     return torch.cos(t * torch.pi / 2.0)
def beta_gvp(t):      return torch.sin(t * torch.pi / 2.0)
def alpha_dot_gvp(t): return -torch.pi / 2.0 * torch.sin(t * torch.pi / 2.0)
def beta_dot_gvp(t):  return  torch.pi / 2.0 * torch.cos(t * torch.pi / 2.0)

### Stochastic Interpolants code

def interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                style: str = "linear") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x0: (B, 1, H, W) noise
        x1: (B, 1, H, W) data (x_hat, the previous network's generated sample)
        t:  (B, 1, 1, 1) broadcastable time
    Returns:
        I_t, dI_t/dt
    """
    if style == "linear":
        I_t     = alpha_linear(t) * x0     + beta_linear(t) * x1
        I_dot_t = alpha_dot_linear(t) * x0 + beta_dot_linear(t) * x1
    elif style == "gvp":
        I_t     = alpha_gvp(t) * x0     + beta_gvp(t) * x1
        I_dot_t = alpha_dot_gvp(t) * x0 + beta_dot_gvp(t) * x1
    else:
        raise ValueError(f"Unknown interpolant style: {style!r}")
    return I_t, I_dot_t

def loss_ConditionalDrift(model: torch.nn, 
                          x0: torch.Tensor, 
                          x1: torch.Tensor, 
                          t: torch.Tensor, 
                          y: torch.Tensor, 
                          style: str = "linear") -> torch.Tensor:
    """
    Args:
        model: Neural network that paramterizes drift field `b_t(x|y)`
        x0: (B, 1, H, W) Noise
        x1: (B, 1, H, W) Data 
        y: (B, 1, H, W) Observation / F(x1)
        t:  (B, 1, 1, 1) Broadcastable time
    Returns:
        scalar loss
    """
    I_t, I_dot_t = interpolant(x0, x1, t, style=style)
    return ((model(I_t, t, y) - I_dot_t)**2).mean()