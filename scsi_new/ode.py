import torch
import torch.nn as nn

@torch.no_grad()
def euler_integration(
    model: nn.Module,
    x_init: torch.Tensor,   # (B, 1, H, W) initial noise image
    y: torch.Tensor,         # (B, 1, W) observation to condition on
    n_steps: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Integrate ODE using Euler's method. dX_t = b_t(X_t | y) dt.
    """
    model.eval()

    B = x_init.size(0)
    x = x_init
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t_frac = torch.full((B,), t_val, device=x_init.device)
        v = model(x, t_frac, y)
        x = x + v * dt
    return x


@torch.no_grad()
def euler_integration_trajectory(
    model: nn.Module,
    x_init: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 64,
    n_snapshots: int = 8,
) -> torch.Tensor:
    """
    Same as euler_integration, but also returns n_snapshots evenly-spaced intermediate states
    (including t=0 and t=1) for visualization.
    """
    model.eval()

    B = x_init.size(0)
    x = x_init
    dt = 1.0 / n_steps
    snap_steps = sorted(set(torch.linspace(0, n_steps, n_snapshots).round().long().tolist()))

    snapshots = []
    if 0 in snap_steps:
        snapshots.append(x.clone())
    for i in range(n_steps):
        t_val = i * dt
        t_frac = torch.full((B,), t_val, device=x_init.device)
        v = model(x, t_frac, y)
        x = x + v * dt
        if (i + 1) in snap_steps:
            snapshots.append(x.clone())
    return torch.stack(snapshots, dim=0)
