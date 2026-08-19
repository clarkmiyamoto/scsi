import torch
from si import Interpolant

class SDE:

    def drift(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift term b_t(X_t | y) of the SDE.
        """
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion term sigma_t of the SDE.
        """
        raise NotImplementedError
   

@torch.no_grad()
def euler_maruyama_integration(
    model: torch.nn.Module,
    x_init_: torch.Tensor,   # (B, 1, H, W) initial noise image
    y: torch.Tensor,         # (B, 1, W) observation to condition on
    n_steps: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Integrate SDE using Euler-Maruyama method. dX_t = b_t(X_t | y) dt + sigma_t dW_t.
    """
    model.eval()

    B = x_init_.size(0)
    x = x_init_
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t_frac = torch.full((B,), t_val, device=x_init_.device)
        v, sigma = model(x, t_frac, y)
        x = x + v * dt + sigma * torch.sqrt(dt) * torch.randn_like(x)
    return x