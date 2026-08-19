import torch

### Styles of Interpolants
class Interpolant:

    def __call__(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.I_t(x0, x1, t), self.I_dot_t(x0, x1, t)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def I_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t) * x0 + self.beta(t) * x1

    def I_dot_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

class LinearInterpolant(Interpolant):
    def alpha(self, t: torch.Tensor) -> torch.Tensor: return 1.0 - t
    def beta(self, t: torch.Tensor) -> torch.Tensor: return t
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor: return -1.0
    def beta_dot(self, t: torch.Tensor) -> torch.Tensor: return 1.0

class GVPInterpolant(Interpolant):
    def alpha(self, t: torch.Tensor) -> torch.Tensor: return torch.cos(t * torch.pi / 2.0)
    def beta(self, t: torch.Tensor) -> torch.Tensor: return torch.sin(t * torch.pi / 2.0)
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor: return -torch.pi / 2.0 * torch.sin(t * torch.pi / 2.0)
    def beta_dot(self, t: torch.Tensor) -> torch.Tensor: return  torch.pi / 2.0 * torch.cos(t * torch.pi / 2.0)

def load_interpolant(style: str):
    if style == "linear":
        return LinearInterpolant()
    elif style == "gvp":
        return GVPInterpolant()

### Stochastic Interpolants code

def loss_ConditionalDrift(model: torch.nn, 
                          x0: torch.Tensor, 
                          x1: torch.Tensor, 
                          t: torch.Tensor, 
                          y: torch.Tensor, 
                          interpolant: Interpolant) -> torch.Tensor:
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
    I_t, I_dot_t = interpolant(x0, x1, t)
    return ((model(I_t, t, y) - I_dot_t)**2).mean()