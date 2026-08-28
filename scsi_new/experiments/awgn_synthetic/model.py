import torch
import torch.nn as nn

POINT_DIM: int = 2  # this experiment is fixed at 2D


class ConditionalVelocityMLP(nn.Module):

    def __init__(self, dim: int = POINT_DIM, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.dim = dim
        layers = [nn.Linear(2 * dim + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x_t, y: (B, dim, 1, 1)   t: (B,) or (B,1,1,1), fractional in [0, 1]
        B = x_t.size(0)
        x_flat = x_t.reshape(B, self.dim)
        y_flat = y.reshape(B, self.dim)
        t_flat = t.reshape(B, 1).to(x_flat.dtype)
        v = self.net(torch.cat([x_flat, t_flat, y_flat], dim=1))
        return v.reshape(B, self.dim, 1, 1)
