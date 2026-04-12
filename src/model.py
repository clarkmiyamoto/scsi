import torch
from torch import Tensor
import math

class MLP(torch.nn.Module):
    def __init__(self, 
                 data_dim: int, 
                 hidden_dim: int, 
                 time_embed_dim: int = 64,
                 max_period: int = 10000,
                 conditional_dim: int = 0):
        super().__init__()
    
        
        # Time embedding layers
        # Note: for stochastic interpolants with t in [0,1], use max_period=2
        # (paper Appendix C.1, Table 3).  For diffusion models, use 10000.
        self.time_mlp = torch.nn.Sequential(
            SinusoidalEmbedding(time_embed_dim, max_period=max_period),
            torch.nn.Linear(time_embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main network - note input is data_dim + hidden_dim (concatenated)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(data_dim + conditional_dim + hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x: Tensor, t: Tensor, conditional: Tensor = None) -> Tensor:
        # x: (batch, data_dim)
        # t: (batch,)
        # conditional: (batch, conditional_dim)
        
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        if conditional is not None:
            x_t = torch.cat([x, t_emb, conditional], dim=-1)  # (batch, data_dim + hidden_dim + conditional_dim)
        else:
            x_t = torch.cat([x, t_emb], dim=-1)  # (batch, data_dim + hidden_dim)
        return self.net(x_t)


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (batch,) tensor of time values in [0, 1]
        Returns:
            (batch, dim) sinusoidal embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]  # (batch, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, dim)
        return embedding