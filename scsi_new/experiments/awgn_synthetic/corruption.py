import torch


def corruption_channel(x: torch.Tensor, noise_std: float = 1.0) -> torch.Tensor:
    """
    AWGN channel: y = x + noise_std * N(0, I)
    """
    return x + noise_std * torch.randn_like(x)
