import torch


class Distribution:

    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError


class IsotropicGaussian(Distribution):

    def __init__(self, shape: tuple[int, ...], device: torch.device | str | None = None):
        self.shape = shape
        self.device = device

    def sample(self, n_samples: int) -> torch.Tensor:
        return torch.randn(n_samples, *self.shape, device=self.device)
