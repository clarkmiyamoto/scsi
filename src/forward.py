from abc import ABC, abstractmethod
import torch
from torch import Tensor


class ForwardModel(ABC):
    """Abstract base class for a forward corruption model F: X -> Y.

    The forward model is treated as a black-box: we can call F(x) to get
    corrupted observations, but we never differentiate through it.
    """

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Apply the forward corruption model.

        Args:
            x: Clean data samples. Shape (batch, dim).
        Returns:
            Corrupted observations. Shape (batch, dim).
        """
        pass


class IdentityForwardModel(ForwardModel):
    """Trivial pass-through forward model (useful for testing/debugging)."""

    def __call__(self, x: Tensor) -> Tensor:
        return x


class AWGNForwardModel(ForwardModel):
    """Additive White Gaussian Noise channel: y = x + sigma * z.

    This is the simplest corruption model from Section 5 / 6.1 of the paper.
    The channel is injective at the distribution level for any finite sigma.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Standard deviation of the additive noise.
        """
        self.sigma = sigma

    def __call__(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x)
        return x + self.sigma * noise
