from abc import ABC, abstractmethod
import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Distribution(ABC):
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device

    @abstractmethod
    def sample(self, num_samples: int) -> Tensor:
        '''Sample from the distribution.

        Args:
            num_samples: Number of samples to draw.
        Returns:
            A tensor of shape (num_samples, dim) containing the samples.
        '''
        pass


# ---------------------------------------------------------------------------
# Concrete 2-D distributions
# ---------------------------------------------------------------------------

class Gaussian(Distribution):
    def __init__(self, dim: int = 2, scale: float = 0.5, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.dim = dim
        self.scale = scale

    def sample(self, num_samples: int) -> Tensor:
        return torch.randn(num_samples, self.dim, device=self.device) * self.scale + 0.5


class Checkerboard(Distribution):
    """Uniform distribution over the dark squares of an n x n checkerboard on [0, 1]^2."""

    def __init__(self, num_squares: int = 4, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.num_squares = num_squares

    def sample(self, n_samples: int) -> Tensor:
        grid_size = self.num_squares
        squares_x = torch.randint(0, grid_size, (n_samples,), device=self.device)
        squares_y = torch.randint(0, grid_size, (n_samples,), device=self.device)

        mask = (squares_x + squares_y) % 2 == 0

        offset_x = torch.rand(n_samples, device=self.device)
        offset_y = torch.rand(n_samples, device=self.device)

        x = (squares_x.float() + offset_x) / grid_size
        y = (squares_y.float() + offset_y) / grid_size

        samples = torch.stack([x[mask], y[mask]], dim=1)

        if len(samples) < n_samples:
            additional = self.sample(n_samples - len(samples))
            samples = torch.cat([samples, additional], dim=0)

        return samples[:n_samples]


class Spiral(Distribution):
    """Two interleaving Archimedean spirals with Gaussian noise."""

    def __init__(self, noise: float = 0.05, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.noise = noise

    def sample(self, num_samples: int) -> Tensor:
        n = num_samples // 2
        # First arm
        theta1 = torch.linspace(0, 3 * math.pi, n, device=self.device) + torch.randn(n, device=self.device) * 0.1
        r1 = theta1 / (3 * math.pi)
        x1 = r1 * torch.cos(theta1)
        y1 = r1 * torch.sin(theta1)
        # Second arm (rotated by pi)
        theta2 = torch.linspace(0, 3 * math.pi, num_samples - n, device=self.device) + torch.randn(num_samples - n, device=self.device) * 0.1
        r2 = theta2 / (3 * math.pi)
        x2 = r2 * torch.cos(theta2 + math.pi)
        y2 = r2 * torch.sin(theta2 + math.pi)

        x = torch.cat([x1, x2])
        y = torch.cat([y1, y2])
        samples = torch.stack([x, y], dim=1)
        samples += torch.randn_like(samples, device=self.device) * self.noise
        # Shift to be centred at (0.5, 0.5) for visual consistency
        return samples + 0.5


class TwoMoons(Distribution):
    """Two interleaving crescent-shaped clusters."""

    def __init__(self, noise: float = 0.05, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.noise = noise

    def sample(self, num_samples: int) -> Tensor:
        n = num_samples // 2
        # Upper moon
        theta1 = torch.rand(n, device=self.device) * math.pi
        x1 = torch.cos(theta1)
        y1 = torch.sin(theta1)
        # Lower moon (shifted)
        theta2 = torch.rand(num_samples - n, device=self.device) * math.pi
        x2 = 1.0 - torch.cos(theta2)
        y2 = 1.0 - torch.sin(theta2) - 0.5

        x = torch.cat([x1, x2])
        y = torch.cat([y1, y2])
        samples = torch.stack([x, y], dim=1)
        samples += torch.randn_like(samples, device=self.device) * self.noise
        # Centre around (0.5, 0.5)
        samples[:, 0] = samples[:, 0] * 0.45 + 0.25
        samples[:, 1] = samples[:, 1] * 0.45 + 0.35
        return samples


class Rings(Distribution):
    """Concentric rings (annuli) with Gaussian noise."""

    def __init__(self, num_rings: int = 3, noise: float = 0.02, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.num_rings = num_rings
        self.noise = noise

    def sample(self, num_samples: int) -> Tensor:
        # Assign each sample to a ring uniformly
        ring_idx = torch.randint(0, self.num_rings, (num_samples,), device=self.device)
        radii = (ring_idx.float() + 1) / (self.num_rings + 1)  # evenly spaced
        theta = torch.rand(num_samples, device=self.device) * 2 * math.pi
        x = radii * torch.cos(theta)
        y = radii * torch.sin(theta)
        samples = torch.stack([x, y], dim=1)
        samples += torch.randn_like(samples, device=self.device) * self.noise
        return samples + 0.5  # centre at (0.5, 0.5)


class GaussianMixture(Distribution):
    """Mixture of isotropic Gaussians arranged in a circle."""

    def __init__(self, num_components: int = 8, std: float = 0.04, radius: float = 0.35, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.num_components = num_components
        self.std = std
        self.radius = radius

    def sample(self, num_samples: int) -> Tensor:
        # Assign to a component
        comp = torch.randint(0, self.num_components, (num_samples,), device=self.device)
        angles = 2 * math.pi * comp.float() / self.num_components
        centres_x = self.radius * torch.cos(angles)
        centres_y = self.radius * torch.sin(angles)
        x = centres_x + torch.randn(num_samples, device=self.device) * self.std
        y = centres_y + torch.randn(num_samples, device=self.device) * self.std
        samples = torch.stack([x, y], dim=1)
        return samples + 0.5  # centre at (0.5, 0.5)


class Pinwheel(Distribution):
    """Pinwheel distribution – radial spokes with a twist."""

    def __init__(self, num_spokes: int = 5, noise: float = 0.02, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        self.num_spokes = num_spokes
        self.noise = noise

    def sample(self, num_samples: int) -> Tensor:
        spoke = torch.randint(0, self.num_spokes, (num_samples,), device=self.device)
        base_angle = 2 * math.pi * spoke.float() / self.num_spokes
        r = torch.rand(num_samples, device=self.device) * 0.4 + 0.05
        twist = r * 3.0  # amount of twist increases with radius
        theta = base_angle + twist
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        samples = torch.stack([x, y], dim=1)
        samples += torch.randn_like(samples, device=self.device) * self.noise
        return samples + 0.5


# ---------------------------------------------------------------------------
# Wrapper for pre-computed data tensors
# ---------------------------------------------------------------------------

class DataDistribution(Distribution):
    def __init__(self, data: Tensor, device: torch.device = torch.device("cpu")):
        super().__init__(device)
        '''
        Args:
            data: Shape (batch, dim).
        '''
        self.data: Tensor = data

    def sample(self, num_samples: int) -> Tensor:
        indices = torch.randint(0, len(self.data), (num_samples,), device=self.device)
        return self.data[indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AVAILABLE_DISTRIBUTIONS = [
    'checkerboard', 'spiral', 'two_moons', 'rings',
    'gaussian_mixture', 'pinwheel', 'gaussian',
]


def select_distribution(name: str) -> Distribution:
    """Instantiate a 2-D distribution by name.

    Available: checkerboard, spiral, two_moons, rings,
               gaussian_mixture, pinwheel, gaussian.
    """
    name = name.lower().strip()
    if name == 'checkerboard':
        return Checkerboard(num_squares=4)
    elif name == 'spiral':
        return Spiral()
    elif name == 'two_moons':
        return TwoMoons()
    elif name == 'rings':
        return Rings()
    elif name == 'gaussian_mixture':
        return GaussianMixture()
    elif name == 'pinwheel':
        return Pinwheel()
    elif name == 'gaussian':
        return Gaussian(dim=2, scale=0.3)
    else:
        raise ValueError(
            f"Unknown distribution '{name}'. "
            f"Choose from: {', '.join(AVAILABLE_DISTRIBUTIONS)}"
        )