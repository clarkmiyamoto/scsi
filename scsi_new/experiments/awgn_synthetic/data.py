'''
Synthetic 2D distributions under an AWGN channel (y = x + noise), for stress-testing whether
SCSI's EM loop can recover the FULL ground-truth distribution -- including all its modes -- when
warm-started on a deliberately poor prior.

Distributions
  - The ground-truth p(x) that build_observations() draws from (default: an 8-mode ring of
    Gaussians -- multi-modal on purpose).
  - The warm-start prior build_warmup() draws x_hat from: a single user-specified Gaussian
    (mean, covariance) that need not resemble the ground truth at all.
'''
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, TensorDataset

from corruption import corruption_channel

POINT_DIM = 2  # this experiment is fixed at 2D


### Sampling primitives ###################################################

def sample_gaussian(n_samples: int, mean: list[float], cov: list[list[float]]) -> torch.Tensor:
    """Single 2D Gaussian. mean: [mx, my]. cov: 2x2 covariance matrix (may be non-diagonal)."""
    mean_t = torch.tensor(mean, dtype=torch.float32)
    cov_t = torch.tensor(cov, dtype=torch.float32)
    return torch.distributions.MultivariateNormal(mean_t, covariance_matrix=cov_t).sample((n_samples,))


def sample_gmm(n_samples: int, means: list[list[float]], covs: list[list[list[float]]],
               weights: list[float] | None = None) -> torch.Tensor:
    """
    2D Gaussian mixture w/ arbitrary modes. means[k]/covs[k] as in sample_gaussian. weights
    default to uniform and are renormalized if they don't already sum to 1.
    """
    K = len(means)
    weights_t = torch.tensor(weights if weights is not None else [1.0 / K] * K, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum()
    counts = torch.bincount(torch.multinomial(weights_t, n_samples, replacement=True), minlength=K)
    samples = torch.cat([sample_gaussian(int(counts[k]), means[k], covs[k]) for k in range(K)], dim=0)
    return samples[torch.randperm(samples.size(0))]


def sample_ring(n_samples: int, center: list[float], radius: float, width: float = 0.0) -> torch.Tensor:
    """2D ring: points at `radius` from `center` (+/- Gaussian radial jitter `width`), uniform angle."""
    angle = torch.rand(n_samples) * 2 * torch.pi
    r = radius + width * torch.randn(n_samples)
    center_t = torch.tensor(center, dtype=torch.float32)
    return center_t + r[:, None] * torch.stack([angle.cos(), angle.sin()], dim=1)


def _ring_centers(center: list[float], radius: float, n_modes: int) -> torch.Tensor:
    angles = torch.arange(n_modes, dtype=torch.float32) * (2 * torch.pi / n_modes)
    center_t = torch.tensor(center, dtype=torch.float32)
    return center_t + radius * torch.stack([angles.cos(), angles.sin()], dim=1)


def sample_ring_gmm(n_samples: int, center: list[float], radius: float, n_modes: int,
                    mode_std: float) -> torch.Tensor:
    """Gaussian mixture model with modes in ring."""
    means = _ring_centers(center, radius, n_modes).tolist()
    covs = [[[mode_std ** 2, 0.0], [0.0, mode_std ** 2]] for _ in range(n_modes)]
    return sample_gmm(n_samples, means, covs)


import numpy as np
import matplotlib.pyplot as plt


def sample_segment(a, b, n, thickness=0.02):
    """Sample points around a line segment."""
    t = np.random.rand(n, 1)
    x = a + t * (b - a)
    return x + thickness * np.random.randn(n, 2)


def sample_ellipse(center, rx, ry, theta0, theta1, n, thickness=0.02):
    """Sample points around an elliptical arc."""
    theta = np.random.uniform(theta0, theta1, n)
    x = np.column_stack([
        center[0] + rx * np.cos(theta),
        center[1] + ry * np.sin(theta),
    ])
    return x + thickness * np.random.randn(n, 2)


def sample_miffy(n=10_000, thickness=0.02):
    """Draw n samples from a simple 2D Miffy-face distribution."""
    parts = []

    # Relative sampling weights
    counts = {
        "head": int(0.45 * n),
        "ears": int(0.25 * n),
        "eyes": int(0.15 * n),
    }
    counts["mouth"] = n - sum(counts.values())

    # ---------------- Head ----------------
    # Lower ~80% of an ellipse, leaving the top open for the ears.
    parts.append(
        sample_ellipse(
            center=np.array([0.0, 0.0]),
            rx=1.0,
            ry=0.85,
            theta0=np.deg2rad(20),
            theta1=np.deg2rad(160),
            n=counts["head"] // 2,
            thickness=thickness,
        )
    )

    parts.append(
        sample_ellipse(
            center=np.array([0.0, 0.0]),
            rx=1.0,
            ry=0.85,
            theta0=np.deg2rad(160),
            theta1=np.deg2rad(380),
            n=counts["head"] - counts["head"] // 2,
            thickness=thickness,
        )
    )

    # ---------------- Ears ----------------
    ne = counts["ears"] // 2

    # Elliptical ears
    parts.append(
        sample_ellipse(
            center=np.array([-0.42, 1.18]),
            rx=0.24,
            ry=0.78,
            theta0=0,
            theta1=2 * np.pi,
            n=ne,
            thickness=thickness,
        )
    )

    parts.append(
        sample_ellipse(
            center=np.array([0.42, 1.18]),
            rx=0.24,
            ry=0.78,
            theta0=0,
            theta1=2 * np.pi,
            n=counts["ears"] - ne,
            thickness=thickness,
        )
    )

    # ---------------- Eyes ----------------
    neye = counts["eyes"] // 2

    parts.append(
        sample_ellipse(
            np.array([-0.32, 0.18]),
            0.045, 0.095,
            0, 2*np.pi,
            neye,
            thickness=thickness / 2,
        )
    )

    parts.append(
        sample_ellipse(
            np.array([0.32, 0.18]),
            0.045, 0.095,
            0, 2*np.pi,
            counts["eyes"] - neye,
            thickness=thickness / 2,
        )
    )

    # ---------------- X-shaped mouth ----------------
    nm = counts["mouth"] // 2

    parts.append(
        sample_segment(
            np.array([-0.12, -0.22]),
            np.array([0.12, -0.42]),
            nm,
            thickness=thickness / 2,
        )
    )

    parts.append(
        sample_segment(
            np.array([-0.12, -0.42]),
            np.array([0.12, -0.22]),
            counts["mouth"] - nm,
            thickness=thickness / 2,
        )
    )

    x = np.concatenate(parts, axis=0)
    np.random.shuffle(x)

    return torch.from_numpy(x).float()


### Config + dispatch ######################################################

@dataclass
class Config_Dataset_Synthetic:
    # Ground-truth distribution p(x) that build_observations() draws from.
    gt_distribution: str = "ring_gmm"  # "gaussian" | "gmm" | "ring" | "ring_gmm"
    n_samples: int = 10_000
    gt_mean: list[float] = field(default_factory=lambda: [0.0, 0.0])   # gaussian/ring/ring_gmm
    gt_cov: list[list[float]] = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])  # gaussian only
    gt_means: list[list[float]] | None = None       # gmm only -- not exposed via CLI, construct
    gt_covs: list[list[list[float]]] | None = None  # gmm only -- this Config directly for a
    gt_weights: list[float] | None = None            # gmm only -- custom mixture.
    ring_radius: float = 4.0
    ring_width: float = 0.3      # ring only (Gaussian radial jitter)
    ring_n_modes: int = 8        # ring_gmm only
    ring_mode_std: float = 0.3   # ring_gmm only
    miffy_width: float = 2.0
    miffy_height: float = 3.0
    miffy_thickness: float = 0.2

    # Corruption channel
    noise_std: float = 1.5

    # Warmup: user-specified (possibly POOR) Gaussian the model is first trained on, independent
    # of the ground-truth distribution above -- see main.py.
    warmup_mean: list[float] = field(default_factory=lambda: [0.0, 0.0])
    warmup_cov: list[list[float]] = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])
    n_warmup_samples: int = 10_000

    # Misc
    seed: int = 42


def sample_gt_distribution(config: Config_Dataset_Synthetic, n_samples: int) -> torch.Tensor:
    """Dispatches on config.gt_distribution. Returns (n_samples, 2)."""
    if config.gt_distribution == "gaussian":
        return sample_gaussian(n_samples, config.gt_mean, config.gt_cov)
    elif config.gt_distribution == "gmm":
        return sample_gmm(n_samples, config.gt_means, config.gt_covs, config.gt_weights)
    elif config.gt_distribution == "ring":
        return sample_ring(n_samples, config.gt_mean, config.ring_radius, config.ring_width)
    elif config.gt_distribution == "ring_gmm":
        return sample_ring_gmm(n_samples, config.gt_mean, config.ring_radius,
                               config.ring_n_modes, config.ring_mode_std)
    elif config.gt_distribution == "miffy":
        return sample_miffy(n_samples, thickness=config.miffy_thickness)
    raise ValueError(f"Unknown gt_distribution: {config.gt_distribution!r}")


def get_mode_centers(config: Config_Dataset_Synthetic) -> torch.Tensor | None:
    """
    Analytic mode centers (K, 2), for the mode-occupancy recovery metric in wandb_logging.py.
    None for continuous/unimodal targets (gaussian/ring) where "mode coverage" doesn't apply, or
    for gmm configs that didn't supply gt_means.
    """
    if config.gt_distribution == "ring_gmm":
        return _ring_centers(config.gt_mean, config.ring_radius, config.ring_n_modes)
    if config.gt_distribution == "gmm" and config.gt_means is not None:
        return torch.tensor(config.gt_means, dtype=torch.float32)
    return None


### Dataset builders ########################################################

def build_observations(config: Config_Dataset_Synthetic) -> Dataset:
    '''
    Draws config.n_samples points from the configured ground-truth distribution and applies the
    AWGN forward model to produce observations y = x + noise. Analogous to the MNIST
    experiments' build_observations(), but the "dataset" is synthetic rather than loaded from
    disk. Seeded off config.seed (save/restore) so the observation set is fixed across runs with
    the same config, regardless of what warm start is being compared against it.
    '''
    rng_state = torch.get_rng_state()
    torch.manual_seed(config.seed)
    x_gt = sample_gt_distribution(config, config.n_samples).view(-1, POINT_DIM, 1, 1)
    torch.set_rng_state(rng_state)

    y_obs = corruption_channel(x_gt, noise_std=config.noise_std)
    return TensorDataset(y_obs)


def build_warmup(config: Config_Dataset_Synthetic) -> Dataset:
    '''
    Builds a DELIBERATELY-POOR warm-start dataset: (x_hat, y_hat) pairs where x_hat is drawn from
    a user-specified Gaussian (config.warmup_mean / config.warmup_cov) that need NOT resemble the
    ground-truth distribution build_observations() draws from, and y_hat = F(x_hat) under the
    SAME AWGN channel. This is the experiment's whole point (see main.py's module docstring):
    unlike the MNIST experiments' build_warmup(), which reconstructs x_hat from the actual
    observations (pseudoinverse / identity), this one is intentionally uninformed by the data --
    it only takes `config`, not `observations`.

    Seeded off config.seed (save/restore, offset by +1 from build_observations' draw so the two
    aren't literally the same RNG stream) so the warm start -- the independent variable this
    experiment studies -- is reproducible given the config.
    '''
    rng_state = torch.get_rng_state()
    torch.manual_seed(config.seed + 1)
    x_hat = sample_gaussian(config.n_warmup_samples, config.warmup_mean, config.warmup_cov)
    x_hat = x_hat.view(-1, POINT_DIM, 1, 1)
    y_hat = corruption_channel(x_hat, noise_std=config.noise_std)
    torch.set_rng_state(rng_state)

    return TensorDataset(x_hat, y_hat)


def build_reference_pool(config: Config_Dataset_Synthetic, n: int) -> torch.Tensor:
    '''
    Deterministic CLEAN sample from the ground-truth distribution, for diagnostics/metrics only
    -- NEVER fed to training (the algorithm only ever sees noisy y). Seeded off config.seed
    (offset +2, independent of build_observations' own draw) so it's reproducible but not
    literally the same points as the training observations.
    '''
    rng_state = torch.get_rng_state()
    torch.manual_seed(config.seed + 2)
    x_gt_ref = sample_gt_distribution(config, n).view(-1, POINT_DIM, 1, 1)
    torch.set_rng_state(rng_state)
    return x_gt_ref


def build_viz_pool(config: Config_Dataset_Synthetic, n_pool: int, viz_seed: int) -> dict:
    '''
    Small diagnostic pool for the per-example trajectory panel: n_pool ground-truth points
    (same distribution build_observations() draws from), with a fixed x0/y seeded independently
    by viz_seed. Runs on the CPU RNG with state saved/restored, per the MNIST experiments'
    convention, so it has no effect on the rest of the run's RNG stream.
    '''
    rng_state = torch.get_rng_state()
    torch.manual_seed(viz_seed)
    x_gt = sample_gt_distribution(config, n_pool).view(-1, POINT_DIM, 1, 1)
    x0 = torch.randn(n_pool, POINT_DIM, 1, 1)
    y = corruption_channel(x_gt, noise_std=config.noise_std)
    torch.set_rng_state(rng_state)

    return {"x_gt": x_gt, "x0": x0, "y": y}
