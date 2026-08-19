import numpy as np
from scipy.integrate import solve_ivp

from experiments import run_fixed_y, run_marginal
from helpers import axis_aligned_rank_reduce


# ============================================================
# Problem setup: 2D Gaussian AWGN
# ============================================================

rng = np.random.default_rng(0)

d = 2

# Rank of the prior covariance. RANK == d (default) is the full-rank instance.
# Set RANK < d for a singular / rank-deficient prior: axis_aligned_rank_reduce
# keeps the top-left RANK x RANK block and zeros the trailing coordinate axes,
# so those coords become exact point masses (RANK = 1 in 2D -> Sigma =
# diag(2, 0): x1 spreads, x2 is a delta at mu). em_exact_drift.py imports the
# full base Sigma_full, so its own --rank is independent of this value.
RANK = 2

mu = np.array([1.0, -0.5])

Sigma_full = np.array([
    [2.0, 0.8],
    [0.8, 0.7],
])
Sigma = axis_aligned_rank_reduce(Sigma_full, RANK)

s = 0.7
s2 = s**2
I = np.eye(d)

# The exact drift diverges as t -> 1 in a singular covariance's null direction
# (it must collapse that variance to exactly 0), where S_t = s^2 M becomes
# singular. Integrating to t = 1 - T_END_EPS keeps S_t invertible and the
# drift finite; the residual null-direction std is ~T_END_EPS (visually
# collapsed). For a full-rank Sigma the truncation is negligible.
T_END_EPS = 1e-3

# AWGN posterior matrix:
# M = Sigma (Sigma + s^2 I)^{-1}
M = Sigma @ np.linalg.inv(Sigma + s2 * I) # Yes I know you're supposed to use np.solve(), I am lazy.


# ============================================================
# Time-dependent matrices, Linear Interpolant
# alpha_t = 1-t, beta_t = t
# ============================================================

def S_t(t):
    """
    For alpha_t = 1-t, beta_t = t:
        S_t = (1-t)^2 I + t^2 s^2 M
    """
    return (1 - t)**2 * I + (t**2) * s2 * M


def R_t(t):
    """
    R_t is chosen so that:
        S_t + t R_t = (1-t) I

    Therefore:
        R_t = (1-t) I - t s^2 M
    """
    return (1 - t) * I - t * s2 * M


def C_matrices(t):
    """
    b_t(a | y) = C1_t mu + C2_t a + C3_t y

    where:
        C1_t = (1-t)(I-M) S_t^{-1}
        C2_t = -R_t S_t^{-1}
        C3_t = (1-t) M S_t^{-1}
    """
    S = S_t(t)
    Sinv = np.linalg.inv(S)
    R = R_t(t)

    C1 = (1 - t) * (I - M) @ Sinv
    C2 = -R @ Sinv
    C3 = (1 - t) * M @ Sinv

    return C1, C2, C3


def drift(t, X, Y):
    """
    Vectorized drift.

    X shape: (N, 2)
    Y shape: (N, 2) or (2,)
    """
    C1, C2, C3 = C_matrices(t)

    return (
        mu @ C1.T
        + X @ C2.T
        + Y @ C3.T
    )


# ============================================================
# ODE solver for many particles
# ============================================================

def solve_particles(Z0, Y, t_eval=None):
    """
    Solves:

        dX_t = b_t(X_t | Y) dt,
        X_0 = Z0.

    Z0 shape: (N, 2)

    Y shape:
        (2,)    for fixed observation y
        (N, 2)  for one observation per particle
    """
    N = Z0.shape[0]

    t_end = 1.0 - T_END_EPS
    if t_eval is None:
        t_eval = np.linspace(0, t_end, 101)

    if Y.ndim == 1:
        Y_used = np.broadcast_to(Y, Z0.shape)
    else:
        Y_used = Y

    def rhs(t, flat_X):
        X = flat_X.reshape(N, d)
        dX = drift(t, X, Y_used)
        return dX.reshape(-1)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=Z0.reshape(-1),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    X_t = sol.y.T.reshape(len(t_eval), N, d)
    return t_eval, X_t


# ============================================================
# Run experiments
# ============================================================

if __name__ == "__main__":
    # When the prior is rank-deficient, plot an active coord (< RANK) against a
    # null coord (>= RANK) so the 2D snapshot shows one non-singular and one
    # singular axis.
    dims = (0, 1) if RANK == d else (0, RANK)
    run_fixed_y(rng, d, mu, Sigma, s2, M, solve_particles, dims=dims)
    run_marginal(rng, d, mu, Sigma, s2, I, solve_particles, dims=dims)
