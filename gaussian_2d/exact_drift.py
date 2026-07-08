import numpy as np
from scipy.integrate import solve_ivp

from experiments import run_fixed_y, run_marginal


# ============================================================
# Problem setup: 2D Gaussian AWGN
# ============================================================

rng = np.random.default_rng(0)

d = 2
mu = np.array([1.0, -0.5])

Sigma = np.array([
    [2.0, 0.8],
    [0.8, 0.7],
])

s = 0.7
s2 = s**2
I = np.eye(d)

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

    if t_eval is None:
        t_eval = np.linspace(0, 1, 101)

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
        t_span=(0.0, 1.0),
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
    run_fixed_y(rng, d, mu, Sigma, s2, M, solve_particles)
    run_marginal(rng, d, mu, Sigma, s2, I, solve_particles)
