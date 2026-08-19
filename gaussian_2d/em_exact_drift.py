"""
Self-consistent EM-style drift learning for the d-dimensional Gaussian AWGN
inverse problem solved exactly in exact_drift.py for d=2:

    X ~ N(mu, Sigma)
    Y = F(X) = X + W,  W ~ N(0, s^2 I_d)

At --d 2 (default), mu, Sigma, s are the exact same fixed instance imported
from exact_drift.py, so both scripts model the exact same problem. At any
other --d, a random SPD Sigma and mu are drawn from --seed (see
make_gaussian_awgn_problem); the exact closed-form b_t(a|y) coefficients
generalize unchanged (see make_C_matrices_fn, same derivation as
exact_drift.C_matrices, parametrized instead of hardcoded to d=2).

The learned drift has the constrained form
    b_theta(t, x | y) = A_theta(t) + B_theta(t) x + C_theta(t) y,
where one MLP of t outputs A_theta, B_theta, C_theta. This matches the exact
closed form derived in exact_drift.py:
    b_t(a | y) = (C1_t mu) + C2_t a + C3_t y.

Plots are always 2D snapshots onto a chosen coordinate pair (--plot-dims,
default the first two axes); diagnostics (mean/std z-scores, Mahalanobis
radius, whitened per-axis z-scores) run over all d dimensions regardless.

Run:
    uv run python em_exact_drift.py

For a better fit, increase --n-train and --n-em, ideally on GPU:
    uv run python em_exact_drift.py --n-train 2000 --n-em 6 --device cuda

For a higher-dimensional problem instance:
    uv run python em_exact_drift.py --d 8 --plot-dims 0 3
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from exact_drift import mu as MU_NP, Sigma_full as SIGMA_NP, s2 as S2_NP
from experiments import run_fixed_y, run_marginal
from helpers import axis_aligned_rank_reduce


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Config:
    d: int = 2
    rank: int = 2
    noise_std: float = 0.7
    plot_dims: tuple = (0, 1)
    init_prior_std: float = 0.8
    n_em: int = 10
    n_train: int = 300
    batch_size: int = 512
    lr_start: float = 1e-3
    lr_end: float = 1e-6
    weight_decay: float = 0.0
    hidden: int = 64
    depth: int = 3
    ode_steps_train: int = 64
    ode_steps_eval: int = 64
    n_eval: int = 20000
    seed: int = 0
    device: str = DEFAULT_DEVICE
    out_prefix: str = "scsi_gaussian"


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2, help="Problem dimension. d=2 reuses the fixed instance from exact_drift.py; any other d draws a random SPD Sigma from --seed.")
    p.add_argument("--rank", type=int, default=None, help="Rank of the prior covariance Sigma (axis-aligned: the last d-rank coordinate axes become exact point masses). Default: full rank d.")
    p.add_argument("--noise-std", type=float, default=0.7, help="AWGN std s (only used when --d != 2; the d=2 instance uses exact_drift.py's fixed s)")
    p.add_argument("--plot-dims", type=int, nargs=2, default=None, metavar=("I", "J"), help="Which 2 coordinates (0-indexed) to use for the 2D scatter/contour plots. Diagnostics still cover all d dims. Default: (0, 1), or (0, rank) when rank < d so one plotted axis is singular and one is not.")
    p.add_argument("--init-prior-std", type=float, default=0.8)
    p.add_argument("--n-em", type=int, default=4)
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr-start", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--ode-steps-train", type=int, default=64, help="Euler steps for training (ODE solve)")
    p.add_argument("--ode-steps-eval", type=int, default=100)
    p.add_argument("--t_start_eps", type=float, default=0.0, help="Integrate from [t_start_eps, 1-t_end_eps] to avoid singularities at t=0,1")
    p.add_argument("--t_end_eps", type=float, default=1e-4, help="Integrate from [t_start_eps, 1-t_end_eps] to avoid singularities at t=0,1")
    p.add_argument("--n-eval", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--out-prefix", type=str, default="scsi_gaussian")
    a = p.parse_args()

    if a.d < 2:
        raise ValueError(f"--d must be >= 2, got {a.d}")

    rank = a.d if a.rank is None else a.rank
    if not (1 <= rank <= a.d):
        raise ValueError(f"--rank must be in [1, {a.d}], got {rank}")

    if a.plot_dims is None:
        # Default: pair an active coord with the first null coord when the
        # prior is rank-deficient, so one plotted axis is singular and one is not.
        plot_dims = (0, 1) if rank == a.d else (0, rank)
    else:
        plot_dims = tuple(a.plot_dims)
    if max(plot_dims) >= a.d or min(plot_dims) < 0:
        raise ValueError(f"--plot-dims {plot_dims} out of range for d={a.d}")

    return Config(
        d=a.d,
        rank=rank,
        noise_std=a.noise_std,
        plot_dims=plot_dims,
        init_prior_std=a.init_prior_std,
        n_em=a.n_em,
        n_train=a.n_train,
        batch_size=a.batch_size,
        lr_start=a.lr_start,
        lr_end=a.lr_end,
        weight_decay=a.weight_decay,
        hidden=a.hidden,
        depth=a.depth,
        ode_steps_train=a.ode_steps_train,
        ode_steps_eval=a.ode_steps_eval,
        n_eval=a.n_eval,
        seed=a.seed,
        device=a.device,
        out_prefix=a.out_prefix,
    )


def make_gaussian_awgn_problem(d, seed, noise_std, rank):
    """
    X ~ N(mu, Sigma), Y = X + N(0, s^2 I_d).

    d == 2 reuses the fixed instance from exact_drift.py (its full-rank base
    Sigma_full) so default runs stay directly comparable to exact_drift.py's
    own plots. For any other d, mu and a random SPD Sigma with nontrivial
    off-diagonal correlation are drawn from `seed` (so runs are reproducible),
    and s2 = noise_std**2.

    The full-rank base Sigma is then reduced to `rank` via
    axis_aligned_rank_reduce: rank == d leaves it unchanged; rank < d turns the
    last d-rank coordinate axes into exact point masses (a singular prior).
    """
    if d == 2:
        mu, Sigma_full, s2 = MU_NP.copy(), SIGMA_NP.copy(), float(S2_NP)
    else:
        rng = np.random.default_rng(seed)
        mu = rng.normal(scale=1.0, size=d)
        A = rng.normal(size=(d, d))
        Sigma_full = A @ A.T / d + 0.5 * np.eye(d)
        s2 = float(noise_std ** 2)

    Sigma = axis_aligned_rank_reduce(Sigma_full, rank)
    return mu, Sigma, s2


def make_C_matrices_fn(Sigma, s2, d):
    """
    Builds the exact closed-form coefficient function
        b_t(a | y) = C1(t) mu + C2(t) a + C3(t) y
    for the linear-schedule (alpha_t=1-t, beta_t=t) Gaussian AWGN problem, for
    arbitrary dimension d. Same derivation as exact_drift.C_matrices /
    S_t / R_t, but parametrized on (Sigma, s2, d) instead of exact_drift.py's
    module-level 2D globals, so it works for any problem instance.
    """
    I = np.eye(d)
    M = Sigma @ np.linalg.inv(Sigma + s2 * I)

    def C_matrices_fn(t):
        S = (1 - t) ** 2 * I + (t ** 2) * s2 * M
        Sinv = np.linalg.inv(S)
        R = (1 - t) * I - t * s2 * M

        C1 = (1 - t) * (I - M) @ Sinv
        C2 = -R @ Sinv
        C3 = (1 - t) * M @ Sinv
        return C1, C2, C3

    return C_matrices_fn, M, I


def alpha(t):
    return 1.0 - t


def beta(t):
    return t


def alphadot(t):
    return -torch.ones_like(t)


def betadot(t):
    return torch.ones_like(t)


def gaussian_drift_coeffs(t, prior_var, noise_var):
    """
    Ideal scalar coefficients for the isotropic, zero-mean Gaussian model
        X ~ N(0, prior_var I),
        Y = X + N(0, noise_var I),
        I_t = (1-t)Z + tX.

    Used only to build a deliberately-wrong initial guess Phi^(0): it assumes
    isotropy and zero mean, unlike the true (anisotropic, mu != 0) target.
    """
    a = alpha(t)
    b = beta(t)

    var_I = a**2 + b**2 * prior_var
    cov_IY = b * prior_var
    var_Y = prior_var + noise_var

    cov_dot_I = b * prior_var - a       # Cov(X-Z, I_t)
    cov_dot_Y = prior_var               # Cov(X-Z, Y)

    det = var_I * var_Y - cov_IY**2
    B = (cov_dot_I * var_Y - cov_dot_Y * cov_IY) / det
    C = (-cov_dot_I * cov_IY + cov_dot_Y * var_I) / det
    return B, C


class AnalyticGaussianDrift(nn.Module):
    """
    Analytic, deliberately-wrong isotropic drift used only to initialize
    Phi^(0). This avoids the trivial fixed point Phi^(0)(z|y)=z, and avoids
    starting EM already at the true (anisotropic) answer.
    """
    def __init__(self, d, prior_var, noise_var):
        super().__init__()
        self.d = d
        self.prior_var = float(prior_var)
        self.noise_var = float(noise_var)

    def forward(self, t, x, y):
        B, C = gaussian_drift_coeffs(t, self.prior_var, self.noise_var)
        return B * x + C * y


class LinearInXYDrift(nn.Module):
    """
    b_theta(t, x | y) = A_theta(t) + B_theta(t)x + C_theta(t)y.
    One MLP sees only t and outputs A, B, C.
    """
    def __init__(self, d, hidden=64, depth=3):
        super().__init__()
        self.d = d
        layers = []
        in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(hidden, d + d * d + d * d)]
        self.net = nn.Sequential(*layers)

        # Stable initial ODE: start near zero drift, then learn from data.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self, t):
        out = self.net(t)
        d = self.d
        A = out[:, :d]
        B = out[:, d : d + d * d].reshape(-1, d, d)
        C = out[:, d + d * d :].reshape(-1, d, d)
        return A, B, C

    def forward(self, t, x, y):
        A, B, C = self.coefficients(t)
        Bx = torch.bmm(B, x.unsqueeze(-1)).squeeze(-1)
        Cy = torch.bmm(C, y.unsqueeze(-1)).squeeze(-1)
        return A + Bx + Cy


@torch.no_grad()
def flow(model, z, y, n_steps):
    """Euler solve dX_t = b_t(X_t | y) dt, X_0=z, t in [0,1]."""
    x = z.clone()
    dt = 1.0 / n_steps
    for j in range(n_steps):
        tj = torch.full((z.shape[0], 1), (j + 0.5) * dt, device=z.device)
        x = x + dt * model(tj, x, y)
    return x


@torch.no_grad()
def flow_path(model, z, y, n_steps):
    """Same as flow, but records the full trajectory X_0, ..., X_1."""
    x = z.clone()
    dt = 1.0 / n_steps
    xs = [x.clone()]
    for j in range(n_steps):
        tj = torch.full((z.shape[0], 1), (j + 0.5) * dt, device=z.device)
        x = x + dt * model(tj, x, y)
        xs.append(x.clone())
    return torch.stack(xs, dim=0)


def make_numpy_solve_particles(model, device, n_steps):
    """
    Wraps a torch drift model as a solve_particles(Z0, Y, t_eval=None)
    callable with the same interface as exact_drift.solve_particles, so the
    EM-learned drift can be dropped directly into experiments.run_fixed_y /
    experiments.run_marginal. t_eval is accepted for interface parity but
    ignored: the output grid is always the n_steps Euler grid.
    """
    def solve_particles(Z0, Y, t_eval=None):
        Z0_t = torch.as_tensor(Z0, dtype=torch.float32, device=device)
        Y_np = np.broadcast_to(Y, Z0.shape) if Y.ndim == 1 else Y
        Y_t = torch.as_tensor(Y_np, dtype=torch.float32, device=device)

        path = flow_path(model, Z0_t, Y_t, n_steps)
        t_grid = np.linspace(0.0, 1.0, n_steps + 1)
        return t_grid, path.cpu().numpy()

    return solve_particles


def sample_observed(mu_t, Sigma_t, s2, batch_size, device):
    """Draws real observations y ~ N(mu, Sigma + s^2 I), the true nu."""
    d = mu_t.shape[0]
    cov = Sigma_t + s2 * torch.eye(d, device=device)
    L = torch.linalg.cholesky(cov)
    z = torch.randn(batch_size, d, device=device)
    return mu_t + z @ L.T


def train_one_em_step(cfg, model, opt, scheduler, target_model, mu_t, Sigma_t, s2, em_idx):
    """
    Trains `model` in place for cfg.n_train steps (persistent across EM
    iterations: same weights, same optimizer state, continuing lr schedule).

    `target_model` supplies the frozen Phi^(k) used to produce guessed clean
    samples xhat. On EM step 1 this is the deliberately-wrong analytic guess;
    from EM step 2 onward it is `model` itself (self-consistent continuation
    of the same network), so training never resets.
    """
    losses = []
    s_std = math.sqrt(s2)

    model.train()
    for step in range(1, cfg.n_train + 1):
        # y is real observed data from nu.
        y = sample_observed(mu_t, Sigma_t, s2, cfg.batch_size, cfg.device)
        z = torch.randn(cfg.batch_size, cfg.d, device=cfg.device)
        z_prime = torch.randn(cfg.batch_size, cfg.d, device=cfg.device)
        t = torch.rand(cfg.batch_size, 1, device=cfg.device)

        # EM-style step: freeze Phi^(k), produce guessed clean samples.
        with torch.no_grad():
            xhat = flow(target_model, z_prime, y, cfg.ode_steps_train)
            yhat = xhat + s_std * torch.randn_like(xhat)

            It = alpha(t) * z + beta(t) * xhat
            dIt = alphadot(t) * z + betadot(t) * xhat  # linear schedule: xhat - z

        pred = model(t, It, yhat)
        loss = torch.mean((pred - dIt) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        scheduler.step()

        losses.append(float(loss.detach().cpu()))
        if step % max(1, cfg.n_train // 5) == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"EM {em_idx:02d} | step {step:05d}/{cfg.n_train} | loss {losses[-1]:.4e} | lr {lr:.2e}")

    return model.eval(), losses


@torch.no_grad()
def estimate_marginal_stats(cfg, model, mu_t, Sigma_t, s2, n=None):
    if n is None:
        n = cfg.n_eval
    y = sample_observed(mu_t, Sigma_t, s2, n, cfg.device)
    z = torch.randn(n, cfg.d, device=cfg.device)
    x = flow(model, z, y, cfg.ode_steps_eval)
    x_cpu = x.cpu()
    mean = x_cpu.mean(0)
    cov = torch.cov(x_cpu.T)
    return x_cpu, mean, cov


@torch.no_grad()
def model_coefficients(model, ts):
    """
    Returns A (N,d), B (N,d,d), C (N,d,d) for either drift model type,
    matching b_theta(t,x|y) = A(t) + B(t)x + C(t)y.
    """
    if isinstance(model, LinearInXYDrift):
        return model.coefficients(ts)

    # AnalyticGaussianDrift: isotropic, zero-mean guess b = B(t)x + C(t)y.
    B_scalar, C_scalar = gaussian_drift_coeffs(ts, model.prior_var, model.noise_var)
    eye = torch.eye(model.d, device=ts.device).unsqueeze(0)
    A = torch.zeros(ts.shape[0], model.d, device=ts.device)
    B = B_scalar.view(-1, 1, 1) * eye
    C = C_scalar.view(-1, 1, 1) * eye
    return A, B, C


def ideal_coefficients(ts_np, mu, C_matrices_fn):
    """
    Evaluates the exact closed-form coefficients from C_matrices_fn (see
    make_C_matrices_fn) at each t, using b_t(a|y) = A(t) + B(t)a + C(t)y with
        A(t) = C1(t) @ mu,  B(t) = C2(t),  C(t) = C3(t).
    """
    A_list, B_list, C_list = [], [], []
    for t in ts_np:
        C1, C2, C3 = C_matrices_fn(float(t))
        A_list.append(C1 @ mu)
        B_list.append(C2)
        C_list.append(C3)
    return np.stack(A_list), np.stack(B_list), np.stack(C_list)


@torch.no_grad()
def coefficient_error_diagnostics(model, ts_grid, mu, C_matrices_fn):
    """Error between the learned drift's coefficients and the ideal ones."""
    ts_np = ts_grid.cpu().numpy().squeeze(-1)
    A, B, C = model_coefficients(model, ts_grid)
    A_true, B_true, C_true = ideal_coefficients(ts_np, mu, C_matrices_fn)

    return {
        "t": ts_np,
        "err_A": np.linalg.norm(A.cpu().numpy() - A_true, axis=1),
        "err_B": np.linalg.norm(B.cpu().numpy() - B_true, axis=(1, 2)),
        "err_C": np.linalg.norm(C.cpu().numpy() - C_true, axis=(1, 2)),
    }


def plot_coefficient_error_evolution(all_diag, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    cmap = plt.get_cmap("viridis")
    n = len(all_diag)

    for k, diag in enumerate(all_diag):
        color = cmap(k / max(1, n - 1))
        label = f"k={k}"
        axes[0].plot(diag["t"], diag["err_A"], color=color, label=label)
        axes[1].plot(diag["t"], diag["err_B"], color=color, label=label)
        axes[2].plot(diag["t"], diag["err_C"], color=color, label=label)

    axes[0].set_title(r"$\|A_\theta(t) - A_{\mathrm{true}}(t)\|$")
    axes[1].set_title(r"$\|B_\theta(t) - B_{\mathrm{true}}(t)\|_F$")
    axes[2].set_title(r"$\|C_\theta(t) - C_{\mathrm{true}}(t)\|_F$")
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_yscale("log")
    axes[0].set_ylabel("coefficient error norm")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Learned drift coefficients converging to exact_drift.py's ideal coefficients over EM iterations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_losses(all_losses, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, losses in enumerate(all_losses, start=1):
        ax.plot(losses, label=f"EM {k}")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Drift regression loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    cfg = parse_args()
    if cfg.device == "cpu":
        # On small MLPs, this is often much faster than many BLAS threads.
        torch.set_num_threads(1)

    torch.manual_seed(cfg.seed)

    mu_np, Sigma_np, s2 = make_gaussian_awgn_problem(cfg.d, cfg.seed, cfg.noise_std, cfg.rank)
    C_matrices_fn, M_np, I_np = make_C_matrices_fn(Sigma_np, s2, cfg.d)

    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=cfg.device)
    Sigma_t = torch.tensor(Sigma_np, dtype=torch.float32, device=cfg.device)

    print(f"device: {cfg.device}")
    print(f"d = {cfg.d}, rank = {cfg.rank}, plot_dims = {cfg.plot_dims}")
    print(f"target prior: N(mu={mu_np}, Sigma=\n{Sigma_np})")
    print(f"noise variance s^2 = {s2:.4f}")
    print(f"initial (wrong, isotropic) analytic prior variance: {cfg.init_prior_std**2:.4f}")

    init_model = AnalyticGaussianDrift(
        d=cfg.d,
        prior_var=cfg.init_prior_std**2,
        noise_var=s2,
    ).to(cfg.device).eval()

    # Single persistent network + optimizer, reused across all EM steps
    # (no re-initialization), with an exponential lr decay over the whole
    # run: multiplying by a constant factor each step means the absolute
    # step size shrinks as lr shrinks, so most steps are spent near lr_end
    # rather than near lr_start.
    model = LinearInXYDrift(cfg.d, cfg.hidden, cfg.depth).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    total_steps = cfg.n_em * cfg.n_train
    gamma = (cfg.lr_end / cfg.lr_start) ** (1.0 / total_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    ts_grid = torch.linspace(1e-3, 1 - 1e-3, 300, device=cfg.device).reshape(-1, 1)

    all_losses = []
    stats = []
    all_coeff_diag = []

    current_model = init_model
    _, m0, c0 = estimate_marginal_stats(cfg, current_model, mu_t, Sigma_t, s2)
    stats.append((0, m0, c0))
    all_coeff_diag.append(coefficient_error_diagnostics(current_model, ts_grid, mu_np, C_matrices_fn))
    print(f"k=00 | mean={m0.numpy()} | cov=\n{c0.numpy()}")

    for k in range(1, cfg.n_em + 1):
        # EM step 1 bootstraps off the deliberately-wrong analytic guess;
        # afterwards the network continues from its own current weights.
        target_model = init_model if k == 1 else model
        model, losses = train_one_em_step(cfg, model, opt, scheduler, target_model, mu_t, Sigma_t, s2, k)
        current_model = model
        all_losses.append(losses)

        _, mk, ck = estimate_marginal_stats(cfg, current_model, mu_t, Sigma_t, s2)
        stats.append((k, mk, ck))
        all_coeff_diag.append(coefficient_error_diagnostics(current_model, ts_grid, mu_np, C_matrices_fn))

        err_mean = torch.linalg.norm(mk - mu_t.cpu())
        err_cov = torch.linalg.norm(ck - Sigma_t.cpu())
        print(f"k={k:02d} | mean error={err_mean:.4e} | cov error={err_cov:.4e}")

    # ============================================================
    # Same experiments as exact_drift.py, but driven by the final
    # EM-learned drift instead of the closed-form one. Scatter/contour
    # plots are always a 2D snapshot on cfg.plot_dims; diagnostics
    # (including the Mahalanobis-radius ones) cover all cfg.d dims.
    # ============================================================

    rng = np.random.default_rng(cfg.seed)
    solve_particles_em = make_numpy_solve_particles(current_model, cfg.device, cfg.ode_steps_eval)

    plot_out_prefix = f"{cfg.out_prefix}_d{cfg.d}_r{cfg.rank}_"
    run_fixed_y(rng, cfg.d, mu_np, Sigma_np, s2, M_np, solve_particles_em,
                dims=cfg.plot_dims, out_prefix=plot_out_prefix)
    run_marginal(rng, cfg.d, mu_np, Sigma_np, s2, I_np, solve_particles_em,
                 dims=cfg.plot_dims, out_prefix=plot_out_prefix)

    # ============================================================
    # EM-specific diagnostics: training loss and coefficient convergence
    # ============================================================

    coeff_evo_path = f"{cfg.out_prefix}_d{cfg.d}_r{cfg.rank}_coeff_error_evolution.png"
    loss_path = f"{cfg.out_prefix}_d{cfg.d}_r{cfg.rank}_losses.png"

    plot_coefficient_error_evolution(all_coeff_diag, coeff_evo_path)
    plot_losses(all_losses, loss_path)

    print("\nSummary")
    for k, m, c in stats:
        err = torch.linalg.norm(c - Sigma_t.cpu())
        print(f"k={k:02d}: mean={m.numpy()}, cov=\n{c.numpy()}, ||cov-target||_F={err:.4e}")

    print("\nSaved plots:")
    print(f"  {coeff_evo_path}")
    print(f"  {loss_path}")


if __name__ == "__main__":
    main()
