"""
Self-consistent EM-style drift learning for the same 2D Gaussian AWGN inverse
problem solved exactly in exact_drift.py:

    X ~ N(mu, Sigma)
    Y = F(X) = X + W,  W ~ N(0, s^2 I_2)

mu, Sigma, s are imported directly from exact_drift.py so both scripts model
the exact same problem instance.

The learned drift has the constrained form
    b_theta(t, x | y) = A_theta(t) + B_theta(t) x + C_theta(t) y,
where one MLP of t outputs A_theta, B_theta, C_theta. This matches the exact
closed form derived in exact_drift.py:
    b_t(a | y) = (C1_t mu) + C2_t a + C3_t y.

Run:
    uv run python em_exact_drift.py

For a better fit, increase --n-train and --n-em, ideally on GPU:
    uv run python em_exact_drift.py --n-train 2000 --n-em 6 --device cuda
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from exact_drift import mu as MU_NP, Sigma as SIGMA_NP, s2 as S2_NP, I as I_NP, M as M_NP, C_matrices
from experiments import run_fixed_y, run_marginal


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Config:
    d: int = 2
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
    p.add_argument("--init-prior-std", type=float, default=0.8)
    p.add_argument("--n-em", type=int, default=4)
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr-start", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--ode-steps-train", type=int, default=64)
    p.add_argument("--ode-steps-eval", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--out-prefix", type=str, default="scsi_gaussian")
    a = p.parse_args()
    return Config(
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


def ideal_coefficients(ts_np):
    """
    Evaluates the exact closed-form coefficients from exact_drift.C_matrices
    at each t, using b_t(a|y) = A(t) + B(t)a + C(t)y with
        A(t) = C1(t) @ mu,  B(t) = C2(t),  C(t) = C3(t).
    """
    A_list, B_list, C_list = [], [], []
    for t in ts_np:
        C1, C2, C3 = C_matrices(float(t))
        A_list.append(C1 @ MU_NP)
        B_list.append(C2)
        C_list.append(C3)
    return np.stack(A_list), np.stack(B_list), np.stack(C_list)


@torch.no_grad()
def coefficient_error_diagnostics(model, ts_grid):
    """Error between the learned drift's coefficients and the ideal ones."""
    ts_np = ts_grid.cpu().numpy().squeeze(-1)
    A, B, C = model_coefficients(model, ts_grid)
    A_true, B_true, C_true = ideal_coefficients(ts_np)

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

    mu_t = torch.tensor(MU_NP, dtype=torch.float32, device=cfg.device)
    Sigma_t = torch.tensor(SIGMA_NP, dtype=torch.float32, device=cfg.device)
    s2 = float(S2_NP)

    print(f"device: {cfg.device}")
    print(f"target prior: N(mu={MU_NP}, Sigma=\n{SIGMA_NP})")
    print(f"noise variance s^2 = {s2:.4f}")
    print(f"initial (wrong, isotropic) analytic prior variance: {cfg.init_prior_std**2:.4f}")

    init_model = AnalyticGaussianDrift(
        d=cfg.d,
        prior_var=cfg.init_prior_std**2,
        noise_var=s2,
    ).to(cfg.device).eval()

    # Single persistent network + optimizer, reused across all EM steps
    # (no re-initialization), with a linear lr decay over the whole run.
    model = LinearInXYDrift(cfg.d, cfg.hidden, cfg.depth).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    total_steps = cfg.n_em * cfg.n_train
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=cfg.lr_end / cfg.lr_start, total_iters=total_steps
    )

    ts_grid = torch.linspace(1e-3, 1 - 1e-3, 300, device=cfg.device).reshape(-1, 1)

    all_losses = []
    stats = []
    all_coeff_diag = []

    current_model = init_model
    _, m0, c0 = estimate_marginal_stats(cfg, current_model, mu_t, Sigma_t, s2)
    stats.append((0, m0, c0))
    all_coeff_diag.append(coefficient_error_diagnostics(current_model, ts_grid))
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
        all_coeff_diag.append(coefficient_error_diagnostics(current_model, ts_grid))
        
        err_mean = torch.linalg.norm(mk - mu_t.cpu())
        err_cov = torch.linalg.norm(ck - Sigma_t.cpu())
        print(f"k={k:02d} | mean error={err_mean:.4e} | cov error={err_cov:.4e}")

    # ============================================================
    # Same experiments as exact_drift.py, but driven by the final
    # EM-learned drift instead of the closed-form one.
    # ============================================================

    rng = np.random.default_rng(cfg.seed)
    solve_particles_em = make_numpy_solve_particles(current_model, cfg.device, cfg.ode_steps_eval)

    run_fixed_y(rng, cfg.d, MU_NP, SIGMA_NP, s2, M_NP, solve_particles_em)
    run_marginal(rng, cfg.d, MU_NP, SIGMA_NP, s2, I_NP, solve_particles_em)

    # ============================================================
    # EM-specific diagnostics: training loss and coefficient convergence
    # ============================================================

    coeff_evo_path = f"{cfg.out_prefix}_coeff_error_evolution.png"
    loss_path = f"{cfg.out_prefix}_losses.png"

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
