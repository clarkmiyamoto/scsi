import numpy as np
import matplotlib.pyplot as plt

from helpers import (
    finite_sample_diagnostics,
    fmt_vec,
    add_mean_std_errorbar,
    add_mean_se_errorbar,
    add_diagnostic_stats_box,
    add_gaussian_contours,
)


def run_fixed_y(rng, d, mu, Sigma, s2, M, solve_particles, N=1000, y_offset=(1.0, 3.0)):
    """
    Fixes y = x_true + y_offset, integrates the conditional ODE from N(0,I)
    particles, and compares the terminal law against the exact posterior
    p(x | y).
    """
    x_true = rng.multivariate_normal(mu, Sigma)
    y_fixed = x_true + np.array(y_offset)

    Z0 = rng.normal(size=(N, d))
    _, X_path = solve_particles(Z0, y_fixed)
    X_final = X_path[-1]

    post_mean = mu + M @ (y_fixed - mu)
    post_cov = s2 * M
    posterior_samples = rng.multivariate_normal(post_mean, post_cov, size=N)

    diag = finite_sample_diagnostics(X_final, post_mean, post_cov)

    # ============================================================
    # Plot 1: fixed-y particle trajectories
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 7))

    num_paths_to_plot = 200

    for i in range(num_paths_to_plot):
        ax.plot(
            X_path[:, i, 0],
            X_path[:, i, 1],
            alpha=0.35,
            linewidth=1,
        )

    ax.scatter(
        Z0[:num_paths_to_plot, 0],
        Z0[:num_paths_to_plot, 1],
        s=15,
        label=r"$X_0 \sim  N(0,I)$"
    )

    ax.scatter(
        X_final[:num_paths_to_plot, 0],
        X_final[:num_paths_to_plot, 1],
        s=15,
        label=r"$X_1$ particles"
    )

    add_mean_std_errorbar(
        ax,
        diag["emp_mean"],
        diag["emp_std"],
        label=r"empirical terminal mean $\pm$ std",
        marker="o",
    )

    add_mean_std_errorbar(
        ax,
        diag["ideal_mean"],
        diag["ideal_std"],
        label=r"ideal posterior mean $\pm$ std",
        marker="x",
    )

    ax.scatter(
        [y_fixed[0]],
        [y_fixed[1]],
        s=260,
        marker="*",
        facecolors="gold",
        edgecolors="black",
        linewidths=1.6,
        label=r"observation $y$",
        zorder=10,
    )

    add_diagnostic_stats_box(
        ax,
        samples=X_final,
        ideal_mean=post_mean,
        ideal_cov=post_cov,
        title=r"Fixed $y$: terminal diagnostic"
    )

    add_gaussian_contours(
        ax,
        mean=mu,
        cov=Sigma,
        label_prefix=r"clean prior $N(\mu,\Sigma)$",
    )

    ax.set_title(r"Particle trajectories for fixed observation $y$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.axis("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("fixed_y_trajectories.png", dpi=300)
    plt.show()

    # ============================================================
    # Plot 2: fixed-y terminal law vs exact posterior
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=10,
        alpha=0.30,
        label=r"exact posterior samples"
    )

    ax.scatter(
        X_final[:, 0],
        X_final[:, 1],
        s=10,
        alpha=0.30,
        label=r"ODE terminal samples"
    )

    add_mean_std_errorbar(
        ax,
        diag["emp_mean"],
        diag["emp_std"],
        label=r"empirical mean $\pm$ marginal std",
        marker="o",
    )

    add_mean_std_errorbar(
        ax,
        diag["ideal_mean"],
        diag["ideal_std"],
        label=r"ideal mean $\pm$ marginal std",
        marker="x",
    )

    add_mean_se_errorbar(
        ax,
        diag["emp_mean"],
        diag["se_mean"],
        label=r"empirical mean $\pm$ mean SE",
        marker="s",
    )

    ax.scatter(
        [y_fixed[0]],
        [y_fixed[1]],
        s=260,
        marker="*",
        facecolors="gold",
        edgecolors="black",
        linewidths=1.6,
        label=r"observation $y$",
        zorder=10,
    )

    add_gaussian_contours(
        ax,
        mean=mu,
        cov=Sigma,
        label_prefix=r"clean prior $N(\mu,\Sigma)$",
    )

    add_gaussian_contours(
        ax,
        mean=post_mean,
        cov=post_cov,
        label_prefix=r"posterior $p(x\mid y)$ contours",
        color="orange"
    )

    add_diagnostic_stats_box(
        ax,
        samples=X_final,
        ideal_mean=post_mean,
        ideal_cov=post_cov,
        title=r"Fixed $y$: terminal law"
    )

    ax.set_title(r"Fixed $y$: ODE terminal law vs exact $p(x\mid y)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.axis("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("fixed_y_terminal_law.png", dpi=300)
    plt.show()

    print("\n================ Fixed y diagnostic ================")
    for key, value in diag.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {fmt_vec(value)}")
        else:
            print(f"{key}: {value}")

    return {
        "y_fixed": y_fixed,
        "X_final": X_final,
        "post_mean": post_mean,
        "post_cov": post_cov,
        "diagnostics": diag,
    }


def run_marginal(rng, d, mu, Sigma, s2, I, solve_particles, N_marginal=3000):
    """
    Draws Y ~ N(mu, Sigma + s^2 I), integrates the conditional ODE per-sample,
    and checks that the terminal marginal recovers the clean prior N(mu, Sigma).
    """
    Y_samples = rng.multivariate_normal(mu, Sigma + s2 * I, size=N_marginal)
    Z0_marginal = rng.normal(size=(N_marginal, d))

    _, X_marginal_path = solve_particles(Z0_marginal, Y_samples)
    X_marginal_final = X_marginal_path[-1]

    prior_samples = rng.multivariate_normal(mu, Sigma, size=N_marginal)

    diag = finite_sample_diagnostics(X_marginal_final, mu, Sigma)

    print("\n================ Marginal diagnostic ================")
    for key, value in diag.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {fmt_vec(value)}")
        else:
            print(f"{key}: {value}")

    # ============================================================
    # Plot 3: marginal recovery diagnostic
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        prior_samples[:, 0],
        prior_samples[:, 1],
        s=8,
        alpha=0.25,
        label=r"true prior samples"
    )

    ax.scatter(
        X_marginal_final[:, 0],
        X_marginal_final[:, 1],
        s=8,
        alpha=0.25,
        label=r"ODE marginal samples"
    )

    add_mean_std_errorbar(
        ax,
        diag["emp_mean"],
        diag["emp_std"],
        label=r"empirical mean $\pm$ marginal std",
        marker="o",
    )

    add_mean_std_errorbar(
        ax,
        diag["ideal_mean"],
        diag["ideal_std"],
        label=r"ideal mean $\pm$ marginal std",
        marker="x",
    )

    add_mean_se_errorbar(
        ax,
        diag["emp_mean"],
        diag["se_mean"],
        label=r"empirical mean $\pm$ mean SE",
        marker="s",
    )

    add_diagnostic_stats_box(
        ax,
        samples=X_marginal_final,
        ideal_mean=mu,
        ideal_cov=Sigma,
        title=r"Marginal law diagnostic"
    )

    add_gaussian_contours(
        ax,
        mean=mu,
        cov=Sigma,
        label_prefix=r"clean prior $p(x) = N(\mu,\Sigma)$",
    )

    ax.set_title(r"Marginal check: $\int p(x_1\mid y)p(y)\,dy = p(x)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.axis("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("marginal_recovery.png", dpi=300)
    plt.show()

    return {
        "X_marginal_final": X_marginal_final,
        "diagnostics": diag,
    }
