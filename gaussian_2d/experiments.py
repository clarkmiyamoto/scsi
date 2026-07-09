import numpy as np
import matplotlib.pyplot as plt

from helpers import (
    finite_sample_diagnostics,
    fmt_vec,
    add_mean_std_errorbar,
    add_mean_se_errorbar,
    add_diagnostic_stats_box,
    add_gaussian_contours,
    radius_diagnostics,
    add_radius_diagnostics_box,
    plot_radius_sq_histogram,
    whitened_diagnostics,
    make_joint_grid,
    add_marginal_hists,
)


def run_fixed_y(rng, d, mu, Sigma, s2, M, solve_particles, N=1000, y_offset=None,
                 dims=(0, 1), out_prefix=""):
    """
    Fixes y = x_true + y_offset, integrates the conditional ODE from N(0,I)
    particles, and compares the terminal law against the exact posterior
    p(x | y).

    All diagnostics (mean/std/z-scores, Mahalanobis radius) are computed over
    the full d dimensions. Spatial scatter/contour plots only ever show 2
    axes at a time -- `dims` selects which coordinate pair -- since a single
    2D snapshot can't depict a d>2 distribution directly. `y_offset` defaults
    to np.linspace(1.0, 3.0, d), which reduces to the original (1.0, 3.0) at
    d=2.
    """
    if y_offset is None:
        y_offset = np.linspace(1.0, 3.0, d)

    i, j = dims

    x_true = rng.multivariate_normal(mu, Sigma)
    y_fixed = x_true + np.array(y_offset)

    Z0 = rng.normal(size=(N, d))
    _, X_path = solve_particles(Z0, y_fixed)
    X_final = X_path[-1]

    post_mean = mu + M @ (y_fixed - mu)
    post_cov = s2 * M
    posterior_samples = rng.multivariate_normal(post_mean, post_cov, size=N)

    diag = finite_sample_diagnostics(X_final, post_mean, post_cov)
    rdiag = radius_diagnostics(X_final, post_mean, post_cov)

    mean_2d = np.array([post_mean[i], post_mean[j]])
    cov_2d = post_cov[np.ix_([i, j], [i, j])]
    prior_mean_2d = np.array([mu[i], mu[j]])
    prior_cov_2d = Sigma[np.ix_([i, j], [i, j])]

    # ============================================================
    # Plot 1: fixed-y particle trajectories (2D snapshot on dims)
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 7))

    num_paths_to_plot = 200

    for k in range(num_paths_to_plot):
        ax.plot(
            X_path[:, k, i],
            X_path[:, k, j],
            alpha=0.35,
            linewidth=1,
        )

    ax.scatter(
        Z0[:num_paths_to_plot, i],
        Z0[:num_paths_to_plot, j],
        s=15,
        label=r"$X_0 \sim  N(0,I)$"
    )

    ax.scatter(
        X_final[:num_paths_to_plot, i],
        X_final[:num_paths_to_plot, j],
        s=15,
        label=r"$X_1$ particles"
    )

    add_mean_std_errorbar(
        ax,
        diag["emp_mean"][[i, j]],
        diag["emp_std"][[i, j]],
        label=r"empirical terminal mean $\pm$ std",
        marker="o",
    )

    add_mean_std_errorbar(
        ax,
        diag["ideal_mean"][[i, j]],
        diag["ideal_std"][[i, j]],
        label=r"ideal posterior mean $\pm$ std",
        marker="x",
    )

    ax.scatter(
        [y_fixed[i]],
        [y_fixed[j]],
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
        mean=prior_mean_2d,
        cov=prior_cov_2d,
        label_prefix=r"clean prior $N(\mu,\Sigma)$",
    )

    ax.set_title(rf"Particle trajectories for fixed observation $y$ (dims {i+1},{j+1} of {d})")
    ax.set_xlabel(rf"$x_{{{i+1}}}$")
    ax.set_ylabel(rf"$x_{{{j+1}}}$")
    ax.axis("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(f"{out_prefix}fixed_y_trajectories.png", dpi=300)
    plt.show()

    # ============================================================
    # Plot 2: fixed-y terminal law vs exact posterior (2D snapshot),
    # with marginal histograms of the ODE terminal samples vs the exact
    # 1D posterior marginals on the top/right axes.
    # ============================================================

    fig, ax, ax_histx, ax_histy = make_joint_grid()

    ax.scatter(
        posterior_samples[:, i],
        posterior_samples[:, j],
        s=10,
        alpha=0.30,
        label=r"exact posterior samples"
    )

    ax.scatter(
        X_final[:, i],
        X_final[:, j],
        s=10,
        alpha=0.30,
        label=r"ODE terminal samples"
    )

    add_mean_std_errorbar(
        ax,
        diag["emp_mean"][[i, j]],
        diag["emp_std"][[i, j]],
        label=r"empirical mean $\pm$ marginal std",
        marker="o",
    )

    add_mean_std_errorbar(
        ax,
        diag["ideal_mean"][[i, j]],
        diag["ideal_std"][[i, j]],
        label=r"ideal mean $\pm$ marginal std",
        marker="x",
    )

    add_mean_se_errorbar(
        ax,
        diag["emp_mean"][[i, j]],
        diag["se_mean"][[i, j]],
        label=r"empirical mean $\pm$ mean SE",
        marker="s",
    )

    ax.scatter(
        [y_fixed[i]],
        [y_fixed[j]],
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
        mean=prior_mean_2d,
        cov=prior_cov_2d,
        label_prefix=r"clean prior $N(\mu,\Sigma)$",
    )

    add_gaussian_contours(
        ax,
        mean=mean_2d,
        cov=cov_2d,
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

    add_radius_diagnostics_box(
        ax,
        samples=X_final,
        ideal_mean=post_mean,
        ideal_cov=post_cov,
        title=r"Mahalanobis radius (all $d$ dims)",
    )

    add_marginal_hists(
        ax_histx, ax_histy,
        samples=X_final,
        dims=dims,
        ideal_mean=post_mean,
        ideal_cov=post_cov,
        label="ODE terminal samples",
        color="tab:orange",
    )

    ax.set_xlabel(rf"$x_{{{i+1}}}$")
    ax.set_ylabel(rf"$x_{{{j+1}}}$")
    ax.legend(loc="upper right")
    fig.suptitle(rf"Fixed $y$: ODE terminal law vs exact $p(x\mid y)$ (dims {i+1},{j+1} of {d})")
    plt.savefig(f"{out_prefix}fixed_y_terminal_law.png", dpi=300)
    plt.show()

    plot_radius_sq_histogram(
        X_final, post_mean, post_cov,
        path=f"{out_prefix}fixed_y_r2_hist.png",
        title=rf"Fixed $y$: Mahalanobis $r^2$ vs $\chi^2_{{{d}}}$ (posterior check)",
    )

    print("\n================ Fixed y diagnostic ================")
    for key, value in diag.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {fmt_vec(value)}")
        else:
            print(f"{key}: {value}")

    print("\n---- radius diagnostics (whole-vector, all d dims) ----")
    for key, value in rdiag.items():
        if key in ("r2_samples", "r_samples"):
            continue
        print(f"{key}: {value}")

    wdiag = whitened_diagnostics(X_final, post_mean, post_cov)
    print(f"\n---- whitened per-axis diagnostics: max z = {wdiag['max_z']:.2f} | {wdiag['status']} ----")

    return {
        "y_fixed": y_fixed,
        "X_final": X_final,
        "post_mean": post_mean,
        "post_cov": post_cov,
        "diagnostics": diag,
        "radius_diagnostics": rdiag,
        "whitened_diagnostics": wdiag,
    }


def run_marginal(rng, d, mu, Sigma, s2, I, solve_particles, N_marginal=3000,
                  dims=(0, 1), out_prefix=""):
    """
    Draws Y ~ N(mu, Sigma + s^2 I), integrates the conditional ODE per-sample,
    and checks that the terminal marginal recovers the clean prior N(mu, Sigma).

    Diagnostics run over all d dimensions; the scatter/contour plot is a 2D
    snapshot on the coordinate pair `dims`.
    """
    i, j = dims

    Y_samples = rng.multivariate_normal(mu, Sigma + s2 * I, size=N_marginal)
    Z0_marginal = rng.normal(size=(N_marginal, d))

    _, X_marginal_path = solve_particles(Z0_marginal, Y_samples)
    X_marginal_final = X_marginal_path[-1]

    prior_samples = rng.multivariate_normal(mu, Sigma, size=N_marginal)

    diag = finite_sample_diagnostics(X_marginal_final, mu, Sigma)
    rdiag = radius_diagnostics(X_marginal_final, mu, Sigma)

    print("\n================ Marginal diagnostic ================")
    for key, value in diag.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {fmt_vec(value)}")
        else:
            print(f"{key}: {value}")

    print("\n---- radius diagnostics (whole-vector, all d dims) ----")
    for key, value in rdiag.items():
        if key in ("r2_samples", "r_samples"):
            continue
        print(f"{key}: {value}")

    wdiag = whitened_diagnostics(X_marginal_final, mu, Sigma)
    print(f"\n---- whitened per-axis diagnostics: max z = {wdiag['max_z']:.2f} | {wdiag['status']} ----")

    # ============================================================
    # Plot 3: marginal recovery diagnostic (2D snapshot on dims), with
    # marginal histograms of the ODE marginal samples vs the exact 1D
    # prior marginals on the top/right axes.
    # ============================================================

    mean_2d = np.array([mu[i], mu[j]])
    cov_2d = Sigma[np.ix_([i, j], [i, j])]

    fig, ax, ax_histx, ax_histy = make_joint_grid()

    ax.scatter(
        prior_samples[:, i],
        prior_samples[:, j],
        s=8,
        alpha=0.25,
        label=r"true prior samples"
    )

    ax.scatter(
        X_marginal_final[:, i],
        X_marginal_final[:, j],
        s=8,
        alpha=0.25,
        label=r"ODE marginal samples"
    )

    add_mean_std_errorbar(
        ax,
        diag["emp_mean"][[i, j]],
        diag["emp_std"][[i, j]],
        label=r"empirical mean $\pm$ marginal std",
        marker="o",
    )

    add_mean_std_errorbar(
        ax,
        diag["ideal_mean"][[i, j]],
        diag["ideal_std"][[i, j]],
        label=r"ideal mean $\pm$ marginal std",
        marker="x",
    )

    add_mean_se_errorbar(
        ax,
        diag["emp_mean"][[i, j]],
        diag["se_mean"][[i, j]],
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

    add_radius_diagnostics_box(
        ax,
        samples=X_marginal_final,
        ideal_mean=mu,
        ideal_cov=Sigma,
        title=r"Mahalanobis radius (all $d$ dims)",
    )

    add_gaussian_contours(
        ax,
        mean=mean_2d,
        cov=cov_2d,
        label_prefix=r"clean prior $p(x) = N(\mu,\Sigma)$",
    )

    add_marginal_hists(
        ax_histx, ax_histy,
        samples=X_marginal_final,
        dims=dims,
        ideal_mean=mu,
        ideal_cov=Sigma,
        label="ODE marginal samples",
        color="tab:orange",
    )

    ax.set_xlabel(rf"$x_{{{i+1}}}$")
    ax.set_ylabel(rf"$x_{{{j+1}}}$")
    ax.legend(loc="upper right")
    fig.suptitle(rf"Marginal check: $\int p(x\mid y)p(y)\,dy = p(x)$ (dims {i+1},{j+1} of {d})")
    plt.savefig(f"{out_prefix}marginal_recovery.png", dpi=300)
    plt.show()

    plot_radius_sq_histogram(
        X_marginal_final, mu, Sigma,
        path=f"{out_prefix}marginal_r2_hist.png",
        title=rf"Marginal recovery: Mahalanobis $r^2$ vs $\chi^2_{{{d}}}$",
    )

    return {
        "X_marginal_final": X_marginal_final,
        "diagnostics": diag,
        "radius_diagnostics": rdiag,
        "whitened_diagnostics": wdiag,
    }
