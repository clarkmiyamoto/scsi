import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import chi2, norm


# ============================================================
# Helpers
# ============================================================

def empirical_mean_std(samples):
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    return mean, std


def ideal_mean_std(mean, cov):
    std = np.sqrt(np.diag(cov))
    return mean, std


def fmt_vec(v):
    return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"


def finite_sample_diagnostics(samples, ideal_mean, ideal_cov):
    """
    Coordinatewise finite-sample diagnostics for a Gaussian target.

    For N independent samples:
        std(mean_i) ≈ sigma_i / sqrt(N)
        std(sample_std_i) ≈ sigma_i / sqrt(2(N-1))
    """
    N = samples.shape[0]

    emp_mean, emp_std = empirical_mean_std(samples)
    ideal_mean, ideal_std = ideal_mean_std(ideal_mean, ideal_cov)

    se_mean = ideal_std / np.sqrt(N)
    se_std = ideal_std / np.sqrt(2 * (N - 1))

    z_mean = np.abs(emp_mean - ideal_mean) / se_mean
    z_std = np.abs(emp_std - ideal_std) / se_std

    max_z = max(np.max(z_mean), np.max(z_std))

    if max_z < 2.0:
        status = "OK: within ~2 SE"
    elif max_z < 3.0:
        status = "WARN: 2-3 SE"
    else:
        status = "FAIL?: >3 SE"

    return {
        "N": N,
        "emp_mean": emp_mean,
        "emp_std": emp_std,
        "ideal_mean": ideal_mean,
        "ideal_std": ideal_std,
        "se_mean": se_mean,
        "se_std": se_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "max_z": max_z,
        "status": status,
    }


def add_diagnostic_stats_box(ax, samples, ideal_mean, ideal_cov, title=""):
    diag = finite_sample_diagnostics(samples, ideal_mean, ideal_cov)

    text = (
        f"{title}\n"
        f"N = {diag['N']}\n"
        f"emp mean   = {fmt_vec(diag['emp_mean'])}\n"
        f"ideal mean = {fmt_vec(diag['ideal_mean'])}\n"
        f"mean SE    = {fmt_vec(diag['se_mean'])}\n"
        f"mean z     = {fmt_vec(diag['z_mean'])}\n"
        f"\n"
        f"emp std    = {fmt_vec(diag['emp_std'])}\n"
        f"ideal std  = {fmt_vec(diag['ideal_std'])}\n"
        f"std SE     = {fmt_vec(diag['se_std'])}\n"
        f"std z      = {fmt_vec(diag['z_std'])}\n"
        f"\n"
        f"max z = {diag['max_z']:.2f} | {diag['status']}"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round", alpha=0.88),
    )


def add_mean_std_errorbar(ax, mean, std, label, marker="o"):
    """
    Shows mean ± marginal std.
    This visualizes the scale of the distribution.
    """
    ax.errorbar(
        mean[0],
        mean[1],
        xerr=std[0],
        yerr=std[1],
        fmt=marker,
        capsize=4,
        markersize=8,
        label=label,
    )


def add_mean_se_errorbar(ax, mean, se, label, marker="s"):
    """
    Shows mean ± standard error of the mean.
    This visualizes finite-sample uncertainty in the empirical mean.
    """
    ax.errorbar(
        mean[0],
        mean[1],
        xerr=se[0],
        yerr=se[1],
        fmt=marker,
        capsize=4,
        markersize=8,
        label=label,
    )

def chi2_moments(d):
    """Mean and variance of a chi^2_d distribution."""
    return float(d), float(2 * d)


def chi_moments(d):
    """
    Mean and variance of a chi_d distribution (the law of sqrt(chi^2_d)),
    via E[chi_d] = sqrt(2) Gamma((d+1)/2) / Gamma(d/2).
    """
    mean = math.sqrt(2.0) * math.exp(gammaln((d + 1) / 2.0) - gammaln(d / 2.0))
    var = d - mean ** 2
    return float(mean), float(var)


def mahalanobis_r2(samples, mean, cov):
    """Squared Mahalanobis radius r_i^2 = (x_i-mean)^T cov^{-1} (x_i-mean)."""
    diff = samples - mean
    Cinv = np.linalg.inv(cov)
    return np.einsum("ni,ij,nj->n", diff, Cinv, diff)


def radius_diagnostics(samples, ideal_mean, ideal_cov):
    """
    Whole-vector counterpart to finite_sample_diagnostics's per-coordinate
    z-scores. For X ~ N(ideal_mean, ideal_cov):
        r^2 = (X-mean)^T cov^{-1} (X-mean) ~ chi^2_d
        r   = sqrt(r^2)                    ~ chi_d
    A single scalar summary of the full d-dim fit (mean, variances, AND
    correlations all feed into r^2 via cov^{-1}), so it stays informative as
    d grows, unlike per-coordinate z-scores which only ever look at 2 axes
    once plotted, or a raw covariance-Frobenius-norm error which has no
    natural finite-sample scale to compare against.
    """
    d = ideal_mean.shape[0]
    N = samples.shape[0]

    r2 = mahalanobis_r2(samples, ideal_mean, ideal_cov)
    r = np.sqrt(r2)

    r2_mean_true, r2_var_true = chi2_moments(d)
    r_mean_true, r_var_true = chi_moments(d)

    r2_mean_emp = float(r2.mean())
    r_mean_emp = float(r.mean())

    se_r2 = math.sqrt(r2_var_true / N)
    se_r = math.sqrt(r_var_true / N)

    z_r2 = abs(r2_mean_emp - r2_mean_true) / se_r2
    z_r = abs(r_mean_emp - r_mean_true) / se_r

    max_z = max(z_r2, z_r)
    if max_z < 2.0:
        status = "OK: within ~2 SE"
    elif max_z < 3.0:
        status = "WARN: 2-3 SE"
    else:
        status = "FAIL?: >3 SE"

    return {
        "d": d,
        "N": N,
        "r2_samples": r2,
        "r_samples": r,
        "r2_mean_emp": r2_mean_emp,
        "r2_mean_true": r2_mean_true,
        "se_r2": se_r2,
        "z_r2": z_r2,
        "r_mean_emp": r_mean_emp,
        "r_mean_true": r_mean_true,
        "se_r": se_r,
        "z_r": z_r,
        "max_z": max_z,
        "status": status,
    }


def whitened_diagnostics(samples, ideal_mean, ideal_cov):
    """
    Per-axis z-scores after whitening by the ideal covariance's Cholesky
    factor: w = L^{-1}(x - mean), which should be ~ N(0, I_d). Unlike
    finite_sample_diagnostics on the raw coordinates, this is sensitive to
    off-diagonal (correlation) errors too: an anisotropic Gaussian can have
    perfect marginal per-axis variances while still having the wrong
    correlation structure, and that shows up here but not in raw z_std.
    """
    L = np.linalg.cholesky(ideal_cov)
    w = np.linalg.solve(L, (samples - ideal_mean).T).T
    d = ideal_mean.shape[0]
    return finite_sample_diagnostics(w, np.zeros(d), np.eye(d))


def add_radius_diagnostics_box(ax, samples, ideal_mean, ideal_cov, title=""):
    diag = radius_diagnostics(samples, ideal_mean, ideal_cov)

    text = (
        f"{title}\n"
        f"N = {diag['N']}, d = {diag['d']}\n"
        f"E[r^2]  emp={diag['r2_mean_emp']:.3f}  true={diag['r2_mean_true']:.3f}  "
        f"SE={diag['se_r2']:.3f}  z={diag['z_r2']:.2f}\n"
        f"E[r]    emp={diag['r_mean_emp']:.3f}  true={diag['r_mean_true']:.3f}  "
        f"SE={diag['se_r']:.3f}  z={diag['z_r']:.2f}\n"
        f"max z = {diag['max_z']:.2f} | {diag['status']}"
    )

    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round", alpha=0.88, facecolor="lightyellow"),
    )


def plot_radius_sq_histogram(samples, ideal_mean, ideal_cov, path, title=""):
    """
    Histogram of the empirical squared Mahalanobis radius r^2 against the
    theoretical chi^2_d density. A whole-vector goodness-of-fit check that
    stays informative regardless of d, unlike a 2D scatter snapshot whose
    view of the distribution shrinks to 2 of d axes as d grows.
    """
    d = ideal_mean.shape[0]
    r2 = mahalanobis_r2(samples, ideal_mean, ideal_cov)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r2, bins=60, density=True, alpha=0.6, label=r"empirical $r^2$")

    xs = np.linspace(0, max(float(r2.max()), chi2.ppf(0.999, d)), 400)
    ax.plot(xs, chi2.pdf(xs, d), color="crimson", linewidth=2, label=rf"$\chi^2_{{{d}}}$ pdf")

    ax.set_xlabel(r"$r^2 = (x-\mu)^\top \Sigma^{-1} (x-\mu)$")
    ax.set_ylabel("density")
    ax.set_title(title or rf"Mahalanobis $r^2$ vs $\chi^2_{{{d}}}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_joint_grid(figsize=(9, 8)):
    """
    Joint-plot style layout: a main 2D axis plus marginal histogram axes on
    top (x) and right (y), sharing the main axis's data ranges via
    sharex/sharey.

    Equal aspect is deliberately not applied to the main axis for this
    layout: forcing equal x/y data scaling resizes the main axis's box to
    keep circles circular, but the marginal axes' boxes are fixed by the
    gridspec and don't resize to match, which throws their alignment with
    the main scatter off.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.10, right=0.95, bottom=0.08, top=0.90,
        wspace=0.05, hspace=0.05,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    return fig, ax_main, ax_histx, ax_histy


def add_marginal_hists(ax_histx, ax_histy, samples, dims, ideal_mean, ideal_cov,
                        label="ODE flow samples", color=None, bins=40, alpha=0.55):
    """
    Draws 1D histograms of `samples` along the two plotted coordinates
    `dims` on the top/right marginal axes (see make_joint_grid), overlaid
    with the exact 1D Gaussian marginal N(ideal_mean[k], ideal_cov[k,k]) for
    each coordinate k. A coordinate projection of a joint Gaussian is
    exactly Gaussian, so this curve is the true target density for that
    axis, not an approximation of it.
    """
    i, j = dims
    x = np.asarray(samples)[:, i]
    y = np.asarray(samples)[:, j]

    ax_histx.hist(x, bins=bins, density=True, alpha=alpha, color=color, label=label)
    ax_histy.hist(y, bins=bins, density=True, alpha=alpha, color=color,
                  orientation="horizontal", label=label)

    mean_i, std_i = float(ideal_mean[i]), float(np.sqrt(ideal_cov[i, i]))
    mean_j, std_j = float(ideal_mean[j]), float(np.sqrt(ideal_cov[j, j]))

    xs = np.linspace(min(x.min(), mean_i - 4 * std_i), max(x.max(), mean_i + 4 * std_i), 300)
    ys = np.linspace(min(y.min(), mean_j - 4 * std_j), max(y.max(), mean_j + 4 * std_j), 300)

    ax_histx.plot(xs, norm.pdf(xs, mean_i, std_i), color="crimson", linewidth=1.8, label="true marginal")
    ax_histy.plot(norm.pdf(ys, mean_j, std_j), ys, color="crimson", linewidth=1.8, label="true marginal")

    ax_histx.set_ylabel("density")
    ax_histy.set_xlabel("density")
    ax_histx.legend(fontsize=7, loc="upper right")


def add_gaussian_contours(
    ax,
    mean,
    cov,
    radii=(1.0, 2.0),
    label_prefix=r"clean prior",
    linewidth=1.5,
    linestyle="--",
    color="red"
):
    """
    Adds Mahalanobis-radius contours for a 2D Gaussian.

    The contours are:
        (x - mean)^T cov^{-1} (x - mean) = r^2

    In 2D, these enclose probability mass:
        r = 1 -> about 39%
        r = 2 -> about 86%
        r = 3 -> about 99%
    """
    eigvals, eigvecs = np.linalg.eigh(cov)

    theta = np.linspace(0, 2 * np.pi, 300)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)

    for j, r in enumerate(radii):
        ellipse = mean[:, None] + eigvecs @ np.diag(np.sqrt(eigvals)) @ (r * circle)

        label = None
        if j == 0:
            label = rf"{label_prefix}"

        ax.plot(
            ellipse[0],
            ellipse[1],
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            color=color
        )