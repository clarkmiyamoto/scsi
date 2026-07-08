import numpy as np
import matplotlib.pyplot as plt


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