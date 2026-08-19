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


def psd_eig(cov, rtol=None):
    """
    Symmetric eigendecomposition of a PSD covariance with a rank tolerance.

    Returns (eigvals, eigvecs, keep, rank) where `keep` is the boolean mask of
    eigenvalues above tolerance (the covariance's range) and rank = keep.sum().
    Eigenvalues at/below tolerance -- the null space of a rank-deficient
    (singular) covariance -- are dropped. The default tolerance follows
    numpy.linalg.matrix_rank: tol = max_eigval * max(cov.shape) * eps.
    """
    cov = np.asarray(cov, dtype=float)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)  # kill tiny negative eigenvalues from round-off
    wmax = float(w.max()) if w.size else 0.0
    if rtol is None:
        tol = wmax * max(cov.shape) * np.finfo(w.dtype).eps
    else:
        tol = rtol * wmax
    keep = w > tol
    return w, V, keep, int(keep.sum())


def effective_rank(cov):
    """Numerical rank of a covariance (eigenvalues above tolerance, see psd_eig)."""
    return psd_eig(cov)[3]


def axis_aligned_rank_reduce(Sigma_full, rank):
    """
    Axis-aligned rank-`rank` reduction of a full-rank SPD covariance: keep the
    top-left rank x rank principal block (still SPD) and zero the trailing
    d-rank rows/columns, turning those coordinate axes into exact point masses.
    The result has numerical rank `rank` with its null space aligned to the
    last d-rank coordinate axes, so a 2D snapshot pairing an active coord
    (< rank) with a null coord (>= rank) shows one non-singular and one
    singular axis. rank == d returns Sigma_full unchanged.
    """
    Sigma_full = np.asarray(Sigma_full, dtype=float)
    d = Sigma_full.shape[0]
    if not (1 <= rank <= d):
        raise ValueError(f"rank must be in [1, {d}], got {rank}")
    if rank == d:
        return Sigma_full.copy()
    Sigma = np.zeros_like(Sigma_full)
    Sigma[:rank, :rank] = Sigma_full[:rank, :rank]
    return Sigma


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

    # Zero-variance axes (a singular covariance's point-mass coordinates) have
    # se = 0, so their z-scores are undefined. There is no finite-sample
    # fluctuation to test on a deterministic coordinate, so we report z = 0
    # there rather than dividing by zero (inf/nan).
    nz = ideal_std > 1e-10 * max(float(ideal_std.max()), 1.0)

    se_mean = np.zeros_like(ideal_std)
    se_std = np.zeros_like(ideal_std)
    se_mean[nz] = ideal_std[nz] / np.sqrt(N)
    se_std[nz] = ideal_std[nz] / np.sqrt(2 * (N - 1))

    z_mean = np.zeros_like(ideal_std)
    z_std = np.zeros_like(ideal_std)
    z_mean[nz] = np.abs(emp_mean - ideal_mean)[nz] / se_mean[nz]
    z_std[nz] = np.abs(emp_std - ideal_std)[nz] / se_std[nz]

    max_z = max(np.max(z_mean), np.max(z_std)) if nz.any() else 0.0

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
    """
    Squared Mahalanobis radius restricted to the covariance's range.

    For a full-rank cov this is the usual (x-mean)^T cov^{-1} (x-mean). For a
    rank-deficient (singular) cov, inv(cov) is undefined -- and np.linalg.inv
    silently returns a garbage inverse rather than raising -- so we use the
    range-restricted pseudo-inverse: project x-mean onto the r eigenvectors
    with nonzero eigenvalue and scale by 1/eigval. The result is ~ chi^2_r
    where r = rank(cov), the correct reference for a degenerate Gaussian
    supported on an r-dimensional subspace.
    """
    w, V, keep, r = psd_eig(cov)
    diff = np.asarray(samples) - mean
    proj = diff @ V[:, keep]  # (N, r) coordinates in the range basis
    return np.einsum("nk,k,nk->n", proj, 1.0 / w[keep], proj)


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
    rank = effective_rank(ideal_cov)
    N = samples.shape[0]

    r2 = mahalanobis_r2(samples, ideal_mean, ideal_cov)
    r = np.sqrt(r2)

    # For a rank-deficient covariance the radius lives in an r-dim subspace, so
    # the reference law is chi^2_rank / chi_rank, not chi^2_d.
    r2_mean_true, r2_var_true = chi2_moments(rank)
    r_mean_true, r_var_true = chi_moments(rank)

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
        "rank": rank,
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
    Per-axis z-scores after whitening within the ideal covariance's range:
    w = diag(1/sqrt(eigval)) V^T (x - mean) over the r eigenvectors with
    nonzero eigenvalue, which should be ~ N(0, I_r). Unlike
    finite_sample_diagnostics on the raw coordinates, this is sensitive to
    off-diagonal (correlation) errors too: an anisotropic Gaussian can have
    perfect marginal per-axis variances while still having the wrong
    correlation structure, and that shows up here but not in raw z_std.

    Eigen-whitening (rather than the Cholesky factor) is used so this also
    works for a rank-deficient (singular) covariance, where Cholesky fails:
    the null directions are simply excluded and r < d.
    """
    w_eig, V, keep, r = psd_eig(ideal_cov)
    proj = (np.asarray(samples) - ideal_mean) @ V[:, keep]  # (N, r)
    w = proj / np.sqrt(w_eig[keep])                         # ~ N(0, I_r)
    return finite_sample_diagnostics(w, np.zeros(r), np.eye(r))


def add_radius_diagnostics_box(ax, samples, ideal_mean, ideal_cov, title=""):
    diag = radius_diagnostics(samples, ideal_mean, ideal_cov)

    text = (
        f"{title}\n"
        f"N = {diag['N']}, d = {diag['d']}, rank = {diag['rank']}\n"
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
    r = effective_rank(ideal_cov)
    r2 = mahalanobis_r2(samples, ideal_mean, ideal_cov)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r2, bins=60, density=True, alpha=0.6, label=r"empirical $r^2$")

    xs = np.linspace(0, max(float(r2.max()), chi2.ppf(0.999, r)), 400)
    ax.plot(xs, chi2.pdf(xs, r), color="crimson", linewidth=2, label=rf"$\chi^2_{{{r}}}$ pdf")

    ax.set_xlabel(r"$r^2 = (x-\mu)^\top \Sigma^{-1} (x-\mu)$")
    ax.set_ylabel("density")
    ax.set_title(title or rf"Mahalanobis $r^2$ vs $\chi^2_{{{r}}}$")
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

    # A singular covariance can make a plotted coordinate a point mass
    # (std == 0); its true marginal is then a Dirac delta at the mean, not a
    # Gaussian. Drawing norm.pdf(std=0) would divide by zero, so we mark the
    # delta with a line at the mean instead.
    tol = 1e-8
    if std_i > tol:
        xs = np.linspace(min(x.min(), mean_i - 4 * std_i), max(x.max(), mean_i + 4 * std_i), 300)
        ax_histx.plot(xs, norm.pdf(xs, mean_i, std_i), color="crimson", linewidth=1.8, label="true marginal")
    else:
        ax_histx.axvline(mean_i, color="crimson", linewidth=1.8, linestyle="--",
                         label=r"true marginal ($\delta$)")

    if std_j > tol:
        ys = np.linspace(min(y.min(), mean_j - 4 * std_j), max(y.max(), mean_j + 4 * std_j), 300)
        ax_histy.plot(norm.pdf(ys, mean_j, std_j), ys, color="crimson", linewidth=1.8, label="true marginal")
    else:
        ax_histy.axhline(mean_j, color="crimson", linewidth=1.8, linestyle="--",
                         label=r"true marginal ($\delta$)")

    ax_histx.set_ylabel("density")
    ax_histy.set_xlabel("density")
    ax_histx.legend(fontsize=7, loc="upper right")


def equalize_degenerate_axis(ax, dims, ideal_mean, ideal_cov, tol=1e-6):
    """
    When a plotted coordinate is (near-)singular under `ideal_cov` (variance
    ~0), the samples collapse to a point mass and matplotlib auto-zooms that
    axis into the tiny numerical residual, which makes the measure read as a 2D
    blob instead of a line. Rescale the degenerate axis to the same span as the
    non-degenerate one (centered at its mean) so a rank-deficient measure shows
    one axis with spread and one collapsed to a thin line. No-op when neither
    or both plotted coordinates are degenerate.
    """
    i, j = dims
    deg_i = ideal_cov[i, i] <= tol
    deg_j = ideal_cov[j, j] <= tol
    if deg_i == deg_j:
        return
    if deg_i:  # x-axis (dim i) is the singular one; scale it to the y span
        half = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, ideal_mean[i] - half), max(hi, ideal_mean[i] + half))
    else:      # y-axis (dim j) is the singular one; scale it to the x span
        half = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, ideal_mean[j] - half), max(hi, ideal_mean[j] + half))


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
    # A singular covariance has a zero (or tiny-negative from round-off)
    # eigenvalue; clip so sqrt is real. That axis then has zero radius, so the
    # contour collapses to a line segment -- the visual signature of the
    # singular direction.
    eigvals = np.clip(eigvals, 0.0, None)

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