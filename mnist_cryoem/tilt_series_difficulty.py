"""
Which tilt-series geometries (n_tilts, angular span) give classical reconstruction that's
"poor but not hopeless" as a seed for a possible classical-recon SCSI pretraining scheme? See
classical_recon.py's module docstring for why classical reconstruction is being evaluated here
at all (feasibility check, not itself part of SCSI). This script answers a narrower follow-up
question, reusing classical_recon.backproject/compare directly rather than duplicating them.

Raw reconstruction fidelity (Pearson r) alone doesn't answer "would SCSI still recover the true
prior" -- it conflates two different failure modes with very different consequences for a
pretraining scheme built on top:

  1. SPAN-limited failure (missing wedge): certain spatial-frequency directions are NEVER
     measured within one acquisition's angular range, however many tilts are packed into that
     range. This is a genuine information gap for a SINGLE acquisition. But every acquisition's
     wedge sits at a DIFFERENT random orientation (data.build_observations draws a fresh random
     start offset per acquisition) -- so across the pool of many acquisitions of similar-class
     digits, the missing directions are different each time. That's exactly the structure SCSI's
     E-step/M-step pooling is built to exploit: if the per-acquisition error is high-VARIANCE
     (random orientation) rather than a shared BIAS, pooling across acquisitions can plausibly
     average it out, and a classical-recon pretraining scheme built on individually-poor
     reconstructions might still work.

  2. DENSITY-limited failure (aliasing): not enough tilts within an otherwise-adequate span,
     causing streak artifacts. Adding tilts at fixed span fixes this directly -- it isn't the
     interesting case for "would SCSI recover," since it's fixable without touching the geometry
     budget's span at all.

Three measurements, none of which is raw fidelity alone:

  A. span x n_tilts quality grid (tilt_increment_deg = span/n_tilts at each cell) -- separates
     the two failure modes: flat rows (span-limited, adding tilts doesn't help) vs. flat columns
     (density-limited, span doesn't matter once dense enough).
  B. Multi-acquisition averaging test -- reconstructs the SAME digit from K independent
     (random-offset) acquisitions at a fixed geometry, then compares the mean per-acquisition
     fidelity against the fidelity of the AVERAGED reconstruction. If averaging recovers most of
     the per-acquisition loss, the degradation is variance (SCSI's pooling can plausibly absorb
     it). If the average plateaus near the individual quality, the error is shared across
     acquisitions and pooling won't fix it -- that IS the "too poor" boundary.
  C. Class-identity nearest-neighbor check -- even where pixel correlation is mediocre, does a
     reconstruction still best-match its own digit class among a labeled pool? Tests whether
     discriminative signal survives independent of raw fidelity.

Every reconstruction here uses TRUE (theta_star) angles, same upper-bound caveat as
classical_recon.py.
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from data import load_mnist_subset, build_observations
from classical_recon import backproject, compare

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


########################################################
# A. span x n_tilts grid
########################################################

def reconstruct_acquisition(x_i: torch.Tensor, n_tilts: int, tilt_increment_deg: float,
                            noise_std: float, filter_type: str = "hann") -> torch.Tensor:
    """One random-offset tilt-series acquisition -> filtered FBP reconstruction. Thin wrapper
    around data.build_observations + classical_recon.backproject, since every measurement in
    this file needs exactly this pair."""
    y, theta, _, _ = build_observations(
        x_i, corruptions_per_object=1, n_tilts=n_tilts,
        tilt_increment_deg=tilt_increment_deg, noise_std=noise_std)
    return backproject(y, theta, x_i.size(-1), filtered=True, filter_type=filter_type)


def span_density_grid(x_gt: torch.Tensor, spans_deg: list[float], n_tilts_list: list[int],
                      noise_std: float, n_repeats: int) -> torch.Tensor:
    """
    Mean Pearson r over (digits x n_repeats random-offset draws) at every (span, n_tilts) cell,
    with tilt_increment_deg = span / n_tilts recomputed per cell -- so n_tilts varies at FIXED
    span (isolating density) and span varies at FIXED n_tilts (isolating coverage).

    Returns:
        (len(spans_deg), len(n_tilts_list)) grid of mean Pearson r
    """
    n = x_gt.size(0)
    grid = torch.zeros(len(spans_deg), len(n_tilts_list))
    for si, span in enumerate(spans_deg):
        for ni, n_tilts in enumerate(n_tilts_list):
            incr = span / n_tilts
            rs = []
            for i in range(n):
                for _ in range(n_repeats):
                    recon = reconstruct_acquisition(x_gt[i:i + 1], n_tilts, incr, noise_std)
                    r, _ = compare(x_gt[i:i + 1], recon)
                    rs.append(r)
            grid[si, ni] = sum(rs) / len(rs)
            print(f"  span={span:5.1f}deg  n_tilts={n_tilts:3d}  incr={incr:5.2f}deg  "
                  f"-> mean r={grid[si, ni]:.3f}  (n={len(rs)})")
    return grid


########################################################
# B. Multi-acquisition averaging test
########################################################

def averaging_test(x_i: torch.Tensor, n_tilts: int, tilt_increment_deg: float,
                   noise_std: float, K: int) -> tuple[float, float]:
    """
    Reconstructs the SAME digit from K independent random-offset acquisitions at one geometry.
    Returns (mean individual r, r of the K-averaged reconstruction) -- the gap between them is
    the diagnostic: large gap = variance-dominated (pooling helps), small gap = bias-dominated
    (pooling doesn't).
    """
    recons = []
    rs_individual = []
    for _ in range(K):
        recon = reconstruct_acquisition(x_i, n_tilts, tilt_increment_deg, noise_std)
        r, _ = compare(x_i, recon)
        rs_individual.append(r)
        recons.append(recon)
    recon_avg = torch.stack(recons, dim=0).mean(dim=0)
    r_avg, _ = compare(x_i, recon_avg)
    return sum(rs_individual) / K, r_avg


########################################################
# C. Class-identity nearest-neighbor check
########################################################

def _pearson_batch(a: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """a: (1,1,H,W), pool: (N,1,H,W) -> (N,) Pearson r between a and each pool member."""
    a_flat = a.reshape(-1)
    pool_flat = pool.reshape(pool.size(0), -1)
    a_c = a_flat - a_flat.mean()
    pool_c = pool_flat - pool_flat.mean(dim=1, keepdim=True)
    num = (pool_c * a_c[None, :]).sum(dim=1)
    den = pool_c.norm(dim=1) * a_c.norm()
    return num / den.clamp_min(1e-8)


def class_identity_test(pool_gt: torch.Tensor, pool_labels: torch.Tensor, n_tilts: int,
                        tilt_increment_deg: float, noise_std: float) -> float:
    """
    For each pool digit, reconstruct one acquisition and find its nearest neighbor (by Pearson
    r) among the WHOLE labeled pool (the true source is a valid, usually-best candidate).
    Returns the fraction whose nearest neighbor shares the true digit's class -- chance level is
    1/n_classes.
    """
    n = pool_gt.size(0)
    correct = 0
    for i in range(n):
        recon = reconstruct_acquisition(pool_gt[i:i + 1], n_tilts, tilt_increment_deg, noise_std)
        rs = _pearson_batch(recon, pool_gt)
        best = rs.argmax().item()
        if pool_labels[best].item() == pool_labels[i].item():
            correct += 1
    return correct / n


########################################################
# CLI
########################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Characterizes which tilt-series geometries give classical reconstruction "
                    "that's poor but plausibly still recoverable by SCSI's pooling. See module "
                    "docstring."
    )
    parser.add_argument("--spans_deg", type=float, nargs="+", default=[30, 60, 90, 120, 180])
    parser.add_argument("--n_tilts_list", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--n_images_per_class", type=int, default=2,
                        help="Grid test (A): images per class in the digit pool")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="Grid test (A): independent random-offset draws averaged per cell")
    parser.add_argument("--K_averaging", type=int, default=8,
                        help="Test (B): number of independent acquisitions pooled")
    parser.add_argument("--n_digits_averaging", type=int, default=5,
                        help="Test (B): how many distinct digits to run the averaging test on")
    parser.add_argument("--n_per_class_identity", type=int, default=5,
                        help="Test (C): images per class in the labeled pool")
    parser.add_argument("--noise_std", type=float, default=3.0)
    parser.add_argument("--out_dir", type=str, default="classical_recon_eval")
    parser.add_argument("--debug", action="store_true", help="Tiny run: quick smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        def _explicit(flag):
            return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

        if not _explicit("--spans_deg"):              args.spans_deg = [30, 90]
        if not _explicit("--n_tilts_list"):            args.n_tilts_list = [8, 16]
        if not _explicit("--n_images_per_class"):      args.n_images_per_class = 1
        if not _explicit("--n_repeats"):               args.n_repeats = 1
        if not _explicit("--K_averaging"):             args.K_averaging = 2
        if not _explicit("--n_digits_averaging"):      args.n_digits_averaging = 2
        if not _explicit("--n_per_class_identity"):    args.n_per_class_identity = 1

    print(f"Device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── A. span x n_tilts grid ────────────────────────────────────────────
    print("\n=== A. span x n_tilts quality grid ===")
    x_grid = load_mnist_subset(args.n_images_per_class, digit_classes=None,
                               train=True).to(device)
    print(f"Grid pool: {x_grid.size(0)} digits, {args.n_repeats} repeats/cell, "
          f"noise_std={args.noise_std}")
    grid = span_density_grid(x_grid, args.spans_deg, args.n_tilts_list, args.noise_std,
                             args.n_repeats)

    fig, ax = plt.subplots(figsize=(1.6 * len(args.n_tilts_list) + 2,
                                    0.9 * len(args.spans_deg) + 2))
    im = ax.imshow(grid.numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(args.n_tilts_list)))
    ax.set_xticklabels(args.n_tilts_list)
    ax.set_yticks(range(len(args.spans_deg)))
    ax.set_yticklabels([f"{s:g}" for s in args.spans_deg])
    ax.set_xlabel("n_tilts")
    ax.set_ylabel("span (deg)")
    for si in range(len(args.spans_deg)):
        for ni in range(len(args.n_tilts_list)):
            ax.text(ni, si, f"{grid[si, ni]:.2f}", ha="center", va="center",
                    color="white" if grid[si, ni] < 0.7 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean Pearson r")
    ax.set_title("Filtered FBP quality: span vs. n_tilts (flat rows = span-limited,\n"
                 "flat columns = density-limited)")
    plt.tight_layout()
    fig.savefig(out_dir / "span_density_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid heatmap -> {out_dir / 'span_density_grid.png'}")

    # ── B. Multi-acquisition averaging test ──────────────────────────────
    print("\n=== B. Multi-acquisition averaging test ===")
    print(f"K={args.K_averaging} independent acquisitions/digit, "
          f"{args.n_digits_averaging} digits, noise_std={args.noise_std}")
    x_avg = load_mnist_subset(1, digit_classes=list(range(min(10, args.n_digits_averaging))),
                              train=True).to(device)[:args.n_digits_averaging]
    # Test the two lowest-span cells actually computed in the grid (the plausibly-poor region)
    test_configs = sorted(set(
        (s, n) for s in args.spans_deg[:2] for n in args.n_tilts_list[:2]
    ))[:3]
    for span, n_tilts in test_configs:
        incr = span / n_tilts
        mean_individual, mean_avg = [], []
        for i in range(x_avg.size(0)):
            r_ind, r_avg = averaging_test(x_avg[i:i + 1], n_tilts, incr, args.noise_std,
                                          args.K_averaging)
            mean_individual.append(r_ind)
            mean_avg.append(r_avg)
        mi = sum(mean_individual) / len(mean_individual)
        ma = sum(mean_avg) / len(mean_avg)
        gap = ma - mi
        verdict = "variance-dominated (pooling helps)" if gap > 0.1 else \
            "bias-dominated (pooling doesn't help much)"
        print(f"  span={span:5.1f}deg n_tilts={n_tilts:3d} incr={incr:5.2f}deg  "
              f"mean single-acq r={mi:.3f}  K-averaged r={ma:.3f}  "
              f"gap={gap:+.3f}  -> {verdict}")

    # ── C. Class-identity nearest-neighbor check ──────────────────────────
    print("\n=== C. Class-identity nearest-neighbor check ===")
    digit_classes = list(range(10))
    pool_gt = load_mnist_subset(args.n_per_class_identity, digit_classes=digit_classes,
                                train=True).to(device)
    pool_labels = torch.tensor(digit_classes).repeat_interleave(args.n_per_class_identity)
    chance = 1.0 / len(digit_classes)
    print(f"Pool: {pool_gt.size(0)} digits across {len(digit_classes)} classes "
          f"(chance level={chance:.2f}), noise_std={args.noise_std}")
    for span, n_tilts in test_configs:
        incr = span / n_tilts
        acc = class_identity_test(pool_gt, pool_labels, n_tilts, incr, args.noise_std)
        print(f"  span={span:5.1f}deg n_tilts={n_tilts:3d} incr={incr:5.2f}deg  "
              f"class-match accuracy={acc:.2f}  (chance={chance:.2f})")

    print("\nDone.")
