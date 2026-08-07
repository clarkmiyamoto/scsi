"""
3D/CryoET analogue of tilt_series_difficulty.py -- see its module docstring for the full
rationale (span-vs-density, multi-acquisition averaging, class-identity check). Run at reduced
scope deliberately (fewer classes, coarser grid): 3D reconstructions are heavier, and a
single-axis SO(3) tilt series leaves a genuinely worse missing region than the 2D SO(2) case --
not just a wedge in one plane, but a whole missing CONE of directions around the fixed tilt
axis (the classic electron-tomography "missing wedge/cone" problem) -- so this script exists to
check whether the 2D script's "even quite poor per-acquisition reconstructions are still
variance-dominated and class-informative" conclusion actually transfers to 3D, not to assume it
does.
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from data import load_mnist_volumes_3d, build_observations_3d
from classical_recon_3d import backproject, compare

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def reconstruct_acquisition(x_i: torch.Tensor, n_tilts: int, tilt_increment_deg: float,
                            noise_std: float, tilt_axis, filter_type: str = "hann") -> torch.Tensor:
    y, R, _, _ = build_observations_3d(
        x_i, corruptions_per_object=1, n_tilts=n_tilts,
        tilt_increment_deg=tilt_increment_deg, noise_std=noise_std, tilt_axis=tilt_axis)
    return backproject(y, R, x_i.size(-1), filtered=True, filter_type=filter_type)


def span_density_grid(x_gt, spans_deg, n_tilts_list, noise_std, n_repeats, tilt_axis):
    n = x_gt.size(0)
    grid = torch.zeros(len(spans_deg), len(n_tilts_list))
    for si, span in enumerate(spans_deg):
        for ni, n_tilts in enumerate(n_tilts_list):
            incr = span / n_tilts
            rs = []
            for i in range(n):
                for _ in range(n_repeats):
                    recon = reconstruct_acquisition(x_gt[i:i + 1], n_tilts, incr, noise_std,
                                                     tilt_axis)
                    r, _ = compare(x_gt[i:i + 1], recon)
                    rs.append(r)
            grid[si, ni] = sum(rs) / len(rs)
            print(f"  span={span:5.1f}deg  n_tilts={n_tilts:3d}  incr={incr:5.2f}deg  "
                  f"-> mean r={grid[si, ni]:.3f}  (n={len(rs)})")
    return grid


def averaging_test(x_i, n_tilts, tilt_increment_deg, noise_std, K, tilt_axis):
    recons, rs_individual = [], []
    for _ in range(K):
        recon = reconstruct_acquisition(x_i, n_tilts, tilt_increment_deg, noise_std, tilt_axis)
        r, _ = compare(x_i, recon)
        rs_individual.append(r)
        recons.append(recon)
    recon_avg = torch.stack(recons, dim=0).mean(dim=0)
    r_avg, _ = compare(x_i, recon_avg)
    return sum(rs_individual) / K, r_avg


def _pearson_batch(a: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(-1)
    pool_flat = pool.reshape(pool.size(0), -1)
    a_c = a_flat - a_flat.mean()
    pool_c = pool_flat - pool_flat.mean(dim=1, keepdim=True)
    num = (pool_c * a_c[None, :]).sum(dim=1)
    den = pool_c.norm(dim=1) * a_c.norm()
    return num / den.clamp_min(1e-8)


def class_identity_test(pool_gt, pool_labels, n_tilts, tilt_increment_deg, noise_std, tilt_axis):
    n = pool_gt.size(0)
    correct = 0
    for i in range(n):
        recon = reconstruct_acquisition(pool_gt[i:i + 1], n_tilts, tilt_increment_deg,
                                        noise_std, tilt_axis)
        rs = _pearson_batch(recon, pool_gt)
        best = rs.argmax().item()
        if pool_labels[best].item() == pool_labels[i].item():
            correct += 1
    return correct / n


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D/CryoET analogue of tilt_series_difficulty.py, reduced scope. See "
                    "module docstring."
    )
    parser.add_argument("--spans_deg", type=float, nargs="+", default=[10, 30, 60, 90, 120])
    parser.add_argument("--n_tilts_list", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--digit_classes", type=int, nargs="+", default=[0, 3, 8])
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--K_averaging", type=int, default=6)
    parser.add_argument("--n_per_class_identity", type=int, default=2,
                        help="Test C: images per class in the labeled pool")
    parser.add_argument("--identity_classes", type=int, nargs="+", default=[0, 1, 3, 6, 8],
                        help="Test C: reduced class set (full 10-way NN search is heavier in "
                             "3D; 5 visually-distinct classes is enough to see above-chance "
                             "signal without the full grid's cost)")
    parser.add_argument("--vol_size", type=int, default=24,
                        help="Smaller than main_3d.py's default 32 -- keeps this reduced-scope "
                             "script fast; not meant to be the final word on absolute numbers")
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--tilt_axis", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--out_dir", type=str, default="classical_recon_eval")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        def _explicit(flag):
            return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

        if not _explicit("--spans_deg"):    args.spans_deg = [10, 60]
        if not _explicit("--n_tilts_list"): args.n_tilts_list = [8]
        if not _explicit("--vol_size"):     args.vol_size = 16
        if not _explicit("--n_repeats"):    args.n_repeats = 1
        if not _explicit("--K_averaging"):  args.K_averaging = 2

    print(f"Device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tilt_axis = tuple(args.tilt_axis)

    print("\n=== A. span x n_tilts quality grid (3D, single-axis tilt) ===")
    x_grid = load_mnist_volumes_3d(1, vol_size=args.vol_size, digit_classes=args.digit_classes,
                                   train=True).to(device)
    print(f"Grid pool: {x_grid.size(0)} volumes ({args.digit_classes}), vol_size={args.vol_size}, "
          f"{args.n_repeats} repeats/cell, noise_std={args.noise_std}")
    grid = span_density_grid(x_grid, args.spans_deg, args.n_tilts_list, args.noise_std,
                             args.n_repeats, tilt_axis)

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
    ax.set_title("3D filtered WBP quality: span vs. n_tilts (single-axis tilt)")
    plt.tight_layout()
    fig.savefig(out_dir / "span_density_grid_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid heatmap -> {out_dir / 'span_density_grid_3d.png'}")

    print("\n=== B. Multi-acquisition averaging test (3D) ===")
    test_configs = sorted(set(
        (s, n) for s in args.spans_deg[:2] for n in args.n_tilts_list[:2]
    ))[:3]
    for span, n_tilts in test_configs:
        incr = span / n_tilts
        mean_individual, mean_avg = [], []
        for i in range(x_grid.size(0)):
            r_ind, r_avg = averaging_test(x_grid[i:i + 1], n_tilts, incr, args.noise_std,
                                          args.K_averaging, tilt_axis)
            mean_individual.append(r_ind)
            mean_avg.append(r_avg)
        mi = sum(mean_individual) / len(mean_individual)
        ma = sum(mean_avg) / len(mean_avg)
        gap = ma - mi
        verdict = "variance-dominated (pooling helps)" if gap > 0.1 else \
            "bias-dominated (pooling doesn't help much)"
        print(f"  span={span:5.1f}deg n_tilts={n_tilts:3d} incr={incr:5.2f}deg  "
              f"mean single-acq r={mi:.3f}  K-averaged r={ma:.3f}  gap={gap:+.3f}  "
              f"-> {verdict}")

    print("\n=== C. Class-identity nearest-neighbor check (3D, reduced class set) ===")
    pool_gt = load_mnist_volumes_3d(args.n_per_class_identity, vol_size=args.vol_size,
                                    digit_classes=args.identity_classes, train=True).to(device)
    pool_labels = torch.tensor(args.identity_classes).repeat_interleave(
        args.n_per_class_identity)
    chance = 1.0 / len(args.identity_classes)
    print(f"Pool: {pool_gt.size(0)} volumes across {len(args.identity_classes)} classes "
          f"(chance level={chance:.2f}), noise_std={args.noise_std}")
    for span, n_tilts in test_configs:
        incr = span / n_tilts
        acc = class_identity_test(pool_gt, pool_labels, n_tilts, incr, args.noise_std, tilt_axis)
        print(f"  span={span:5.1f}deg n_tilts={n_tilts:3d} incr={incr:5.2f}deg  "
              f"class-match accuracy={acc:.2f}  (chance={chance:.2f})")

    print("\nDone.")
