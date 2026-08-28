"""
3D weighted-backprojection (WBP) pseudoinverse for the CryoET channel (Haar-SO(3) mount +
fixed-axis tilt series -> 2D parallel-beam projection -> AWGN). This is the importable
filtered-backprojection module `data.py::build_warmup` calls to build a classical-reconstruction
warm start -- the 3D analogue of `experiments/cryoet_mnist/pseudoinverse.py`.

This file is the CANONICAL live copy of this geometry. `classical_recon_3d.py` carries an older
snapshot of the same `ramp_filter_2d` / `_rotate_raw` / `backproject` code plus a diagnostic
gallery/sweep CLI, but its module-level imports are stale (`load_mnist_volumes_3d`,
`build_observations_3d`, `forward_channel_3d`, and a `corruption`- rather than `rotation`-sourced
`sample_uniform_rotation_so3`) and importing it here would be circular besides -- so the math is
COPIED and rewired to the current module names, not imported. (CLAUDE.md: confirm live-vs-
snapshot before editing.)

Method: ramp-filter each 2D projection with the radial filter |k| = sqrt(kx^2 + ky^2)
(optionally Hann-tapered) -- the 3D central-slice theorem's 2D analogue of 2D CT's 1D Ram-Lak
filter, matching `corruption.rotate_3d` + `corruption.project_2d` exactly -- then backproject
(smear along depth, undo the known rotation, average over tilts).

POSE-BLIND WARM START. `backproject` still needs a rotation per projection, but
`data.py::build_warmup` does not observe the true ones -- it resamples its own tilt series from
the channel config and backprojects `build_observations`' `y` against that. So the warm start
is identity/label-blind AND pose-blind: `pseudoinverse()` only ever sees `(y, rotations)`, and
here `rotations` is a fresh draw, not the series that produced `y`. Unlike the 2D sibling this
is a real handicap rather than a harmless global rotation: a freshly resampled tilt series
differs from the true one by a fixed rotation A on the SPECIMEN side of every pose --
`R_i^fresh = R_i^true @ A`, with A a ratio of two independent Haar mounts. In backprojection
that A lands between `rotate(R_i^T, .)` and the depth-smear, where it does not commute, so it
scrambles the geometry rather than globally reorienting the volume. (2D has no separate mount:
SO(2) is abelian, left- and right-multiplication coincide, and there a consistently-wrong angle
offset merely rotates the recon -- see the 2D file's docstring.) The 3D warm start is therefore
only roughly right -- a symmetry-breaking seed for EM, not an upper bound w.r.t. pose.

DC-offset note: `corruption`'s y = project_2d(rotate_3d(x, R)) + noise carries a constant
pedestal of -vol_size -- extruded-MNIST background is -1 (not 0), `rotate_3d`'s shifted-frame
trick maps zero-padding to that -1, and summing D = vol_size depth slices of -1 is a flat -D on
every projection. `backproject` shifts each projection by +vol_size before filtering (recovering
the x+1 frame, where background is truly 0) and by -1 at the end. It also uses its own raw
zero-padded rotation helper (`_rotate_raw`) instead of `corruption.rotate_3d`, whose own
internal +1/-1 shift would double-count on the already-zero-background smears here.

Scale note: `resp` is peak-normalized (|k|/nyquist, not physical |k| dk) and tilts are combined
with a plain mean (not FBP's pi/N_angles), so the reconstructed *shape* is right and the
background lands near -1 (the ramp filter's exact DC null), but the ink *gain* is an unmodeled
O(1)-O(10) constant. `build_warmup` min-max-renormalizes each volume to the [-1, 1] / background
= -1 convention that the M-step interpolant and the E-step re-corruption both assume.
"""

import torch
import torch.nn.functional as F

from corruption import corruption_channel
from rotation import sample_tilt_series_rotations_so3


########################################################
# Filtering -- radial 2D ramp, the isotropic generalization of 2D CT's 1D Ram-Lak
########################################################

def ramp_filter_2d(images: torch.Tensor, filter_type: str = "hann") -> torch.Tensor:
    """
    Radial ramp filter |k| = sqrt(kx^2 + ky^2) (optionally Hann-tapered), applied to each 2D
    projection independently -- the standard weighted-backprojection filter for electron
    tomography, matching corruption.project_2d's single-axis integration after an SO(3)
    rotation. Exactly zero at k = 0 by construction, so the forward-channel DC pedestal (see the
    module docstring) is a no-op after filtering whether or not the caller shifted it away first.

    Args:
        images: (..., H, W)
    Returns:
        (..., H, W) filtered, same shape.
    """
    H, W = images.shape[-2:]
    ky = torch.fft.fftfreq(H, device=images.device, dtype=images.dtype)
    kx = torch.fft.fftfreq(W, device=images.device, dtype=images.dtype)
    kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = torch.sqrt(kyy ** 2 + kxx ** 2)
    nyquist = k_mag.max().clamp_min(1e-8)
    resp = k_mag / nyquist
    if filter_type == "hann":
        resp = resp * 0.5 * (1.0 + torch.cos(torch.pi * k_mag / nyquist))
    elif filter_type != "ramp":
        raise ValueError(f"Unknown filter_type: {filter_type!r}")
    spec = torch.fft.fft2(images, dim=(-2, -1)) * resp
    return torch.fft.ifft2(spec, dim=(-2, -1)).real


########################################################
# Backprojection -- the adjoint of corruption.rotate_3d + corruption.project_2d
########################################################

def _rotate_raw(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Plain affine rotation, zero-padded -- deliberately NOT corruption.rotate_3d's black-fill
    ([-1, 1]-background) convention: the smeared projections this operates on already live in a
    zero-background frame (module docstring's DC-offset note), where genuine zero-padding IS the
    correct out-of-canvas value.

    Args:
        x: (B, 1, D, H, W)
        R: (B, 3, 3)
    """
    zeros = torch.zeros(R.size(0), 3, 1, device=x.device, dtype=R.dtype)
    theta = torch.cat([R, zeros], dim=2)  # (B, 3, 4)
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    return F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="zeros")


def backproject(y: torch.Tensor, R: torch.Tensor, vol_size: int,
                filtered: bool, filter_type: str = "hann") -> torch.Tensor:
    """
    Reconstruct one canonical-frame volume from N known-rotation 2D projections. Unfiltered
    (filtered=False) gives the classical blurred backprojection; filtered gives sharp WBP.

    Args:
        y: (N, 1, H, W) projections -- RAW corruption.corruption_channel output for one tilt
            series. The DC-offset shift is applied internally; callers must NOT pre-shift.
        R: (N, 3, 3) the KNOWN rotation used to generate each projection.
        vol_size: D = H = W of the reconstructed canonical volume.
        filtered: apply the ramp filter (True) or plain backprojection (False).
        filter_type: passed to ramp_filter_2d when filtered=True.

    Returns:
        (1, 1, vol_size, vol_size, vol_size) reconstruction in the [-1, 1] convention -- shape
        right, ink gain an unmodeled O(1-10) constant (module docstring's scale note).
    """
    N, _, H, W = y.shape
    assert H == vol_size and W == vol_size, \
        f"projection shape {(H, W)} != (vol_size, vol_size) {(vol_size, vol_size)}"
    y_shifted = y[:, 0] + vol_size                                          # (N, H, W)
    if filtered:
        y_shifted = ramp_filter_2d(y_shifted, filter_type)
    smear = y_shifted[:, None, None, :, :].expand(-1, -1, vol_size, -1, -1)  # (N, 1, D, H, W)
    R_inv = R.transpose(-1, -2)                                             # R^{-1} = R^T
    recon = _rotate_raw(smear, R_inv).mean(dim=0, keepdim=True)            # (1, 1, D, H, W)
    return recon - 1.0


def pseudoinverse(y: torch.Tensor, rotations: torch.Tensor, filtered: bool = True,
                  filter_type: str = "hann") -> torch.Tensor:
    """
    Batched classical reconstruction: one weighted backprojection per tilt series. The entry
    point data.py::build_warmup calls -- see this module's docstring for the POSE-BLIND caveat.
    `backproject` does the per-series math; this just loops it over the batch (small B/T at
    MNIST scale, not worth vectorizing further).

    Args:
        y: (B, T, 1, H, W) RAW corruption.corruption_channel output -- one T-tilt series per
            volume. vol_size is inferred as W (backproject asserts H == W == vol_size).
        rotations: (B, T, 3, 3) the tilt-series rotations to backproject against -- ideally the
            SAME ones passed to corruption_channel (the __main__ smoke test does that);
            data.py::build_warmup instead feeds an independent resample, having no access to the
            true poses (module docstring's POSE-BLIND WARM START note).
        filtered: apply the ramp filter (True, sharp WBP) or plain backprojection (False).
        filter_type: passed to ramp_filter_2d when filtered=True.

    Returns:
        (B, 1, W, W, W) reconstruction in the [-1, 1] convention, one per series.
    """
    B, T, _, H, W = y.shape
    recons = [
        backproject(y[b], rotations[b], vol_size=W, filtered=filtered, filter_type=filter_type)
        for b in range(B)
    ]
    return torch.cat(recons, dim=0)


if __name__ == "__main__":
    # Standalone smoke test: reconstruct a few KNOWN-pose tilt series and report Pearson r plus
    # the raw-vs-GT amplitude gap (module docstring's scale note). classical_recon_3d.py is the
    # fuller diagnostic (qualitative gallery + quality-vs-views sweep); this is just a fast check
    # that the geometry and the shapes line up. CPU only -- no device autodetect, since the real
    # build_warmup path runs on CPU and torch.fft on MPS is a needless risk.
    import argparse
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Deferred (see the module-level NOTE): data.py imports `pseudoinverse` from this file.
    from data import Config_Dataset_MNIST, load_mnist_volumes

    parser = argparse.ArgumentParser(
        description="Smoke-test pseudoinverse() on KNOWN-pose tilt series")
    parser.add_argument("--digit_classes", type=int, nargs="+", default=[0, 1, 7])
    parser.add_argument("--n_images_per_class", type=int, default=2)
    parser.add_argument("--vol_size", type=int, default=32)
    parser.add_argument("--num_tilts", type=int, default=16)
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5)
    parser.add_argument("--noise_std", type=float, default=3.0)
    parser.add_argument("--filter_type", type=str, default="hann", choices=["hann", "ramp"])
    parser.add_argument("--no_filtered", dest="filtered", action="store_false")
    parser.add_argument("--out_dir", type=str, default="pseudoinverse_smoke")
    args = parser.parse_args()

    config = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class, vol_size=args.vol_size,
        digit_classes=args.digit_classes, num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg, noise_std=args.noise_std,
    )
    vol_gt = load_mnist_volumes(config)                        # (N, 1, V, V, V) in [-1, 1]
    N, V = vol_gt.size(0), config.vol_size

    tilt_increment_rad = config.tilt_increment_deg * torch.pi / 180.0
    rotations = sample_tilt_series_rotations_so3(N, config.num_tilts, tilt_increment_rad,
                                                tilt_axis=config.tilt_axis)
    y = corruption_channel(vol_gt, rotations=rotations, noise_std=config.noise_std)
    x_hat = pseudoinverse(y, rotations, filtered=args.filtered, filter_type=args.filter_type)

    g, r = vol_gt.reshape(N, -1), x_hat.reshape(N, -1)
    pear = torch.stack([torch.corrcoef(torch.stack([g[i], r[i]]))[0, 1] for i in range(N)])
    print(f"pseudoinverse smoke  |  N={N}  V={V}  T={config.num_tilts}  "
          f"filtered={args.filtered} ({args.filter_type})")
    print(f"  x_hat  min/mean/max  {x_hat.min():+.3f} / {x_hat.mean():+.3f} / {x_hat.max():+.3f}"
          f"   (raw WBP, pre-renorm)")
    print(f"  vol_gt min/mean/max  {vol_gt.min():+.3f} / {vol_gt.mean():+.3f} / {vol_gt.max():+.3f}")
    print(f"  corner-voxel bg      x_hat {x_hat[:, 0, 0, 0, 0].mean():+.3f}   "
          f"vol_gt {vol_gt[:, 0, 0, 0, 0].mean():+.3f}")
    print(f"  Pearson r (known pose): mean {pear.mean():.3f}  min {pear.min():.3f}  "
          f"max {pear.max():.3f}")

    def _mosaic(v):  # (n, 1, D, H, W) -> (n, H, 3W): xy | xz | yz sum-projections
        return torch.cat([v.sum(2), v.sum(3), v.sum(4)], dim=-1)[:, 0]

    gt_m, hat_m = _mosaic(vol_gt), _mosaic(x_hat)
    fig, axes = plt.subplots(2, N, figsize=(2.4 * N, 4.6), squeeze=False)
    axes[0, 0].set_ylabel("GT xy|xz|yz", fontsize=9)
    axes[1, 0].set_ylabel("pseudoinverse", fontsize=9)
    for j in range(N):
        for row, m in enumerate((gt_m, hat_m)):
            axes[row, j].imshow(m[j].numpy(), cmap="gray")
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
    fig.suptitle(f"pseudoinverse() KNOWN-pose smoke  |  r_mean={pear.mean():.3f}", fontsize=10)
    plt.tight_layout()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "smoke.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Mosaic -> {out_dir / 'smoke.png'}")
