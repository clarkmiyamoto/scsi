"""
Classical (non-learned) tomographic reconstruction for the 3D/CryoET channel (SO(3) rotation
-> 2D projection -> AWGN), as a pre-implementation feasibility check for a possible
classical-reconstruction-based pretraining scheme -- the 3D analogue of classical_recon.py,
same motivation: could a classical inversion of a real tilt series stand in for pretrain_3d.py's
GT-supervised x_hat target? Standalone diagnostic script: it doesn't touch
si.py/scsi.py/em.py/model.py, and isn't wired into the SCSI algorithm anywhere.

Method: weighted backprojection (WBP), the classical electron-tomography reconstruction
algorithm -- ramp-filter each 2D projection (the isotropic 2D generalization of 2D CT's 1D
Ram-Lak filter, matching corruption.py's rotate_3d + project_2d parallel-beam geometry exactly),
then backproject (smear along the depth axis, undo the known rotation, average). Re-derives this
codebase's own geometry directly rather than reaching for a general-purpose tomography library --
see classical_recon.py's module docstring for why (same reasoning applies here, one dimension up).

*** Every reconstruction in this file uses the TRUE rotation (R_star / pose_star) -- diagnostic-
only ground truth never fed to the model elsewhere in this codebase (see mnist_cryoem/CLAUDE.md).
Real CryoET has no such oracle; pose has to be estimated (alignment) jointly with the
reconstruction. Every number and panel here is an UPPER BOUND conditional on known pose. For
--mode tilt_series specifically, a per-acquisition reconstruction additionally lands at an
ARBITRARY absolute rotation (random per-acquisition mount orientation) -- see
reconstruct_examples' docstring, same caveat as the 2D script's tilt_series mode. ***

DC-offset note: corruption.forward_channel_3d's y = project_2d(rotate_3d(x,R)) + noise carries a
constant pedestal of -vol_size, for the same reason as the 2D channel (MNIST/extruded-volume
background is -1, not 0; rotate_3d's shifted-frame trick maps zero-padding to that -1
background; summing D=vol_size depth slices of -1 contributes a flat -D to every projection).
backproject() shifts by +vol_size before filtering/backprojecting and by -1 at the end, for
exactly the reasons documented in classical_recon.py's module docstring -- see there for the
full derivation, not repeated here. Backprojection likewise uses its own raw zero-padded
rotation helper (_rotate_raw) instead of corruption.rotate_3d, for the same double-shift reason.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from model import VOL_SIZE
from data import load_mnist_volumes_3d, build_observations_3d
from corruption import forward_channel_3d, sample_uniform_rotation_so3
from si import matrix_to_gram_schmidt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


########################################################
# Filtering -- isotropic 2D generalization of classical_recon.py's 1D ramp_filter
########################################################

def ramp_filter_2d(images: torch.Tensor, filter_type: str = "hann") -> torch.Tensor:
    """
    Radial ramp filter |k| = sqrt(kx^2+ky^2) (optionally Hann-tapered), applied to each 2D
    projection independently -- the standard weighted-backprojection filter for electron
    tomography, matching corruption.project_2d's single-axis integration after an SO(3)
    rotation (the 3D central-slice theorem's 2D analogue of 2D CT's 1D Ram-Lak filter). Exactly
    zero at k=0 by construction, same DC-nulling property as classical_recon.ramp_filter.

    Args:
        images: (..., H, W)
    Returns:
        (..., H, W) filtered, same shape
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
    ([-1,1]-background) convention, for the same reason classical_recon._rotate_raw isn't
    corruption.rotate_2d: the smeared projections this operates on already live in a
    zero-background frame (see module docstring's DC-offset note), where genuine zero-padding
    IS the correct out-of-canvas value.

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
    Reconstruct one canonical-frame volume from N known-rotation 2D projections -- the 3D
    analogue of classical_recon.backproject. Unfiltered (filtered=False) gives the classical
    blurred backprojection; filtered gives sharp weighted backprojection (WBP).

    Args:
        y: (N, 1, H, W) projections -- RAW corruption.forward_channel_3d output. The DC-offset
            shift is applied internally; callers should NOT pre-shift.
        R: (N, 3, 3) the KNOWN rotation matrix used to generate each projection.
        vol_size: D=H=W of the reconstructed canonical volume.
        filtered: apply the ramp filter (True) or plain backprojection (False).
        filter_type: passed to ramp_filter_2d when filtered=True.

    Returns:
        (1, 1, vol_size, vol_size, vol_size) reconstruction in [-1, 1] convention.
    """
    N, _, H, W = y.shape
    assert H == vol_size and W == vol_size, \
        f"projection shape {(H, W)} != (vol_size, vol_size) {(vol_size, vol_size)}"
    y_shifted = y[:, 0] + vol_size                                          # (N, H, W)
    if filtered:
        y_shifted = ramp_filter_2d(y_shifted, filter_type)
    smear = y_shifted[:, None, None, :, :].expand(-1, -1, vol_size, -1, -1)  # (N,1,D,H,W)
    R_inv = R.transpose(-1, -2)                                             # R^{-1} = R^T
    recon = _rotate_raw(smear, R_inv).mean(dim=0, keepdim=True)             # (1,1,D,H,W)
    return recon - 1.0


########################################################
# Reconstruction pipeline + metrics
########################################################

def reconstruct_examples(
    x_gt: torch.Tensor, n_views: int, noise_std: float, filter_type: str, mode: str,
    tilt_increment_deg: float, tilt_axis: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each GT volume, draw n_views known-rotation observations and reconstruct via both
    unfiltered and filtered (WBP) backprojection.

    mode="random": n_views independent Haar-uniform SO(3) rotations -- the best-case,
        full-coverage upper bound.
    mode="tilt_series": ONE realistic acquisition (data.build_observations_3d,
        corruptions_per_object=1) of n_views tilts about a FIXED lab-frame axis, evenly spaced
        by tilt_increment_deg from a random mount orientation -- the SAME acquisition geometry
        main_3d.py's actual EM pool produces: a real single-axis missing-wedge limited-angle
        problem, and (same caveat as classical_recon.py's tilt_series mode) the reconstruction
        lands at an ARBITRARY absolute orientation set by the random mount rotation.

    Returns:
        recon_unfiltered, recon_filtered: each (n, 1, D, H, W)
    """
    n = x_gt.size(0)
    vol_size = x_gt.size(-1)
    device = x_gt.device
    recon_unfilt = torch.zeros_like(x_gt)
    recon_filt = torch.zeros_like(x_gt)

    for i in range(n):
        x_i = x_gt[i:i + 1]
        if mode == "random":
            R = sample_uniform_rotation_so3(n_views, device)
            pose6 = matrix_to_gram_schmidt(R)
            y, _ = forward_channel_3d(x_i.repeat(n_views, 1, 1, 1, 1),
                                      noise_std=noise_std, pose6=pose6)
        elif mode == "tilt_series":
            y, R, _, _ = build_observations_3d(
                x_i, corruptions_per_object=1, n_tilts=n_views,
                tilt_increment_deg=tilt_increment_deg, noise_std=noise_std,
                tilt_axis=tilt_axis)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        recon_unfilt[i] = backproject(y, R, vol_size, filtered=False)[0]
        recon_filt[i] = backproject(y, R, vol_size, filtered=True,
                                    filter_type=filter_type)[0]

    return recon_unfilt, recon_filt


def _norm01(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def compare(gt: torch.Tensor, recon: torch.Tensor) -> tuple[float, float]:
    """
    Pearson r (scale/offset invariant) and SSIM (after independently min-max normalizing each
    volume to [0,1] -- same reasoning as classical_recon.compare, one dimension up; skimage's
    structural_similarity handles the 3D array directly).

    This full-volume SSIM is noticeably stricter than it looks from the gallery panel: WBP with
    a finite view count (and the Hann-windowed filter, which trades some sharpness for noise
    control) softens the sharp GT depth-band edges into a gradual transition -- confirmed
    empirically (GT's depth profile is a step function, the reconstruction's is a smooth
    sigmoid-like ramp), not a geometry bug. That softening barely shows up in the projection
    mosaic (a sum over the whole blurred band still looks like a clean rectangle) but tanks a
    voxelwise metric. See compare_mosaic for a metric that matches what the gallery panel shows.

    Args:
        gt, recon: (1, 1, D, H, W)
    """
    g, r = gt.reshape(-1), recon.reshape(-1)
    r_pearson = torch.corrcoef(torch.stack([g, r]))[0, 1].item()
    g01 = _norm01(gt)[0, 0].cpu().numpy()
    r01 = _norm01(recon)[0, 0].cpu().numpy()
    s = ssim(g01, r01, data_range=1.0)
    return r_pearson, s


def compare_mosaic(gt_mosaic: torch.Tensor, recon_mosaic: torch.Tensor) -> float:
    """
    SSIM of the xy|xz|yz projection mosaic (see _projection_mosaic) -- matches what the gallery
    panel actually shows, unlike compare()'s stricter voxelwise full-volume SSIM. Report both:
    a low full-volume SSIM next to a healthy mosaic SSIM means "shape recovered, depth
    resolution soft," not "reconstruction failed."

    Args:
        gt_mosaic, recon_mosaic: (H, 3W)
    """
    g01 = _norm01(gt_mosaic).cpu().numpy()
    r01 = _norm01(recon_mosaic).cpu().numpy()
    return ssim(g01, r01, data_range=1.0)


def _projection_mosaic(x: torch.Tensor) -> torch.Tensor:
    """
    (n, 1, D, H, W) -> (n, H, 3W): xy|xz|yz sum-projections concatenated horizontally, for
    visualizing a volume as a 2D panel. Assumes D=H=W. Same convention as
    wandb_logging._projection_mosaic (duplicated locally rather than imported -- this script is
    deliberately standalone, see module docstring).
    """
    xy = x.sum(dim=2)   # (n, 1, H, W), sum over D -- matches corruption.project_2d
    xz = x.sum(dim=3)   # (n, 1, D, W), sum over H
    yz = x.sum(dim=4)   # (n, 1, D, H), sum over W
    return torch.cat([xy, xz, yz], dim=-1)[:, 0]  # (n, H, 3W)


########################################################
# CLI
########################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical weighted-backprojection reconstruction of extruded-MNIST "
                    "volumes from the 3D/CryoET channel's 2D projections, using KNOWN (GT) "
                    "rotations -- an upper-bound feasibility check for a possible "
                    "classical-recon pretraining scheme, run BEFORE implementing it. See "
                    "module docstring."
    )
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="MNIST classes to draw the gallery from (default: all 10)")
    parser.add_argument("--n_images_per_class", type=int, default=3)
    parser.add_argument("--vol_size", type=int, default=VOL_SIZE)
    parser.add_argument("--inplane_size", type=int, default=None)
    parser.add_argument("--depth_extent", type=int, default=None)
    parser.add_argument("--mode", type=str, default="random", choices=["random", "tilt_series"])
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5,
                        help="Only used by --mode tilt_series; matches main_3d.py's default")
    parser.add_argument("--tilt_axis", type=float, nargs=3, default=[0.0, 1.0, 0.0],
                        help="Only used by --mode tilt_series; matches main_3d.py's default")
    parser.add_argument("--noise_std", type=float, default=0.5,
                        help="Matches main_3d.py's default for this channel")
    parser.add_argument("--filter_type", type=str, default="hann", choices=["hann", "ramp"])
    parser.add_argument("--n_views_gallery", type=int, default=64,
                        help="View count used for the qualitative gallery panel")
    parser.add_argument("--n_views_sweep", type=int, nargs="+",
                        default=[2, 4, 8, 16, 32, 64, 128],
                        help="View counts swept for the quantitative quality-vs-views plot")
    parser.add_argument("--out_dir", type=str, default="classical_recon_eval")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Tiny run: quick smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        def _explicit(flag):
            return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

        if not _explicit("--digit_classes"):      args.digit_classes = [0, 1]
        if not _explicit("--n_images_per_class"): args.n_images_per_class = 1
        if not _explicit("--vol_size"):           args.vol_size = 16
        if not _explicit("--n_views_gallery"):    args.n_views_gallery = 8
        if not _explicit("--n_views_sweep"):      args.n_views_sweep = [2, 8]

    print(f"Device: {device}")
    x_gt = load_mnist_volumes_3d(args.n_images_per_class, vol_size=args.vol_size,
                                 inplane_size=args.inplane_size, depth_extent=args.depth_extent,
                                 digit_classes=args.digit_classes, seed=args.seed,
                                 train=True).to(device)
    n = x_gt.size(0)
    print(f"Loaded {n} GT volumes (classes={args.digit_classes or list(range(10))}, "
          f"split=train, vol_size={args.vol_size}, mode={args.mode}, "
          f"noise_std={args.noise_std})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tilt_axis = tuple(args.tilt_axis)

    # ── Gallery panel at a fixed view count ──────────────────────────────
    recon_unfilt, recon_filt = reconstruct_examples(
        x_gt, args.n_views_gallery, args.noise_std, args.filter_type, args.mode,
        args.tilt_increment_deg, tilt_axis)

    m_unfilt = [compare(x_gt[i:i + 1], recon_unfilt[i:i + 1]) for i in range(n)]
    m_filt = [compare(x_gt[i:i + 1], recon_filt[i:i + 1]) for i in range(n)]

    gt_mosaic = _projection_mosaic(x_gt).cpu()
    unfilt_mosaic = _projection_mosaic(recon_unfilt).cpu()
    filt_mosaic = _projection_mosaic(recon_filt).cpu()

    ms_unfilt = [compare_mosaic(gt_mosaic[j], unfilt_mosaic[j]) for j in range(n)]
    ms_filt = [compare_mosaic(gt_mosaic[j], filt_mosaic[j]) for j in range(n)]

    row_labels = ["GT xy|xz|yz", "Unfiltered BP xy|xz|yz",
                 f"Filtered WBP ({args.filter_type}) xy|xz|yz"]
    row_data = [gt_mosaic, unfilt_mosaic, filt_mosaic]
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 6.5), squeeze=False)
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for j in range(n):
        for r, data in enumerate(row_data):
            img = data[j].numpy()
            axes[r, j].imshow(img, cmap="gray", vmin=img.min(), vmax=img.max(), aspect="auto")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        axes[1, j].set_title(
            f"r={m_unfilt[j][0]:.2f}  vol_ssim={m_unfilt[j][1]:.2f}\n"
            f"mosaic_ssim={ms_unfilt[j]:.2f}", fontsize=7)
        axes[2, j].set_title(
            f"r={m_filt[j][0]:.2f}  vol_ssim={m_filt[j][1]:.2f}\n"
            f"mosaic_ssim={ms_filt[j]:.2f}", fontsize=7)
    fig.suptitle(
        f"Classical WBP  |  mode={args.mode}  n_views={args.n_views_gallery}  "
        f"noise_std={args.noise_std}\nKNOWN GT pose (upper bound -- see module docstring)",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "gallery_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Gallery -> {out_dir / 'gallery_3d.png'}")
    print(f"  mean Pearson r    : unfiltered={sum(m[0] for m in m_unfilt) / n:.3f}  "
          f"filtered={sum(m[0] for m in m_filt) / n:.3f}")
    print(f"  mean vol SSIM     : unfiltered={sum(m[1] for m in m_unfilt) / n:.3f}  "
          f"filtered={sum(m[1] for m in m_filt) / n:.3f}  "
          f"(voxelwise -- strict, penalizes depth-axis blur; see compare()'s docstring)")
    print(f"  mean mosaic SSIM  : unfiltered={sum(ms_unfilt) / n:.3f}  "
          f"filtered={sum(ms_filt) / n:.3f}  (matches what the gallery panel shows)")

    # ── Sweep panel: quality vs. number of views ─────────────────────────
    sweep_r_unfilt, sweep_r_filt = [], []
    sweep_ssim_unfilt, sweep_ssim_filt = [], []
    print(f"\nSweeping n_views={args.n_views_sweep} (mode={args.mode}) ...")
    for nv in args.n_views_sweep:
        ru, rf = reconstruct_examples(x_gt, nv, args.noise_std, args.filter_type, args.mode,
                                      args.tilt_increment_deg, tilt_axis)
        pu = [compare(x_gt[i:i + 1], ru[i:i + 1]) for i in range(n)]
        pf = [compare(x_gt[i:i + 1], rf[i:i + 1]) for i in range(n)]
        sweep_r_unfilt.append(sum(p[0] for p in pu) / n)
        sweep_r_filt.append(sum(p[0] for p in pf) / n)
        sweep_ssim_unfilt.append(sum(p[1] for p in pu) / n)
        sweep_ssim_filt.append(sum(p[1] for p in pf) / n)
        print(f"  n_views={nv:4d}  "
              f"r: unfilt={sweep_r_unfilt[-1]:.3f} filt={sweep_r_filt[-1]:.3f}  "
              f"ssim: unfilt={sweep_ssim_unfilt[-1]:.3f} filt={sweep_ssim_filt[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ys_unfilt, ys_filt, ylabel in [
        (axes[0], sweep_r_unfilt, sweep_r_filt, "mean Pearson r"),
        (axes[1], sweep_ssim_unfilt, sweep_ssim_filt,
         "mean SSIM (voxelwise -- strict, see gallery panel for mosaic SSIM)"),
    ]:
        ax.plot(args.n_views_sweep, ys_unfilt, "o-", label="unfiltered BP")
        ax.plot(args.n_views_sweep, ys_filt, "o-", label="filtered WBP")
        ax.set_xlabel("n_views")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend()
    fig.suptitle(f"mode={args.mode}, noise_std={args.noise_std}  |  KNOWN GT pose (upper bound)")
    plt.tight_layout()
    fig.savefig(out_dir / "sweep_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep -> {out_dir / 'sweep_3d.png'}")
    print("Done.")
