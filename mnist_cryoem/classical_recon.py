"""
Classical (non-learned) tomographic reconstruction for the 2D CryoEM channel (SO(2)
in-plane rotation -> 1D projection -> AWGN), as a pre-implementation feasibility check for a
possible classical-reconstruction-based pretraining scheme -- an alternative to pretrain.py's
GT-supervised Theta^(0) warm start, where the "pseudo-GT" x_hat fed to the interpolant loss
would come from a classical inversion of a real tilt series instead of literal known MNIST
labels. Standalone diagnostic script: it doesn't touch si.py/scsi.py/em.py/model.py, and isn't
wired into the SCSI algorithm anywhere.

Method: filtered backprojection (FBP), the classical exact inversion of the parallel-beam
Radon transform -- exactly the geometry corruption.py's rotate_2d + project_1d implement, so
this script re-derives that SAME geometry from scratch rather than reaching for
skimage.transform.iradon (which has its own padding / rotation-direction / sinogram-width
conventions that would need separate verification against this codebase's channel; matching
them would be a verification task with zero payoff over just deriving the adjoint directly).

*** Every reconstruction in this file uses the TRUE (theta_star) rotation angle -- the
diagnostic-only ground truth that's never fed to the model elsewhere in this codebase (see
mnist_cryoem/CLAUDE.md's theta_star gotcha). Real CryoEM has no such oracle; pose has to be
estimated (alignment) jointly with the reconstruction. So every number and panel here is an
UPPER BOUND conditional on known pose -- it answers "is the reconstruction math good enough,"
NOT "is a classical-recon pretraining scheme viable end-to-end." That second question
additionally needs a pose-estimation step this script does not implement (and, for
--mode tilt_series in particular, needs to resolve that a per-acquisition reconstruction lands
at an ARBITRARY absolute rotation -- see reconstruct_examples' docstring). ***

DC-offset note: corruption.forward_channel's y = project_1d(rotate_2d(x,theta)) + noise carries
a constant pedestal of -image_size: MNIST's background is -1, not 0, and rotate_2d's
shifted-frame trick maps zero-padding to that -1 background, so summing H=image_size rows of
-1 contributes a flat -H to every projection, regardless of column or viewing angle. Standard
FBP assumes zero density outside the object, so backproject() shifts every projection by
+image_size before filtering/backprojecting (recovering the "x+1" frame, where background truly
is 0), then shifts the final averaged reconstruction by -1 to land back in MNIST's [-1,1]
convention. This is also why backprojection uses its own raw zero-padded rotation helper
(_rotate_raw) instead of corruption.rotate_2d: rotate_2d's OWN internal shift trick assumes ITS
input already has a -1 background, which the shifted/filtered projections here do not -- reusing
it directly would double-shift.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from data import load_mnist_subset, build_observations
from corruption import forward_channel, sample_uniform_angle

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


########################################################
# Filtering
########################################################

def ramp_filter(sino: torch.Tensor, filter_type: str = "hann") -> torch.Tensor:
    """
    Ram-Lak ramp filter (optionally Hann-tapered) along the last dim -- the classical FBP
    filter, |freq| in Fourier space. Exactly zero at freq=0 by construction, which is what
    makes the forward_channel DC pedestal (see module docstring) a no-op after filtering,
    whether or not the caller has separately shifted it away first.

    Args:
        sino: (..., W)
    Returns:
        (..., W) filtered, same shape
    """
    W = sino.size(-1)
    freqs = torch.fft.fftfreq(W, device=sino.device, dtype=sino.dtype)
    nyquist = freqs.abs().max().clamp_min(1e-8)
    resp = freqs.abs() / nyquist  # Ram-Lak: linear ramp, 0 at DC, 1 at Nyquist
    if filter_type == "hann":
        resp = resp * 0.5 * (1.0 + torch.cos(torch.pi * freqs / nyquist))
    elif filter_type != "ramp":
        raise ValueError(f"Unknown filter_type: {filter_type!r}")
    spec = torch.fft.fft(sino, dim=-1) * resp
    return torch.fft.ifft(spec, dim=-1).real


########################################################
# Backprojection -- the adjoint of corruption.rotate_2d + corruption.project_1d
########################################################

def _rotate_raw(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Plain affine rotation, zero-padded -- deliberately NOT corruption.rotate_2d's black-fill
    ([-1,1]-background) convention. Used only inside backproject(): the smeared projections it
    operates on live in a zero-background frame (see module docstring's DC-offset note), where
    genuine zero-padding IS the correct out-of-canvas value.

    Args:
        x: (B, 1, H, W)
        theta: (B,) radians
    """
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(theta)
    rot = torch.stack([
        torch.stack([cos, -sin, zeros], dim=-1),
        torch.stack([sin,  cos, zeros], dim=-1),
    ], dim=-2)  # (B, 2, 3)
    grid = F.affine_grid(rot, x.shape, align_corners=True)
    return F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="zeros")


def backproject(y: torch.Tensor, theta: torch.Tensor, image_size: int,
                filtered: bool, filter_type: str = "hann") -> torch.Tensor:
    """
    Reconstruct one canonical-frame image from N known-angle 1D projections. Unfiltered
    (filtered=False) gives the classical 1/r-blurred backprojection; filtered gives sharp FBP.

    Args:
        y: (N, 1, W) projections -- RAW corruption.forward_channel output. The DC-offset shift
            is applied internally; callers should NOT pre-shift.
        theta: (N,) the KNOWN angle used to generate each projection.
        image_size: H=W of the reconstructed canonical image.
        filtered: apply the ramp filter (True) or plain backprojection (False).
        filter_type: passed to ramp_filter when filtered=True.

    Returns:
        (1, 1, image_size, image_size) reconstruction in MNIST's [-1, 1] convention.
    """
    N, _, W = y.shape
    assert W == image_size, f"projection width {W} != image_size {image_size}"
    y_shifted = y[:, 0, :] + image_size                                    # (N, W)
    if filtered:
        y_shifted = ramp_filter(y_shifted, filter_type)
    smear = y_shifted[:, None, None, :].expand(-1, -1, image_size, -1)     # (N, 1, H, W)
    recon = _rotate_raw(smear, -theta).mean(dim=0, keepdim=True)           # (1, 1, H, W)
    return recon - 1.0


########################################################
# Reconstruction pipeline + metrics
########################################################

def reconstruct_examples(
    x_gt: torch.Tensor, n_views: int, noise_std: float, filter_type: str, mode: str,
    tilt_increment_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each GT digit, draw n_views known-angle observations and reconstruct via both
    unfiltered and filtered backprojection.

    mode="random": n_views independent Haar-uniform angles -- the best-case, full-coverage
        upper bound (what unlimited-view tomography could achieve).
    mode="tilt_series": ONE realistic acquisition (data.build_observations,
        corruptions_per_object=1) of n_views tilts evenly spaced by tilt_increment_deg from a
        random start offset -- the SAME acquisition geometry main.py's actual EM pool produces,
        so a real missing-wedge limited-angle problem when n_views * tilt_increment_deg < 180.
        NOTE: because the start offset is random per acquisition, this mode's reconstruction
        lands at an ARBITRARY absolute rotation, not the canonical one -- fine for a shape
        sanity check (this function uses theta_star for backprojection, which registers it back
        to canonical for free), but a real classical-recon pretraining scheme would need its own
        registration step this script does not implement. See the module docstring.

    Returns:
        recon_unfiltered, recon_filtered: each (n, 1, H, W)
    """
    n = x_gt.size(0)
    image_size = x_gt.size(-1)
    device = x_gt.device
    recon_unfilt = torch.zeros_like(x_gt)
    recon_filt = torch.zeros_like(x_gt)

    for i in range(n):
        x_i = x_gt[i:i + 1]
        if mode == "random":
            theta = sample_uniform_angle(n_views, device)
            y, theta = forward_channel(x_i.repeat(n_views, 1, 1, 1),
                                       noise_std=noise_std, theta=theta)
        elif mode == "tilt_series":
            y, theta, _, _ = build_observations(
                x_i, corruptions_per_object=1, n_tilts=n_views,
                tilt_increment_deg=tilt_increment_deg, noise_std=noise_std)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        recon_unfilt[i] = backproject(y, theta, image_size, filtered=False)[0]
        recon_filt[i] = backproject(y, theta, image_size, filtered=True,
                                    filter_type=filter_type)[0]

    return recon_unfilt, recon_filt


def _norm01(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def compare(gt: torch.Tensor, recon: torch.Tensor) -> tuple[float, float]:
    """
    Pearson r (scale/offset invariant -- unaffected by the DC pedestal or any calibration
    choice) and SSIM (computed after independently min-max normalizing each image to [0,1],
    since raw backprojection output isn't naturally on MNIST's [-1,1] scale).

    Args:
        gt, recon: (1, 1, H, W)
    """
    g, r = gt.reshape(-1), recon.reshape(-1)
    r_pearson = torch.corrcoef(torch.stack([g, r]))[0, 1].item()
    g01 = _norm01(gt)[0, 0].cpu().numpy()
    r01 = _norm01(recon)[0, 0].cpu().numpy()
    s = ssim(g01, r01, data_range=1.0)
    return r_pearson, s


########################################################
# CLI
########################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical filtered-backprojection reconstruction of MNIST digits from "
                    "the 2D CryoEM channel's 1D projections, using KNOWN (GT) rotation angles "
                    "-- an upper-bound feasibility check for a possible classical-recon "
                    "pretraining scheme, run BEFORE implementing it. See module docstring."
    )
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="MNIST classes to draw the gallery from (default: all 10)")
    parser.add_argument("--n_images_per_class", type=int, default=3)
    parser.add_argument("--mode", type=str, default="random", choices=["random", "tilt_series"])
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5,
                        help="Only used by --mode tilt_series; matches main.py's default")
    parser.add_argument("--noise_std", type=float, default=3.0,
                        help="Matches main.py's default for this channel")
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
        if not _explicit("--n_views_gallery"):    args.n_views_gallery = 8
        if not _explicit("--n_views_sweep"):      args.n_views_sweep = [2, 8]

    print(f"Device: {device}")
    x_gt = load_mnist_subset(args.n_images_per_class, digit_classes=args.digit_classes,
                             seed=args.seed, train=True).to(device)
    n = x_gt.size(0)
    print(f"Loaded {n} GT digits (classes={args.digit_classes or list(range(10))}, "
          f"split=train, mode={args.mode}, noise_std={args.noise_std})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Gallery panel at a fixed view count ──────────────────────────────
    recon_unfilt, recon_filt = reconstruct_examples(
        x_gt, args.n_views_gallery, args.noise_std, args.filter_type, args.mode,
        args.tilt_increment_deg)

    m_unfilt = [compare(x_gt[i:i + 1], recon_unfilt[i:i + 1]) for i in range(n)]
    m_filt = [compare(x_gt[i:i + 1], recon_filt[i:i + 1]) for i in range(n)]

    row_labels = ["GT", "Unfiltered BP", f"Filtered FBP ({args.filter_type})"]
    row_data = [x_gt, recon_unfilt, recon_filt]
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6.5), squeeze=False)
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for j in range(n):
        for r, data in enumerate(row_data):
            img = data[j, 0].detach().cpu().numpy()
            axes[r, j].imshow(img, cmap="gray", vmin=img.min(), vmax=img.max())
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        axes[1, j].set_title(f"r={m_unfilt[j][0]:.2f}\nssim={m_unfilt[j][1]:.2f}", fontsize=7)
        axes[2, j].set_title(f"r={m_filt[j][0]:.2f}\nssim={m_filt[j][1]:.2f}", fontsize=7)
    fig.suptitle(
        f"Classical FBP  |  mode={args.mode}  n_views={args.n_views_gallery}  "
        f"noise_std={args.noise_std}\nKNOWN GT pose (upper bound -- see module docstring)",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "gallery_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Gallery -> {out_dir / 'gallery_2d.png'}")
    print(f"  mean Pearson r : unfiltered={sum(m[0] for m in m_unfilt) / n:.3f}  "
          f"filtered={sum(m[0] for m in m_filt) / n:.3f}")
    print(f"  mean SSIM      : unfiltered={sum(m[1] for m in m_unfilt) / n:.3f}  "
          f"filtered={sum(m[1] for m in m_filt) / n:.3f}")

    # ── Sweep panel: quality vs. number of views ─────────────────────────
    sweep_r_unfilt, sweep_r_filt = [], []
    sweep_ssim_unfilt, sweep_ssim_filt = [], []
    print(f"\nSweeping n_views={args.n_views_sweep} (mode={args.mode}) ...")
    for nv in args.n_views_sweep:
        ru, rf = reconstruct_examples(x_gt, nv, args.noise_std, args.filter_type, args.mode,
                                      args.tilt_increment_deg)
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
        (axes[1], sweep_ssim_unfilt, sweep_ssim_filt, "mean SSIM"),
    ]:
        ax.plot(args.n_views_sweep, ys_unfilt, "o-", label="unfiltered BP")
        ax.plot(args.n_views_sweep, ys_filt, "o-", label="filtered FBP")
        ax.set_xlabel("n_views")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend()
    fig.suptitle(f"mode={args.mode}, noise_std={args.noise_std}  |  KNOWN GT pose (upper bound)")
    plt.tight_layout()
    fig.savefig(out_dir / "sweep_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep -> {out_dir / 'sweep_2d.png'}")
    print("Done.")
