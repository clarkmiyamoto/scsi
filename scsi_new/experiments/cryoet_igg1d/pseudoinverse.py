"""
Pose-blind classical reconstruction for the IgG-1D warm start: the single-particle counterpart of
cryoet_mnist/pseudoinverse.py's filtered backprojection (FBP). main.py's warm start trains on
scsi.ResampledPairs(data.PseudoinverseVolumes(...), pair_sample), i.e. on (target, F(x0)) pairs
re-drawn from x0 = pseudoinverse(y) for the observed images, and data.calibrate_gain sets the
default --vol_gain so these volumes have unit std.

cryoet_mnist backprojects a tilt series of ONE object: ramp-filter each projection, smear it along
its viewing direction, undo its angle, average. Here every observation is a single image of its
own particle at an unknown pose, so there is one view to backproject, in the image's own frame
(z is the viewing axis), and nothing to average:

- The ramp filter has no role. It undoes the oversampling of low frequencies by many central
  slices; a single view fills one central plane exactly once.
- The CTF is the filter to undo instead. At gain 1 the channel is A = |CTF| * P_z
  (corruption.image_formation: the observations are already phase-flipped), and P_z P_z^T = D,
  so A^+ y = P_z^T(|CTF|^+ y) / D: divide out |CTF|, smear along z, divide by D.

Plain A^+ is useless at SNR 0.01: 1/|CTF| amplifies the noise 10x at low frequency
(|CTF(0)| = amplitude contrast 0.1) and without bound at the CTF zeros. pseudoinverse() therefore
regularises with the data's own spectral prior, i.e. it applies the Wiener filter

    W(k) = |CTF(k)| S(|k|) / (|CTF(k)|^2 S(|k|) + 1)

before smearing, where S is the radially averaged spectral SNR of a CTF-free projection, estimated
from the observations alone (estimate_ssnr). W -> 1/|CTF| where the signal dominates and -> 0 where
the noise does, so it also low-passes: S falls below 1 near 30 A on IgG-1D at 64 px.

Support: the exact minimum-norm solution smears over the whole 384 A box, several times the
particle's size (its density lies within ~100 A of the box centre, shift included).
`config.pinv_diameter_A` (--pinv_diameter_A) cuts the smear to a centred sphere of that diameter,
the spherical mask every cryo-EM reconstruction is given, and divides by the diameter in voxels:
the central column then projects to the corrected image exactly, and the chord length falling
off towards the rim acts as a soft in-plane mask. None keeps the whole-box smear (divided by D),
the exact pseudoinverse of the corrected image.

POSE-BLIND, as cryoet_mnist's warm start: no pose is used, only (y, its CTF). One view fixes the
projection along z and says nothing about depth, so x0 is the CTF-corrected image extruded
through a ball, a rough symmetry-breaking start for EM and not a reconstruction.
`python pseudoinverse.py` scores it against the GT image-frame targets (GT used there only).
"""

import torch

from corruption import apply_filter, ctf_2d


def shell_index(D: int, device=None) -> torch.Tensor:
    """(D, D) radial frequency shell round(|k| D) of each FFT-order pixel, 0 .. round(D / sqrt 2)."""
    k = torch.fft.fftfreq(D, device=device, dtype=torch.float64)
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    return (torch.hypot(ky, kx) * D).round().long()


@torch.no_grad()
def estimate_ssnr(y: torch.Tensor, ctf_params: torch.Tensor, apix: float, device: torch.device,
                  n: int = 4096, batch_size: int = 512) -> torch.Tensor:
    """
    Radially averaged spectral SNR S of a CTF-free projection, from the observations alone. Their
    noise is white with unit variance (data.build_observations), so with an unnormalised FFT
    E|Y(k)|^2 = D^2 (|CTF(k)|^2 S(|k|) + 1), and per shell S = (mean |Y|^2 / D^2 - 1) / mean |CTF|^2,
    clamped at 0. Deterministic: uses n observations spread evenly over y, which is sorted by
    conformation.

    Args:
        y: (N, 1, D, D) observations, any device; ctf_params: (N, 7) their CTF parameters

    Returns:
        (n_shells,) float32 on `device`, indexed by shell_index(D)
    """
    N, D = y.size(0), y.size(-1)
    idx = torch.linspace(0, N - 1, min(n, N)).round().long()
    shell = shell_index(D, device).flatten()
    n_shells = int(shell.max()) + 1
    power = torch.zeros(n_shells, dtype=torch.float64, device=device)
    ctf2 = torch.zeros(n_shells, dtype=torch.float64, device=device)
    for chunk in idx.split(batch_size):
        Y = torch.fft.fft2(y[chunk, 0].to(device, torch.float64))
        C = ctf_2d(D, apix, ctf_params[chunk].to(device)).double()
        power.index_add_(0, shell, Y.abs().pow(2).sum(0).flatten())
        ctf2.index_add_(0, shell, C.pow(2).sum(0).flatten())
    count = torch.bincount(shell, minlength=n_shells).double() * len(idx)
    ssnr = (power / count / D**2 - 1.0) / (ctf2 / count).clamp_min(1e-12)
    return ssnr.clamp_min(0.0).float()


def wiener_filter(ctf_params: torch.Tensor, ssnr: torch.Tensor, apix: float, D: int) -> torch.Tensor:
    """(B, 7) CTF parameters -> (B, D, D) FFT-order |CTF| S / (|CTF|^2 S + 1)."""
    C = ctf_2d(D, apix, ctf_params).abs()
    S = ssnr.to(C.device)[shell_index(D, C.device)]
    return C * S / (C.pow(2) * S + 1.0)


def support(diameter_A: float | None, apix: float, D: int,
            device=None) -> tuple[torch.Tensor | None, float]:
    """
    (mask, norm): the (D, D, D) centred ball of diameter diameter_A and that diameter in voxels,
    or (None, D) for the whole box when diameter_A is None.
    """
    if diameter_A is None:
        return None, float(D)
    d = diameter_A / apix
    r = torch.arange(D, device=device, dtype=torch.float32) - D // 2
    zz, yy, xx = torch.meshgrid(r, r, r, indexing="ij")
    return (zz**2 + yy**2 + xx**2).sqrt() <= d / 2, d


def pseudoinverse(y: torch.Tensor, ctf_params: torch.Tensor, apix: float, config,
                  ssnr: torch.Tensor) -> torch.Tensor:
    """
    Pose-blind classical reconstruction of each observed image, in that image's frame: Wiener
    CTF-correct it, smear it along z, cut the smear to the support() ball and divide by its
    diameter in voxels.

    Args:
        y:          (B, 1, D, D) observations as data.build_observations returns them
                    (phase-flipped with their own CTF, noise std 1)
        ctf_params: (B, 7) those images' CTF parameters (corruption.ctf_2d layout)
        apix:       pixel size in A at this resolution
        config:     data.Config_Dataset_IgG (pinv_diameter_A)
        ssnr:       estimate_ssnr(...) of the observations

    Returns:
        (B, 1, D, D, D) volumes at gain 1; data.calibrate_gain rescales them to unit std.
    """
    D = y.size(-1)
    p = apply_filter(y, wiener_filter(ctf_params, ssnr, apix, D))           # (B, 1, D, D)
    mask, norm = support(config.pinv_diameter_A, apix, D, y.device)
    x = (p / norm)[:, :, None].expand(-1, -1, D, -1, -1)                   # (B, 1, D, D, D)
    return x if mask is None else x * mask


if __name__ == "__main__":
    # Scores the warm start against GT, which the EM loop never sees. For training images with
    # their image-frame targets: r of the Wiener-corrected image with the target's z-projection,
    # then over a sweep of --pinv_diameter_A, Pearson r of x0 with its own target vs another
    # image's (the gap is the image-specific part), and r of their projections at a fresh pose,
    # which is what the warm start's (target, F(x0)) pairs see. Data loading as data.py's check.
    import argparse
    from dataclasses import replace

    from corruption import project_z
    from rotation import center_of_mass, pose_volumes, sample_uniform_rotation_so3
    from data import (DEFAULT_CRYOBENCH_ROOT, DEFAULT_DATA_ROOT, Config_Dataset_IgG,
                      build_observations, import_cryobench)

    parser = argparse.ArgumentParser(description="Score pseudoinverse() against GT targets")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cryobench_root", default=DEFAULT_CRYOBENCH_ROOT)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n_observations", type=int, default=2000,
                        help="Observations for the noise std and the SSNR estimate.")
    parser.add_argument("--n_check", type=int, default=48)
    parser.add_argument("--diameters_A", type=float, nargs="+",
                        default=[144.0, 192.0, 240.0, 288.0])
    parser.add_argument("--save", type=str, default=None, help="Write a PNG of example panels here.")
    args = parser.parse_args()

    config = Config_Dataset_IgG(data_root=args.data_root, cryobench_root=args.cryobench_root,
                                resolution=args.resolution, n_observations=args.n_observations)
    D, apix = config.resolution, config.apix
    observations, info = build_observations(config)
    ssnr = estimate_ssnr(observations.tensors[0], info["ctf_pool"], apix, torch.device("cpu"))
    k_half = int((ssnr[:D // 2 + 1] >= 1).nonzero().max()) if (ssnr >= 1).any() else 0
    print(f"SSNR >= 1 up to shell {k_half} ({apix * D / max(k_half, 1):.1f} A)")

    cb = import_cryobench(config.cryobench_root)
    ds = cb.IgG1DDataset(config.data_root, resolution=D, snr=config.snr, return_meta=True,
                         volume_frame="image", phase_flip=True)
    check_idx = info["index"][:: max(1, len(info["index"]) // args.n_check)][:args.n_check]
    items = [ds[int(i)] for i in check_idx]
    x_gt = torch.stack([x for x, _, _ in items])[:, None]
    y = torch.stack([y for _, y, _ in items])[:, None] / info["sigma"]
    ctf = ds.ctf[torch.from_numpy(check_idx), 2:]

    def corr(u, v):
        u, v = u.flatten(1), v.flatten(1)
        u, v = u - u.mean(1, keepdim=True), v - v.mean(1, keepdim=True)
        return ((u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1)).clamp_min(1e-12)).mean().item()

    proj = project_z(x_gt)
    p_wiener = apply_filter(y, wiener_filter(ctf, ssnr, apix, D))
    print(f"z-projection r with GT projection: phase-flipped y {corr(y, proj):.3f}   "
          f"Wiener-corrected {corr(p_wiener, proj):.3f}   (mismatched {corr(p_wiener, proj.roll(1, 0)):.3f})")
    g = torch.Generator().manual_seed(0)
    R = sample_uniform_rotation_so3(len(check_idx), generator=g)

    def reposed_proj(x):
        return project_z(pose_volumes(x, R, center=center_of_mass(x)))

    print(f"{'diameter_A':>10} {'r own':>7} {'r other':>8} {'gap':>6} {'new-pose proj r':>16} "
          f"{'std (gain)':>11}")
    for d in [*args.diameters_A, None]:
        x0 = pseudoinverse(y, ctf, apix, replace(config, pinv_diameter_A=d), ssnr)
        own, other = corr(x0, x_gt), corr(x0, x_gt.roll(1, 0))
        print(f"{str(d):>10} {own:7.3f} {other:8.3f} {own - other:6.3f} "
              f"{corr(reposed_proj(x0), reposed_proj(x_gt)):16.3f} {x0.std().item():11.4g}")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path

        x0 = pseudoinverse(y, ctf, apix, config, ssnr)
        n = min(6, len(check_idx))
        rows = [("y (observed)", y[:, 0]), ("Wiener-corrected", p_wiener[:, 0]),
                ("GT z-proj", proj[:, 0]), ("x0 x-proj", x0[:, 0].sum(-1)),
                ("GT x-proj", x_gt[:, 0].sum(-1))]
        fig, axes = plt.subplots(len(rows), n, figsize=(2 * n, 2 * len(rows)), squeeze=False)
        for r, (name, imgs) in enumerate(rows):
            for j in range(n):
                img = imgs[j].numpy()
                lo, hi = torch.quantile(imgs[j].flatten(), torch.tensor([0.01, 0.99])).tolist()
                axes[r, j].imshow(img, cmap="gray", vmin=lo, vmax=hi)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            axes[r, 0].set_ylabel(name, fontsize=8)
        fig.suptitle(f"pseudoinverse, --pinv_diameter_A {config.pinv_diameter_A}", fontsize=10)
        fig.tight_layout()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=110)
        print(f"saved {args.save}")
