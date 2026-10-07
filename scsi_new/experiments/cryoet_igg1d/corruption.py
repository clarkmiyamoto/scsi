"""
Black-box forward model for single-particle cryo-EM on CryoBench IgG-1D.

The released images follow y_raw = -CTF * P(T_s R x) + noise (IgG1DDataset docstring), with
P the sum over z, R a uniform SO(3) pose, T_s an in-plane shift and a per-image CTF. The
observations are phase-flipped with their own (known) CTF, as in the supervised cryofm run
(phase_flip=True), which turns -CTF into |CTF|, and divided by the noise std estimated from the
image corners (data.build_observations). In those units the channel is

    F(x) = vol_gain * IFFT2( |CTF_c| * FFT2( P(T_s R x) ) ) + noise_std * eps,   eps ~ N(0, I)

with a fresh R (Haar), s ~ U[-shift_extent, shift_extent]^2 and c a random row of the
observations' CTF parameters. Per-image CTFs are estimated in real cryo-EM; poses are not.
vol_gain sets the units of the volumes the model generates (see args.py --vol_gain).

x_hat from the E-step sits in the frame of the image it was sampled for, shift included. With
recenter=True (the default) the channel first moves x_hat's centre of mass to the box centre,
so the shifts of re-posed pairs follow the channel's shift distribution instead of piling each
source image's shift on top of a fresh one.
"""

import torch

from .rotation import center_of_mass, pose_volumes, sample_uniform_rotation_so3

PAIR_FRAMES = ("image", "canonical", "lift")


def ctf_2d(D: int, apix: float, params: torch.Tensor) -> torch.Tensor:
    """
    cryoDRGN's CTF (CryoBench cryobench_data/igg1d.py::compute_ctf), batched: one D x D CTF per
    row of params, in FFT (unshifted) order.

    Args:
        params: (B, 7) [dfU (A), dfV (A), dfang (deg), kV, Cs (mm), w, phase_shift (deg)]

    Returns:
        (B, D, D) float32
    """
    p = params.to(torch.float64)
    dfu, dfv, dfang, volt, cs, w, phase_shift = (p[:, i, None, None] for i in range(7))
    f = torch.fft.fftfreq(D, d=apix, dtype=torch.float64, device=params.device)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    volt = volt * 1000
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2) ** 0.5
    ang = torch.atan2(fy, fx)
    s2 = fx**2 + fy**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - torch.deg2rad(dfang))))
    gamma = 2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * 1e7 * lam**3 * s2**2)
    gamma = gamma - torch.deg2rad(phase_shift)
    return ((1 - w**2) ** 0.5 * torch.sin(gamma) - w * torch.cos(gamma)).float()


def apply_filter(img: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """(B, 1, D, D) images times a (B, D, D) FFT-order filter in Fourier space."""
    return torch.fft.ifft2(torch.fft.fft2(img.float()) * filt[:, None]).real


def phase_flip(img: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
    """Multiply each image's Fourier transform by sign(-CTF), as IgG1DDataset(phase_flip=True)."""
    return apply_filter(img, torch.sign(-ctf))


def project_z(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, D, H, W) -> (B, 1, H, W): parallel-beam projection along z."""
    return x.sum(dim=-3)


def image_formation(x_frame: torch.Tensor, ctf_params: torch.Tensor, apix: float) -> torch.Tensor:
    """Noiseless, gain-1 phase-flipped image of an already-posed volume: |CTF| * P(x_frame)."""
    return apply_filter(project_z(x_frame), ctf_2d(x_frame.size(-1), apix, ctf_params).abs())


def corruption_channel(x: torch.Tensor, *,
                       ctf_pool: torch.Tensor,
                       vol_gain: float,
                       apix: float,
                       shift_extent_px: float,
                       noise_std: float = 1.0,
                       recenter: bool = True,
                       return_frame: bool = False):
    """
    Black-box forward model F: a fresh random pose, shift and CTF per volume.

    Args:
        x: (B, 1, D, D, D) volumes
        ctf_pool: (N, 7) CTF parameters to draw from (ctf_2d layout), any device

    Returns:
        y: (B, 1, D, D); with return_frame, (x_frame, y) where x_frame (B, 1, D, D, D) is the
        posed volume whose image y is (the image-frame target).
    """
    B, device = x.size(0), x.device
    R = sample_uniform_rotation_so3(B, device=device)
    shift = (torch.rand(B, 2, device=device) * 2 - 1) * shift_extent_px
    idx = torch.randint(len(ctf_pool), (B,), device=ctf_pool.device)
    ctf_params = ctf_pool[idx].to(device)
    center = center_of_mass(x) if recenter else None

    x_frame = pose_volumes(x, R, shift, center)
    y = vol_gain * image_formation(x_frame, ctf_params, apix)
    y = y + noise_std * torch.randn_like(y)
    return (x_frame, y) if return_frame else y


def build_pair_sample(corruption_channel, *, frame: str, recenter: bool = True):
    """
    `(x) -> (target, y)` for scsi.estep and the warm start's scsi.ResampledPairs.

    y = corruption_channel(x) in every case. The target is
        image:     the posed volume F imaged (its z-projection underlies y), the frame the
                   supervised cryofm model is trained in (IgG1DDataset volume_frame="image");
        canonical: x itself, whatever frame it is in;
        lift:      x under a fresh Haar rotation independent of F's pose (cryoet_mnist3d --lift),
                   recentred first when `recenter` is set.
    """
    if frame not in PAIR_FRAMES:
        raise ValueError(f"unknown pair frame {frame!r}; choose from {PAIR_FRAMES}")

    def pair_sample(x: torch.Tensor):
        if frame == "image":
            return corruption_channel(x, return_frame=True)
        y = corruption_channel(x)
        if frame == "canonical":
            return x, y
        R = sample_uniform_rotation_so3(x.size(0), device=x.device)
        return pose_volumes(x, R, center=center_of_mass(x) if recenter else None), y
    return pair_sample


if __name__ == "__main__":
    # Convention checks, CPU, a few seconds: pose_volumes against CryoBench's rotate_volume,
    # ctf_2d against compute_ctf, recentring, and mirror_z's projection invariance.
    import argparse

    from .data import DEFAULT_CRYOBENCH_ROOT, import_cryobench
    from .rotation import mirror_z

    parser = argparse.ArgumentParser(description="Check the channel's conventions against CryoBench")
    parser.add_argument("--cryobench_root", default=DEFAULT_CRYOBENCH_ROOT)
    parser.add_argument("--D", type=int, default=32)
    args = parser.parse_args()
    cb = import_cryobench(args.cryobench_root)
    torch.manual_seed(0)
    D = args.D

    # A smooth off-centre blob, so interpolation differences stay small.
    vol = torch.zeros(2, 1, D, D, D)
    vol[:, :, D // 2 - 3:D // 2 + 5, D // 2 - 6:D // 2 + 2, D // 2 - 2:D // 2 + 7] = 1.0
    vol = torch.nn.functional.avg_pool3d(vol, 3, stride=1, padding=1)
    R = sample_uniform_rotation_so3(2)
    shift = torch.tensor([[2.5, -1.0], [-3.0, 4.0]])
    ours = pose_volumes(vol, R, shift)
    ref = torch.stack([cb.rotate_volume(vol[b, 0], R[b], shift[b]) for b in range(2)])[:, None]
    print(f"pose_volumes vs rotate_volume: max |diff| {(ours - ref).abs().max():.2e} "
          f"(volume max {vol.max():.2f})")

    params = torch.tensor([[12000.0, 11000.0, 30.0, 300.0, 2.7, 0.1, 0.0],
                           [40000.0, 38000.0, -120.0, 300.0, 2.7, 0.1, 0.0]])
    ours = ctf_2d(D, 12.0, params)
    ref = torch.stack([cb.compute_ctf(D, 12.0, *p.tolist()) for p in params])
    print(f"ctf_2d vs compute_ctf:          max |diff| {(ours - ref).abs().max():.2e}")

    com = center_of_mass(pose_volumes(vol, center=center_of_mass(vol)))
    print(f"recentred centre of mass:       {com.abs().max():.2e} voxels "
          f"(was {center_of_mass(vol).abs().max():.2f})")
    posed = pose_volumes(vol, R)
    print(f"mirror_z projection change:     "
          f"{(project_z(mirror_z(posed)) - project_z(posed)).abs().max():.2e}")
