"""Pseudo-inverse of the CryoET tilt-series forward model: filtered
back-projection (FBP).

Minimal port of ``src/classical_uvt.py::fbp`` plus the essential steps of
``src/forward_maps.py::_radon_tilt_warmup_target``. FBP is the classical
(no-learning) reconstruction; here it serves two roles:
  * the honest ``x_0`` estimate that seeds the SCSI warmup phase, and
  * the baseline we compare the learned reconstruction against.

The TV / L0 / disk-mask / sinogram-smoothing polish from the full recipe is
deliberately dropped for this minimal version (documented as a future add-on).
"""
import math

import torch

from forward import rotate_image, tilt_angles


def fbp(projections: torch.Tensor, angles: torch.Tensor, D: int,
        ramp_filter: bool = True, window: str = 'hamming') -> torch.Tensor:
    """Filtered back-projection at known angles.

    Args:
        projections: ``[N, K, D]`` 1-D projections.
        angles:      ``[N, K]`` projection angles in radians (same convention as
                     the forward map: forward rotated by ``-angle``, so FBP
                     rotates by ``+angle``).
        D:           image side length (== ``projections.shape[-1]``).
        ramp_filter: apply the ``|omega|`` ramp filter in Fourier space.
        window:      ``'ramp'`` or ``'hamming'`` (Hamming suppresses ringing).

    Returns ``[N, 1, D, D]``.
    """
    N, K, P = projections.shape
    if P != D:
        raise ValueError(f"projection length {P} != D {D}")
    device = projections.device

    if ramp_filter:
        # Ramp |omega| in radians-per-sample (factor 2*pi over rfftfreq's
        # cycles-per-sample), optionally Hamming-windowed.
        proj_freq = torch.fft.rfft(projections, dim=-1)
        freqs = (2.0 * math.pi) * torch.fft.rfftfreq(D, d=1.0, device=device).abs()
        if window == 'hamming':
            n = freqs.shape[0]
            idx = torch.arange(n, device=device, dtype=torch.float32)
            freqs = freqs * (0.54 + 0.46 * torch.cos(math.pi * idx / max(n - 1, 1)))
        elif window != 'ramp':
            raise ValueError(f"unknown window '{window}'; use 'ramp' or 'hamming'")
        proj_filt = torch.fft.irfft(proj_freq * freqs, n=D, dim=-1)
    else:
        proj_filt = projections

    # Tile each filtered projection along width, rotate by +angle, sum over K.
    # bg=0.0 here (the filtered tiles are zero-mean signals, not pm1 images).
    tile = proj_filt.unsqueeze(-1).expand(N, K, D, D).reshape(N * K, 1, D, D)
    rotated = rotate_image(tile, angles.reshape(N * K), bg=0.0)
    back = rotated.reshape(N, K, 1, D, D).sum(dim=1)          # [N, 1, D, D]
    if ramp_filter:
        return back * (math.pi * D / (2.0 * K))               # half-circle FBP norm
    return back / float(K)


def warmup_target(cond: torch.Tensor, K: int, tilt_span_deg: float = 60.0) -> torch.Tensor:
    """FBP reconstruction used as the SCSI warmup ``x_0``.

    ``cond`` is the ``[N, K, H, W]`` tiled projections from
    :func:`forward.radon_tilt_series`; the 1-D projection ``p_k`` is the first
    column of each tile. We FBP at the *known relative* schedule (treating the
    unknown global ``theta0 = 0``), so the result is canonically oriented and may
    be rotated relative to the true image — that ambiguity is inherent to the
    cryo-ET problem and is spread over the orbit during warmup.

    Returns ``[N, 1, H, W]`` rescaled per-image to ``[-1, 1]`` (pm1 MNIST scale).
    """
    N, K_, H, W = cond.shape
    projections = cond[..., 0]                                 # [N, K, H]
    delta = tilt_angles(K, tilt_span_deg).to(cond.device)
    angles = delta.unsqueeze(0).expand(N, K)                  # [N, K]
    recon = fbp(projections, angles, H, ramp_filter=True, window='hamming')
    rmin = recon.amin(dim=(-2, -1), keepdim=True)
    rmax = recon.amax(dim=(-2, -1), keepdim=True)
    return 2.0 * (recon - rmin) / (rmax - rmin + 1e-8) - 1.0
