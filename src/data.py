"""Cryo-EM dataset loading and CTF forward model using cryodrgn.

Downloads and loads the cryodrgn 'hand' test dataset (100 projection
images, 64x64) and defines a CTF (Contrast Transfer Function) forward
model compatible with the SCSI training framework.

The cryo-EM imaging forward model is:
    y(k) = CTF(k) · x(k) + noise
where x is a clean 2D projection image and y is the observed
(corrupted) particle image, both in Fourier space.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from cryodrgn.mrcfile import parse_mrc
from cryodrgn.ctf import compute_ctf
from cryodrgn import utils as cryodrgn_utils

from forward import ForwardModel
from distribution import Distribution


_GITHUB_RAW = (
    "https://raw.githubusercontent.com/ml-struct-bio/cryodrgn/main/tests/data"
)
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hand"

# Files needed for the hand dataset
_HAND_FILES = {
    "hand.mrcs": f"{_GITHUB_RAW}/hand.mrcs",
    "ctf.pkl": f"{_GITHUB_RAW}/test_ctf.100.pkl",
    "poses.pkl": f"{_GITHUB_RAW}/hand_rot.pkl",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_hand_dataset(data_dir: str | Path | None = None) -> Path:
    """Download the cryodrgn 'hand' test dataset.

    Contents:
        hand.mrcs   – 100 projection images of a hand volume (64 x 64).
        ctf.pkl     – Per-particle CTF parameters (9 columns).
        poses.pkl   – Rotation matrices for each projection.

    Returns the path to the data directory.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    for local_name, url in _HAND_FILES.items():
        dest = data_dir / local_name
        if not dest.exists():
            print(f"Downloading {local_name} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  -> {dest}")

    return data_dir


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class CryoEMParticles(Distribution):
    """Distribution over cryo-EM particle images.

    Wraps an .mrcs stack loaded via cryodrgn.  Images are stored as
    flattened vectors of shape (N, D*D) so they slot directly into the
    existing SCSI framework (which expects (batch, dim) tensors).
    """

    def __init__(
        self,
        mrcs_path: str | Path,
        device: torch.device = torch.device("cpu"),
        normalize: bool = True,
    ):
        images, header = parse_mrc(str(mrcs_path))
        self.D: int = images.shape[-1]
        self.N: int = images.shape[0]
        self.Apix: float = float(header.apix)

        flat = torch.from_numpy(images.reshape(self.N, -1)).float()
        if normalize:
            self._mean = flat.mean()
            self._std = flat.std()
            flat = (flat - self._mean) / self._std
        else:
            self._mean = torch.tensor(0.0)
            self._std = torch.tensor(1.0)

        self.data = flat.to(device)
        self.device = device

    def sample(self, num_samples: int) -> Tensor:
        idx = torch.randint(0, self.N, (num_samples,))
        return self.data[idx]


# ---------------------------------------------------------------------------
# CTF forward model
# ---------------------------------------------------------------------------

def _freq_grid(D: int, Apix: float) -> Tensor:
    """2-D spatial-frequency grid matching the FFT output layout.

    Uses ``torch.fft.fftfreq`` so that index [i, j] in the grid matches
    index [i, j] in the output of ``torch.fft.fft2``.  This is critical:
    ``fft2`` places DC at [0, 0], NOT at [D//2, D//2].

    Returns a (D*D, 2) tensor of frequencies in units of 1/Angstrom.
    """
    freq_1d = torch.fft.fftfreq(D, d=Apix)
    gy, gx = torch.meshgrid(freq_1d, freq_1d, indexing="ij")
    freqs = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    return freqs


def _compute_ctf_batch(
    freqs: Tensor,
    params: Tensor,
) -> Tensor:
    """Vectorised CTF computation over a batch of parameter sets.

    Args:
        freqs:  (M, 2) frequency grid.
        params: (B, 7) CTF columns [dfu, dfv, dfang, volt, cs, w, phase_shift].

    Returns:
        (B, M) CTF values.
    """
    dfu = params[:, 0]
    dfv = params[:, 1]
    dfang = params[:, 2] * (np.pi / 180)
    volt = params[:, 3] * 1000
    cs = params[:, 4] * 1e7
    w = params[:, 5]
    phase_shift = params[:, 6] * (np.pi / 180)

    lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt ** 2)  # (B,)

    x = freqs[:, 0]  # (M,)
    y = freqs[:, 1]
    ang = torch.arctan2(y, x)  # (M,)
    s2 = x ** 2 + y ** 2  # (M,)

    # Broadcast: (B, 1) op (M,) -> (B, M)
    dfu = dfu[:, None]
    dfv = dfv[:, None]
    dfang = dfang[:, None]
    lam = lam[:, None]
    cs = cs[:, None]
    w = w[:, None]
    phase_shift = phase_shift[:, None]

    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang[None] - dfang)))
    gamma = (
        2 * torch.pi * (-0.5 * df * lam * s2[None] + 0.25 * cs * lam ** 3 * s2[None] ** 2)
        - phase_shift
    )
    ctf = torch.sqrt(1 - w ** 2) * torch.sin(gamma) - w * torch.cos(gamma)
    return ctf  # (B, M)


class CTFForwardModel(ForwardModel):
    """Cryo-EM CTF corruption model.

    Given a clean 2-D image x (in real space, flattened to D*D):
        1. FFT to Fourier space.
        2. Multiply by a randomly sampled CTF.
        3. IFFT back to real space.
        4. Optionally add Gaussian noise.

    CTF parameters are drawn uniformly from the supplied per-particle
    parameter table, so each call to ``__call__`` applies a different
    (random) CTF – matching the real imaging process where every
    particle sees a different defocus.
    """

    def __init__(
        self,
        ctf_params_pkl: str | Path,
        D: int,
        Apix: float,
        noise_sigma: float = 0.0,
    ):
        """
        Args:
            ctf_params_pkl: Path to a cryodrgn CTF .pkl file (N x 9 array
                with columns [D, Apix, dfu, dfv, dfang, volt, cs, w, phase_shift]).
            D:  Image side length in pixels.
            Apix: Pixel size in Angstroms/pixel.
            noise_sigma: Std-dev of additive white Gaussian noise
                (applied *after* CTF, in real space).  0 means no noise.
        """
        raw: np.ndarray = cryodrgn_utils.load_pkl(str(ctf_params_pkl))
        assert raw.shape[1] == 9, f"Expected 9 CTF columns, got {raw.shape[1]}"

        # Columns 2-8 are: dfu, dfv, dfang, volt, cs, w, phase_shift
        self._ctf_table = torch.from_numpy(raw[:, 2:].astype(np.float32))
        self.n_ctfs = self._ctf_table.shape[0]
        self.D = D
        self.Apix = Apix
        self.noise_sigma = noise_sigma
        self._freqs = _freq_grid(D, Apix)

    def __call__(self, x: Tensor) -> Tensor:
        """Apply CTF corruption to a batch of clean (flattened) images.

        Args:
            x: (B, D*D) real-space images.

        Returns:
            (B, D*D) CTF-corrupted images.
        """
        B = x.shape[0]
        D = self.D
        device = x.device

        imgs = x.view(B, D, D)
        imgs_ft = torch.fft.fft2(imgs)

        # Sample a random CTF for each image in the batch
        idx = torch.randint(0, self.n_ctfs, (B,))
        params = self._ctf_table[idx].to(device)         # (B, 7)
        freqs = self._freqs.to(device)                    # (D*D, 2)

        ctf_vals = _compute_ctf_batch(freqs, params)      # (B, D*D)
        ctf_2d = ctf_vals.view(B, D, D)

        corrupted_ft = imgs_ft * ctf_2d
        corrupted = torch.fft.ifft2(corrupted_ft).real

        y = corrupted.view(B, -1)
        if self.noise_sigma > 0:
            y = y + self.noise_sigma * torch.randn_like(y)

        return y


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_hand_dataset(
    data_dir: str | Path | None = None,
    noise_sigma: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> tuple[CryoEMParticles, CTFForwardModel]:
    """Download (if needed) and return the hand dataset + its forward model.

    Returns:
        particles: ``CryoEMParticles`` distribution (100 images, 64 x 64).
        forward_model: ``CTFForwardModel`` with the dataset's CTF params.
    """
    data_dir = download_hand_dataset(data_dir)

    particles = CryoEMParticles(
        data_dir / "hand.mrcs",
        device=device,
        normalize=True,
    )

    forward_model = CTFForwardModel(
        ctf_params_pkl=data_dir / "ctf.pkl",
        D=particles.D,
        Apix=particles.Apix,
        noise_sigma=noise_sigma,
    )

    print(
        f"Loaded hand dataset: {particles.N} images, "
        f"{particles.D}x{particles.D} px, Apix={particles.Apix:.2f}, "
        f"noise_sigma={noise_sigma}"
    )
    return particles, forward_model
