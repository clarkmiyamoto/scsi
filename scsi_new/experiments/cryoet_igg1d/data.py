"""
CryoBench IgG-1D for SCSI: the observed particle images, the warm start, and a GT eval pool.

IgG-1D is 100 conformations of an IgG antibody (a 1D circular motion, 3.6 deg per conformation),
1000 simulated particle images each, released at 128 px / 3.0 A/px with uniform SO(3) poses,
+-30 A shifts, experimental CTFs and SNR 0.01. Loading reuses ~/CryoBench/cryobench_data/igg1d.py
(read_mrc, fourier_downsample, IgG1DDataset), the same module the supervised cryofm run uses.

What SCSI sees (build_observations): the particle images and their CTF parameters, nothing else.
Images are Fourier-cropped to `resolution`, phase-flipped with their own CTF (as cryofm's
phase_flip=True) and divided by the noise std estimated from the image corners, so the channel
(corruption.py) adds unit-variance noise. The split matches cryofm's exactly (split_seed,
val_fraction), so the observations are cryofm's training images and the eval pool comes from its
held-out ones.

What only the metrics see (build_eval_pool): IgG1DDataset(volume_frame="image") targets -- the GT
volume under each image's own pose, so its z-projection underlies the image -- plus the GT
conformation label, pose and the 100 canonical volumes. With --eval_n 16 the pool is exactly the
images cryofm's sampling eval uses (spread over the held-out set).

`python data.py` checks the forward model against the real images (GT used there only).
"""

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from corruption import ctf_2d, image_formation, phase_flip
from pseudoinverse import estimate_ssnr, pseudoinverse

DEFAULT_DATA_ROOT = "/mnt/ceph/users/cmiyamoto/IgG-1D"
DEFAULT_CRYOBENCH_ROOT = "/mnt/home/cmiyamoto/CryoBench"
ORIG_D = 128        # released box size (px); cryobench_data.igg1d.ORIG_D
BOX_A = 384.0       # physical box (A), unchanged by Fourier cropping

# Images per Fourier-crop / phase-flip call while loading observations.
_LOAD_CHUNK = 1000


@dataclass
class Config_Dataset_IgG:
    data_root: str = DEFAULT_DATA_ROOT
    cryobench_root: str = DEFAULT_CRYOBENCH_ROOT
    resolution: int = 64                 # volumes resolution^3, images resolution^2
    snr: float = 0.01                    # reads images/snr{snr}/
    val_fraction: float = 0.02           # held out as in cryofm; the eval pool comes from these
    split_seed: int = 0
    n_observations: int | None = None    # random subset of the training images; None = all
    seed: int = 42                       # draws that subset

    # Corruption channel
    shift_extent_A: float = 30.0         # CryoBench's project3d --t-extent 20 at 1.5 A/px

    # Warm start (pseudoinverse.py): z-extent of the backprojection; None = the whole box
    pinv_diameter_A: float | None = 192.0

    @property
    def apix(self) -> float:
        return BOX_A / self.resolution

    @property
    def shift_extent_px(self) -> float:
        return self.shift_extent_A / self.apix


def import_cryobench(root: str):
    """cryobench_data.igg1d from a CryoBench checkout (not an installed package)."""
    if root not in sys.path:
        sys.path.insert(0, root)
    import cryobench_data.igg1d as cb
    return cb


def split_indices(n_images: int, config: Config_Dataset_IgG) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx), both sorted -- the same sets as cryofm's scripts/train_igg1d.py."""
    perm = np.random.default_rng(config.split_seed).permutation(n_images)
    n_val = int(round(config.val_fraction * n_images))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def spread(indices: np.ndarray, n: int) -> np.ndarray:
    """n entries of `indices` evenly spaced along it (all of them if n >= len); cryofm's helper."""
    if n >= len(indices):
        return indices
    return indices[np.linspace(0, len(indices) - 1, n).round().astype(int)]


def _stack_paths(config: Config_Dataset_IgG, cb) -> tuple[list[str], np.ndarray]:
    img_dir = Path(config.data_root) / "images" / f"snr{config.snr}"
    with open(img_dir / f"sorted_particles.{ORIG_D}.txt") as f:
        paths = [str(img_dir / line.strip()) for line in f if line.strip()]
    offsets = np.concatenate([[0], np.cumsum([cb.read_mrc_count(p) for p in paths])])
    return paths, offsets


def load_ctf_params(config: Config_Dataset_IgG) -> torch.Tensor:
    """(N, 7) per-image CTF parameters in ctf_2d layout, from combined_ctfs.pkl."""
    with open(Path(config.data_root) / "combined_ctfs.pkl", "rb") as f:
        ctf = np.array(pickle.load(f), dtype=np.float32)   # [D, Apix, dfU, dfV, dfang, kV, Cs, w, ps]
    return torch.from_numpy(ctf[:, 2:])


def _corner_mask(D: int) -> torch.Tensor:
    idx = torch.arange(D, dtype=torch.float32) - D // 2
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    return torch.hypot(xx, yy) > D / 2


def build_observations(config: Config_Dataset_IgG) -> tuple[TensorDataset, dict]:
    """
    The observed images, preprocessed as the channel's output: Fourier-cropped, phase-flipped,
    divided by the corner-pixel noise std.

    Returns:
        TensorDataset of one (N, 1, D, D) tensor, and a dict with
            ctf_pool (N, 7)  the observations' CTF parameters, in the same order
            sigma            the noise std the raw (cropped, flipped) images were divided by
            apix, index      pixel size (A) and the observations' global image indices
    """
    cb = import_cryobench(config.cryobench_root)
    D, apix = config.resolution, config.apix
    paths, offsets = _stack_paths(config, cb)
    train_idx, _ = split_indices(int(offsets[-1]), config)
    idx = train_idx
    if config.n_observations is not None and config.n_observations < len(idx):
        rng = np.random.default_rng(config.seed)
        idx = np.sort(rng.choice(idx, config.n_observations, replace=False))
    ctf_pool = load_ctf_params(config)[torch.from_numpy(idx)]

    images = torch.empty(len(idx), 1, D, D)
    stack_of = np.searchsorted(offsets, idx, side="right") - 1
    for k in np.unique(stack_of):
        rows = np.nonzero(stack_of == k)[0]
        stack = cb.read_mrc(paths[k], mmap=True)
        for c in range(0, len(rows), _LOAD_CHUNK):
            r = rows[c:c + _LOAD_CHUNK]
            raw = torch.from_numpy(np.array(stack[idx[r] - offsets[k]]))
            y = cb.fourier_downsample(raw, D, ndim=2)[:, None]
            images[r] = phase_flip(y, ctf_2d(D, apix, ctf_pool[r]))
        del stack

    corners = images[:, 0, _corner_mask(D)].double()
    sigma = (corners - corners.mean()).pow(2).mean().sqrt().item()
    images /= sigma
    print(f"observations: {len(idx)} images at {D} px ({apix:.2f} A/px), noise std "
          f"{sigma:.4g} from the corners (r > D/2)", flush=True)
    return TensorDataset(images), {"ctf_pool": ctf_pool, "sigma": sigma, "apix": apix,
                                   "index": idx}


class PseudoinverseVolumes(Dataset):
    """
    Warm-start volumes x0_i = pseudoinverse(y_i) / vol_gain, computed per item on `device`: all
    of them at 64^3 would be ~100 GB. Wrap in scsi.ResampledPairs for (target, F(x0)) pairs.
    """

    def __init__(self, observations: TensorDataset, ctf_pool: torch.Tensor, apix: float,
                 config: Config_Dataset_IgG, vol_gain: float, device: torch.device):
        (self.y,) = observations.tensors
        self.ctf_pool, self.apix, self.config = ctf_pool, apix, config
        self.vol_gain, self.device = vol_gain, device
        self.ssnr = estimate_ssnr(self.y, ctf_pool, apix, device)

    def __len__(self) -> int:
        return self.y.size(0)

    def __getitem__(self, i: int) -> torch.Tensor:
        x = pseudoinverse(self.y[i:i + 1].to(self.device), self.ctf_pool[i:i + 1].to(self.device),
                          self.apix, self.config, self.ssnr)
        return x[0] / self.vol_gain


@torch.no_grad()
def calibrate_gain(observations: TensorDataset, ctf_pool: torch.Tensor, apix: float,
                   config: Config_Dataset_IgG, device: torch.device, n: int = 1024,
                   batch_size: int = 64) -> float:
    """
    Default --vol_gain: the std of the gain-1 pseudoinverse volumes of n observations spread over
    all of them (they are sorted by conformation), so the warm-start volumes have unit std (as
    cryofm standardises volumes by vol_std, but without GT). Deterministic given the observations.
    """
    (y,) = observations.tensors
    ssnr = estimate_ssnr(y, ctf_pool, apix, device)
    idx = torch.linspace(0, y.size(0) - 1, min(n, y.size(0))).round().long()
    total, total_sq, count = 0.0, 0.0, 0
    for chunk in idx.split(batch_size):
        x = pseudoinverse(y[chunk].to(device), ctf_pool[chunk].to(device), apix, config, ssnr).double()
        total, total_sq, count = total + x.sum().item(), total_sq + x.pow(2).sum().item(), count + x.numel()
    return float(np.sqrt(total_sq / count - (total / count) ** 2))


def build_eval_pool(config: Config_Dataset_IgG, n_pool: int, seed: int, sigma: float) -> dict:
    """
    Fixed held-out images with their GT, for eval_metrics and the wandb panels. Uses a private
    generator for x0, so building it leaves the training RNG stream untouched.

    Returns a dict with
        x_gt  (n, 1, D, D, D)  image-frame GT targets (GT units)
        y     (n, 1, D, D)     the images, preprocessed exactly as build_observations
        x0    (n, 1, D, D, D)  fixed ODE initial noise
        ctf   (n, 7)           CTF parameters
        label (n,)             conformation index
        rot, shift             (n, 3, 3) poses and (n, 2) in-plane shifts (px), for posing refs
        gt_volumes (C, D, D, D) the 100 canonical conformations
    """
    cb = import_cryobench(config.cryobench_root)
    ds = cb.IgG1DDataset(config.data_root, resolution=config.resolution, snr=config.snr,
                         return_meta=True, invert=False, volume_frame="image", phase_flip=True)
    _, val_idx = split_indices(len(ds), config)
    idx = spread(val_idx, n_pool)
    items = [ds[int(i)] for i in idx]
    D = config.resolution
    g = torch.Generator().manual_seed(seed)
    return {
        "x_gt": torch.stack([x for x, _, _ in items])[:, None],
        "y": torch.stack([y for _, y, _ in items])[:, None] / sigma,
        "x0": torch.randn(len(idx), 1, D, D, D, generator=g),
        "ctf": ds.ctf[torch.from_numpy(idx), 2:],
        "label": ds.labels[torch.from_numpy(idx)],
        "rot": ds.rotations[torch.from_numpy(idx)],
        "shift": torch.stack([ds.image_shift(int(i)) for i in idx]),
        "gt_volumes": ds.volumes,
    }


if __name__ == "__main__":
    # Forward-model check against the real images. For training images with their GT
    # image-frame targets, the channel's noiseless image |CTF| * P(target) should match the
    # observed (phase-flipped, noise-normalised) image up to one global scale a > 0, leaving
    # residual std ~1. A wrong projection axis, image transpose, CTF sign or phase flip shows up
    # as a matched correlation no better than a mismatched or transposed one.
    import argparse

    parser = argparse.ArgumentParser(description="Check corruption.py against the real IgG-1D images")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cryobench_root", default=DEFAULT_CRYOBENCH_ROOT)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n_sigma", type=int, default=2000,
                        help="Observations used to estimate the noise std (all of them in main.py).")
    parser.add_argument("--n_check", type=int, default=64)
    parser.add_argument("--save", type=str, default=None, help="Write a PNG of example panels here.")
    args = parser.parse_args()

    config = Config_Dataset_IgG(data_root=args.data_root, cryobench_root=args.cryobench_root,
                                resolution=args.resolution, n_observations=args.n_sigma)
    observations, info = build_observations(config)
    y_obs = observations.tensors[0]
    corner = y_obs[:, 0, _corner_mask(config.resolution)]
    print(f"normalised observations: std {y_obs.std():.3f}, corner std {corner.std():.3f}")

    cb = import_cryobench(config.cryobench_root)
    ds = cb.IgG1DDataset(config.data_root, resolution=config.resolution, snr=config.snr,
                         return_meta=True, volume_frame="image", phase_flip=True)
    check_idx = info["index"][:args.n_check]
    items = [ds[int(i)] for i in check_idx]
    x_gt = torch.stack([x for x, _, _ in items])[:, None]
    y = torch.stack([y for _, y, _ in items])[:, None] / info["sigma"]
    f0 = image_formation(x_gt, ds.ctf[torch.from_numpy(check_idx), 2:], config.apix)

    a = ((f0 * y).sum() / (f0 * f0).sum()).item()
    resid = y - a * f0

    def corr(u, v):
        u, v = u.flatten(1), v.flatten(1)
        u, v = u - u.mean(1, keepdim=True), v - v.mean(1, keepdim=True)
        return ((u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1)).clamp_min(1e-12)).mean().item()

    print(f"scale a = {a:.4g} (should be > 0); residual std {resid.std():.3f} (should be ~1)")
    print(f"corr(y, |CTF| P(target)): matched {corr(y, f0):.3f}   "
          f"mismatched {corr(y, f0.roll(1, 0)):.3f}   transposed {corr(y, f0.transpose(-1, -2)):.3f}   "
          f"noiseless ceiling ~{corr(a * f0, a * f0 + torch.randn_like(f0)):.3f}")
    vol_std = ds.volumes.std().item()
    print(f"GT-derived --vol_gain (model volumes = GT / its std, like cryofm's vol_std "
          f"{vol_std:.4g}): {a * vol_std:.4g}")
    pinv_gain = calibrate_gain(observations, info["ctf_pool"], config.apix, config, torch.device("cpu"))
    print(f"pseudoinverse-calibrated --vol_gain (main.py's default, --pinv_diameter_A "
          f"{config.pinv_diameter_A}): {pinv_gain:.4g}; `python pseudoinverse.py` scores the warm start")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(6, len(check_idx))
        rows = [("y (observed)", y), ("a |CTF| P(target)", a * f0), ("residual", resid),
                ("target proj", x_gt.sum(-3))]
        fig, axes = plt.subplots(len(rows), n, figsize=(2 * n, 2 * len(rows)), squeeze=False)
        for r, (name, imgs) in enumerate(rows):
            for j in range(n):
                img = imgs[j, 0].numpy()
                lo, hi = np.percentile(img, [1, 99])
                axes[r, j].imshow(img, cmap="gray", vmin=lo, vmax=hi)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            axes[r, 0].set_ylabel(name, fontsize=8)
        fig.tight_layout()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=110)
        print(f"saved {args.save}")
