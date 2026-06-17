"""MNIST loading + on-the-fly CryoET tilt-series corruption.

Minimal port of ``get_dataset('mnist', pm1=True)`` + ``CorruptedDataset`` from
``src/custom_datasets.py``. MNIST is padded to 32x32 and normalized to ``[-1, 1]``
(pm1). The corruption is applied lazily in ``__getitem__`` with a per-index
*tied* RNG so each image's unknown global rotation ``theta0`` is fixed across
epochs — this stabilizes the observation of image ``i`` and well-poses the
self-consistency bootstrap.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_mnist_pm1(data_root: str):
    """torchvision MNIST, padded 28->32, normalized to ``[-1, 1]`` (returns
    ``(img, label)`` pairs; the corruption wrapper drops the label)."""
    tf = transforms.Compose([
        transforms.Pad(2),                       # 28x28 -> 32x32
        transforms.ToTensor(),                   # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),    # -> [-1, 1]
    ])
    return datasets.MNIST(root=str(Path(data_root) / "mnist"),
                          train=True, download=True, transform=tf)


class CorruptedTiltDataset(Dataset):
    """Wrap a clean image dataset and apply the tilt-series forward map on the fly.

    ``__getitem__(idx)`` returns ``(clean, z_out, cond)``:
        clean : ``[1, 32, 32]`` pm1 MNIST image
        z_out : ``[1, 32, 32]`` standard-Gaussian forward-model prior (the SI
                ``t=1`` endpoint; the trainer treats this as the observation)
        cond  : ``[K, 32, 32]`` tiled tilt-series projections (model conditioning)
    """

    def __init__(self, base, fwd, base_seed: int = 42):
        self.base = base
        self.fwd = fwd
        self.base_seed = base_seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base[idx]
        if isinstance(img, (tuple, list)):       # (image, label) -> image
            img = img[0]
        gen = torch.Generator().manual_seed(self.base_seed + int(idx))
        z_out, cond = self.fwd(img, return_latents=True, generator=gen)
        return img, z_out, cond
