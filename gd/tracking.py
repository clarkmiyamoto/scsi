"""Thin, always-on Weights & Biases wrapper.

Unlike the sibling packages' `Tracker` (which no-ops when disabled), this experiment
assumes W&B is always available, so every method logs unconditionally.
"""
from __future__ import annotations

import numpy as np
import torch
import wandb


def _colorize(cloud: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 6) [x, y, z, r, g, b] with a height (z) color ramp."""
    z = cloud[:, 2]
    zc = (z - z.min()) / (np.ptp(z) + 1e-8)
    rgb = np.stack([zc * 255, zc * 255, (1 - zc) * 255], axis=1)
    return np.concatenate([cloud, rgb], axis=1).astype(np.float32)


class Tracker:
    """W&B run wrapper. All methods log directly -- no enabled/disabled branching."""

    def __init__(self, project: str = "gd-cryoet", name: str | None = None, config: dict | None = None):
        self.run = wandb.init(project=project, name=name, config=config or {})

    def log(self, data: dict, step: int | None = None) -> None:
        wandb.log(data, step=step)

    def log_image(self, key: str, path: str, step: int | None = None) -> None:
        wandb.log({key: wandb.Image(path)}, step=step)

    def log_cloud(self, key: str, cloud: torch.Tensor, step: int | None = None, color: bool = True) -> None:
        arr = cloud.detach().float().cpu().numpy()
        obj = wandb.Object3D(_colorize(arr) if color else arr)
        wandb.log({key: obj}, step=step)

    def finish(self) -> None:
        self.run.finish()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
