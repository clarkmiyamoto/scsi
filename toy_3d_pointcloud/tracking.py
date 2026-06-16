"""Thin Weights & Biases wrapper.

Everything is a no-op unless `enabled=True`, so the rest of the code never has to
branch on whether logging is on. Point clouds are logged as `wandb.Object3D`,
which renders an interactive (rotatable / zoomable) 3D viewer in the W&B UI.
"""
from __future__ import annotations

import numpy as np
import torch


def _colorize(cloud: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 6) [x, y, z, r, g, b] with a height (z) color ramp.

    wandb.Object3D reads columns 4-6 as RGB in [0, 255].
    """
    z = cloud[:, 2]
    zc = (z - z.min()) / (np.ptp(z) + 1e-8)  # normalize height to [0, 1]
    rgb = np.stack([zc * 255, zc * 255, (1 - zc) * 255], axis=1)  # blue -> yellow
    return np.concatenate([cloud, rgb], axis=1).astype(np.float32)


class Tracker:
    """W&B run wrapper; all methods are safe no-ops when disabled."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "pointcloud-fm",
        name: str | None = None,
        config: dict | None = None,
        job_type: str | None = None,
    ):
        self.enabled = enabled
        self.run = None
        self.wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "wandb not installed. Add it (`uv add wandb`) or drop --wandb."
            ) from exc
        self.wandb = wandb
        self.run = wandb.init(
            project=project, name=name, config=config or {}, job_type=job_type
        )

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.wandb.log(data, step=step)

    def log_clouds(
        self, key: str, clouds: torch.Tensor, step: int | None = None, color: bool = True
    ) -> None:
        """Log a batch of clouds (M, N, 3) as interactive 3D objects."""
        if self.run is None:
            return
        arrs = clouds.detach().float().cpu().numpy()
        objs = [
            self.wandb.Object3D(_colorize(c) if color else c) for c in arrs
        ]
        self.wandb.log({key: objs}, step=step)

    def log_meshes(self, key: str, paths: list[str], step: int | None = None) -> None:
        """Log .obj mesh files as interactive 3D objects (e.g. union-of-balls)."""
        if self.run is None:
            return
        objs = []
        for p in paths:
            try:
                objs.append(self.wandb.Object3D.from_file(p))
            except Exception:  # older API: construct from an open file handle
                objs.append(self.wandb.Object3D(open(p)))
        self.wandb.log({key: objs}, step=step)

    def log_image(self, key: str, path: str, step: int | None = None) -> None:
        if self.run is not None:
            self.wandb.log({key: self.wandb.Image(path)}, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    # context-manager sugar so callers can `with Tracker(...) as t:`
    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
