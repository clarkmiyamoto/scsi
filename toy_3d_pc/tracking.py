"""Weights & Biases tracking, ON by default.

W&B logging is enabled unless the caller passes ``enabled=False`` (the CLI exposes
``--no-wandb``). Initialization is wrapped so that a missing/unconfigured ``wandb``
never crashes a run: the ``Tracker`` warns once and degrades to a silent no-op, so
the rest of the code never has to branch on whether logging actually came up.

Point clouds are logged as ``wandb.Object3D`` (an interactive rotatable 3D viewer).
"""
from __future__ import annotations

import numpy as np
import torch


def _colorize(cloud: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 6) [x, y, z, r, g, b] with a height (z) color ramp."""
    z = cloud[:, 2]
    zc = (z - z.min()) / (np.ptp(z) + 1e-8)
    rgb = np.stack([zc * 255, zc * 255, (1 - zc) * 255], axis=1)  # blue -> yellow
    return np.concatenate([cloud, rgb], axis=1).astype(np.float32)


class Tracker:
    """W&B run wrapper; enabled by default, all methods safe no-ops if logging is off."""

    def __init__(
        self,
        enabled: bool = True,
        project: str = "toy3d-pc-scsi",
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

            self.run = wandb.init(
                project=project, name=name, config=config or {}, job_type=job_type
            )
            self.wandb = wandb
        except Exception as exc:  # missing wandb, no login, offline failure, ...
            print(f"[wandb] disabled ({type(exc).__name__}: {exc}); continuing without logging")
            self.enabled = False
            self.run = None
            self.wandb = None

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
        objs = [self.wandb.Object3D(_colorize(c) if color else c) for c in arrs]
        self.wandb.log({key: objs}, step=step)

    def log_meshes(self, key: str, paths: list[str], step: int | None = None) -> None:
        """Log .obj mesh files as interactive 3D objects (e.g. union-of-balls)."""
        if self.run is None:
            return
        objs = []
        for p in paths:
            try:
                objs.append(self.wandb.Object3D.from_file(p))
            except Exception:
                objs.append(self.wandb.Object3D(open(p)))
        self.wandb.log({key: objs}, step=step)

    def log_image(self, key: str, path: str, step: int | None = None) -> None:
        if self.run is not None:
            self.wandb.log({key: self.wandb.Image(path)}, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
