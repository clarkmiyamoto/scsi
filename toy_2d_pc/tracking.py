"""Weights & Biases tracking, ON by default.

W&B logging is enabled unless the caller passes ``enabled=False`` (the CLI exposes
``--no-wandb``). Initialization is wrapped so that a missing/unconfigured ``wandb``
never crashes a run: the ``Tracker`` warns once and degrades to a silent no-op, so
the rest of the code never has to branch on whether logging actually came up.

Unlike ``toy_3d_pc``'s tracker there is no ``log_clouds``/``log_meshes`` (those log
interactive 3D objects to W&B); 2D point clouds and sinograms are visualized as
matplotlib PNG panels and logged via ``log_image`` instead.
"""
from __future__ import annotations


class Tracker:
    """W&B run wrapper; enabled by default, all methods safe no-ops if logging is off."""

    def __init__(
        self,
        enabled: bool = True,
        project: str = "toy2d-pc-scsi",
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
