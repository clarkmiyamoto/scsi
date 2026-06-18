"""Optional wandb logging. All methods are silent no-ops when enabled=False,
so callers never need an `if wandb:` guard."""
import torch


class WandbLogger:

    def __init__(self, enabled: bool, config: dict | None = None,
                 project: str = "scsi-image-2d-jiequn", **init_kwargs):
        self.enabled = enabled
        if enabled:
            import wandb as _wandb
            self._wandb = _wandb
            _wandb.init(project=project, config=config, **init_kwargs)

    def log_step(self, step: int, loss: float, grad_norm: float, phase: str) -> None:
        if not self.enabled:
            return
        self._wandb.log(
            {"train/loss": loss, "train/grad_norm": grad_norm, "train/phase": phase},
            step=step,
        )

    def log_recon(self, step: int, gt: torch.Tensor, fbp: torch.Tensor,
                  recon: torch.Tensor, n: int = 8) -> None:
        if not self.enabled:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _to_grid(t):
            imgs = t[:n].detach().cpu().float()
            # [-1, 1] -> [0, 1], squeeze channel dim
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            if imgs.ndim == 4:
                imgs = imgs[:, 0]  # (N, H, W)
            return imgs

        rows = [_to_grid(x) for x in (gt, fbp, recon)]
        labels = ["GT", "FBP", "Recon"]
        ncols = min(n, rows[0].shape[0])
        fig, axes = plt.subplots(3, ncols, figsize=(ncols * 1.5, 4.5))
        for r, (imgs, label) in enumerate(zip(rows, labels)):
            for c in range(ncols):
                ax = axes[r, c] if ncols > 1 else axes[r]
                ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if c == 0:
                    ax.set_title(label, fontsize=8)
        plt.tight_layout()
        self._wandb.log({"viz/recon": self._wandb.Image(fig)}, step=step)
        plt.close(fig)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._wandb.finish()
