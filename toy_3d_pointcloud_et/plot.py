"""Save a grid of generated point clouds as a PNG (matplotlib imported lazily)."""
from __future__ import annotations

import torch


def save_scatter(clouds: torch.Tensor, path: str, lim: float = 1.6) -> None:
    """clouds: (M, N, 3) tensor (any device). Writes an M-panel 3D scatter PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    pts = clouds.detach().float().cpu().numpy()
    m = pts.shape[0]
    cols = min(m, 4)
    rows = (m + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(m):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(pts[i, :, 0], pts[i, :, 1], pts[i, :, 2], s=2, alpha=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"sample {i}")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")
