"""Viz utilities for the direct-optimization loop: 3D point-cloud scatter (used for
both parameterizations) and a real-vs-simulated tilt-image montage (matplotlib,
headless)."""
from __future__ import annotations

import torch


def save_scatter(cloud: torch.Tensor, path: str, lim: float | None = None) -> None:
    """cloud: (N, 3) -> single-panel 3D scatter PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = cloud.detach().float().cpu().numpy()
    lim = lim if lim is not None else float(abs(pts).max()) * 1.2 + 1e-3
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_tilt_montage(
    y_real: torch.Tensor, y_sim: torch.Tensor, path: str, max_tilts: int = 6
) -> None:
    """y_real/y_sim: (T, P, P) single object -> 2-row montage, real (top) vs sim
    (bottom). Poses differ between the two (real has its own fixed unknown pose,
    sim uses a fresh random one), so this is a qualitative appearance check only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(y_real.shape[0], max_tilts)
    real = y_real.detach().float().cpu().numpy()
    sim = y_sim.detach().float().cpu().numpy()
    vmin = min(real[:T].min(), sim[:T].min())
    vmax = max(real[:T].max(), sim[:T].max())
    fig, axes = plt.subplots(2, T, figsize=(2 * T, 4), squeeze=False)
    for t in range(T):
        axes[0, t].imshow(real[t], cmap="gray", vmin=vmin, vmax=vmax); axes[0, t].set_axis_off()
        axes[1, t].imshow(sim[t], cmap="gray", vmin=vmin, vmax=vmax); axes[1, t].set_axis_off()
    axes[0, 0].set_ylabel("real", rotation=0, labelpad=20)
    axes[1, 0].set_ylabel("sim", rotation=0, labelpad=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
