import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corruption import corruption_channel
from ode import euler_integration_trajectory

POINT_DIM = 2  # this experiment is fixed at 2D


def _to_xy(t: torch.Tensor) -> torch.Tensor:
    """(N, dim, 1, 1) -> (N, dim) on CPU."""
    return t.reshape(t.size(0), POINT_DIM).detach().cpu()


### Headline panel: does the recovered pool cover the full ground truth? ###

@torch.no_grad()
def log_distribution_scatter(x_hat_pool: torch.Tensor, x_gt_ref: torch.Tensor,
                             warmup_ref: torch.Tensor, em_step: int, wandb_step: int,
                             panel_name: str = "pool"):
    """
    The main "did SCSI recover the full ground-truth distribution" panel: overlays the
    ground-truth reference pool, the (fixed) warm-start prior's samples, and the CURRENT bulk
    E-step proposal pool (x_hat_pool -- e.g. scsi.estep()'s x1s, passed in by the caller so this
    reuses the algorithm's own proposals rather than re-integrating). Mode coverage/collapse is
    visible directly here, unlike a handful of per-example panels.
    """
    x_hat_xy = _to_xy(x_hat_pool).numpy()
    x_gt_xy = _to_xy(x_gt_ref).numpy()
    warmup_xy = _to_xy(warmup_ref).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(warmup_xy[:, 0], warmup_xy[:, 1], s=6, alpha=0.15, color="tab:orange", label="warmup prior")
    ax.scatter(x_gt_xy[:, 0], x_gt_xy[:, 1], s=8, alpha=0.35, color="tab:gray", label="ground truth")
    ax.scatter(x_hat_xy[:, 0], x_hat_xy[:, 1], s=8, alpha=0.35, color="tab:blue", label="x_hat (recovered)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{panel_name} distribution | EM step {em_step}", fontsize=11)
    plt.tight_layout()

    wandb.log({f"viz/{panel_name}/distribution": wandb.Image(fig), "em/step": em_step}, step=wandb_step)
    plt.close(fig)


### Recovery metrics, as scalar curves over EM steps ########################

def energy_distance(x: torch.Tensor, y: torch.Tensor, max_n: int = 2000) -> float:
    """
    2-sample energy distance between point clouds x (N, 2) and y (M, 2): 0 iff the two
    distributions coincide, and it decreases monotonically towards 0 as they converge -- unlike
    eyeballing scatter plots, this gives one number to track per EM step. O(n^2) via torch.cdist,
    so both sides are subsampled to `max_n` to keep it cheap to log every step.
    """
    if x.size(0) > max_n:
        x = x[torch.randperm(x.size(0))[:max_n]]
    if y.size(0) > max_n:
        y = y[torch.randperm(y.size(0))[:max_n]]
    d_xy, d_xx, d_yy = torch.cdist(x, y).mean(), torch.cdist(x, x).mean(), torch.cdist(y, y).mean()
    return (2 * d_xy - d_xx - d_yy).item()


def mode_occupancy(x_hat: torch.Tensor, mode_centers: torch.Tensor) -> torch.Tensor:
    """Fraction of x_hat points nearest each mode center (nearest-center assignment). (K,), sums to 1."""
    assign = torch.cdist(x_hat, mode_centers).argmin(dim=1)
    counts = torch.bincount(assign, minlength=mode_centers.size(0)).float()
    return counts / counts.sum().clamp_min(1)


@torch.no_grad()
def log_distribution_metrics(x_hat_pool: torch.Tensor, x_gt_ref: torch.Tensor,
                             mode_centers: torch.Tensor | None, em_step: int, wandb_step: int):
    """
    Scalar recovery metrics against the CLEAN ground-truth reference pool (never seen by
    training): an overall 2-sample distance, plus -- when the ground truth has discrete modes
    (ring_gmm / gmm with gt_means) -- how many modes the proposal pool actually covers. This is
    the direct, single-number answer to "recovered the FULL distribution" that the scatter plot
    alone can't give you.
    """
    x_hat_xy, x_gt_xy = _to_xy(x_hat_pool), _to_xy(x_gt_ref)
    log = {"metrics/energy_distance": energy_distance(x_hat_xy, x_gt_xy), "em/step": em_step}

    if mode_centers is not None:
        mass = mode_occupancy(x_hat_xy, mode_centers)
        log["metrics/n_modes_covered"] = (mass > 0.01).sum().item()
        log["metrics/mode_mass_std"] = mass.std().item()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(mass.size(0)), mass.numpy())
        ax.axhline(1.0 / mass.size(0), color="red", linestyle="--", linewidth=1, label="uniform")
        ax.set_xlabel("mode index"); ax.set_ylabel("mass fraction"); ax.legend(fontsize=8)
        ax.set_title(f"mode occupancy | EM step {em_step}", fontsize=10)
        plt.tight_layout()
        log["viz/mode_occupancy"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(log, step=wandb_step)


### Cheap per-example diagnostic: ODE paths in state space #################

@torch.no_grad()
def log_trajectory_lines(model, x0, y, n_steps_sampling, n_snapshots, n_rows, em_step,
                         wandb_step, panel_name, device):
    """
    State-space analog of the image experiments' log_trajectory_grid: for n_rows points, draws
    the full path from x0 (noise, x-marker) to x1 (recovered point, o-marker) as a connected
    line, colored by progress along t in [0, 1].
    """
    x0, y = x0[:n_rows].to(device), y[:n_rows].to(device)
    traj = euler_integration_trajectory(model, x0, y, n_steps_sampling, n_snapshots)  # (S, n_rows, dim, 1, 1)
    traj = traj.reshape(traj.size(0), traj.size(1), POINT_DIM).cpu().numpy()
    S, n = traj.shape[0], traj.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("viridis")
    for i in range(n):
        path = traj[:, i]  # (S, 2)
        for s in range(S - 1):
            ax.plot(path[s:s + 2, 0], path[s:s + 2, 1], color=cmap(s / max(S - 2, 1)), linewidth=1.2)
        ax.scatter(path[0, 0], path[0, 1], marker="x", color="black", s=25, zorder=3)
        ax.scatter(path[-1, 0], path[-1, 1], marker="o", color="red", s=25, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(f"{panel_name} trajectories (x=noise, o=recovered) | EM step {em_step}", fontsize=10)
    plt.tight_layout()

    wandb.log({f"viz/{panel_name}/trajectory": wandb.Image(fig), "em/step": em_step}, step=wandb_step)
    plt.close(fig)


def random_draw(pool: dict, config_dataset, n: int) -> dict:
    idx = torch.randperm(pool["x_gt"].size(0))[:n]
    x_gt = pool["x_gt"][idx]
    x0 = torch.randn(n, POINT_DIM, 1, 1)
    y = corruption_channel(x_gt, noise_std=config_dataset.noise_std)
    return {"x_gt": x_gt, "x0": x0, "y": y}
