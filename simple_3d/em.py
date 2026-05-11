import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import matplotlib.pyplot as plt
from typing import Tuple

from si import loss_func, sample
from corruption import forward_channel
from model import VOL_SIZE

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable


def train_estep(
    model: nn.Module,
    x_pool: torch.Tensor,          # (N, 1, D, H, W)  current 3D volume pool
    noise_std: float,
    style: str,
    epochs: int,
    batch_size: int,
    lr: float,
    global_step: list,
    device: torch.device,
    z_pool: torch.Tensor | None = None,
    coupled_fraction: float = 0.0,
) -> None:
    """
    E-step: train the velocity model on (x ~ x_pool, y = F(x)) pairs.

    y is computed fresh from x_pool each batch — this maintains correct
    (x, F(x)) pairing as x_pool evolves, and fresh random rotations act
    as data augmentation.
    """
    if z_pool is not None and coupled_fraction > 0.0:
        dataset = TensorDataset(x_pool, z_pool)
        has_z = True
    else:
        dataset = TensorDataset(x_pool)
        has_z = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"  E-step epoch {epoch}/{epochs}", leave=False):
            if has_z:
                x_batch, z_batch_coupled = batch
                x_batch = x_batch.to(device)
                z_batch_coupled = z_batch_coupled.to(device)
                B = x_batch.size(0)
                n_coupled = int(round(coupled_fraction * B))
                n_random = B - n_coupled
                if n_coupled == B:
                    z_batch = z_batch_coupled
                elif n_coupled == 0:
                    z_batch = torch.randn_like(x_batch)
                else:
                    z_random = torch.randn(n_random, *x_batch.shape[1:], device=device)
                    z_batch = torch.cat([z_batch_coupled[:n_coupled], z_random], dim=0)
            else:
                (x_batch,) = batch
                x_batch = x_batch.to(device)
                z_batch = None

            # Generate fresh 2D projections from current 3D pool samples
            y_batch = forward_channel(x_batch, noise_std=noise_std)  # (B, 1, H, W)

            loss = loss_func(model, x_batch, y_batch, style=style, z=z_batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

            if global_step is not None:
                wandb.log({"train/loss": loss.item(),
                           "train/grad_norm": grad_norm.item()},
                          step=global_step[0])
                global_step[0] += 1

        sched.step()
        print(f"    epoch {epoch:2d}  |  loss = {running / len(loader):.5f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def update_prior(
    model: nn.Module,
    y_obs: torch.Tensor,           # (N, 1, H, W)  2D observations
    n_steps: int = 50,
    batch_size: int = 32,
    method: str = "euler",
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    M-step: sample new 3D volumes conditioned on 2D observations.

    Initial state is drawn from N(0, I) with shape (B, 1, D, H, W).
    Returns:
        x_new:  (N, 1, D, H, W)
        z_pool: (N, 1, D, H, W)  initial noise (for coupling)
    """
    model.eval()
    N = y_obs.size(0)
    D = H = W = VOL_SIZE
    chunks, z_chunks = [], []
    y_gpu = y_obs.to(device)

    for start in tqdm(range(0, N, batch_size), desc="  M-step", leave=False):
        end = min(start + batch_size, N)
        B = end - start
        initial_state = torch.randn(B, 1, D, H, W, device=device)
        x_batch = sample(model, initial_state, y_gpu[start:end],
                         n_steps=n_steps, method=method)
        chunks.append(x_batch.cpu())
        z_chunks.append(initial_state.cpu())

    result = torch.cat(chunks, dim=0)    # (N, 1, D, H, W)
    z_pool = torch.cat(z_chunks, dim=0)  # (N, 1, D, H, W)
    print(f"    prior range=[{result.min():.3f}, {result.max():.3f}]"
          f"  mean={result.mean():.4f}  std={result.std():.4f}")

    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    return result, z_pool


def log_em_step_wandb(
    x_gt: torch.Tensor,      # (N, 1, D, H, W)
    y_obs: torch.Tensor,     # (N, 1, H, W)
    x_pool: torch.Tensor,    # (N, 1, D, H, W)
    em_step: int,
    n: int = 8,
) -> None:
    """
    Log EM step visuals to W&B.

    Row 0: GT volume mid-slice (z = D//2)
    Row 1: 2D projection observations y_obs
    Row 2: Pool volume mid-slice (reconstruction quality)
    Row 3: Pool projection (consistency with y_obs)
    """
    mid_d = VOL_SIZE // 2
    gt_slices   = x_gt[:n, 0, mid_d, :, :].cpu()
    pool_slices = x_pool[:n, 0, mid_d, :, :].cpu()
    obs         = y_obs[:n, 0].cpu()
    pool_proj   = x_pool[:n, 0].sum(dim=1).cpu()    # rough projection of pool

    all_data = torch.cat([gt_slices.flatten(), obs.flatten()])
    vmin, vmax = float(all_data.min()), float(all_data.max())
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8

    rows = [
        (gt_slices,   "GT slice"),
        (obs,         "Obs F(X)"),
        (pool_slices, f"π({em_step}) slice"),
        (pool_proj,   f"π({em_step}) proj"),
    ]
    fig, axes = plt.subplots(4, n + 1, figsize=(2 * (n + 1), 8))
    for r, (data, label) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
        for j in range(n):
            axes[r, j].imshow(data[j].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            for spine in axes[r, j].spines.values():
                spine.set_visible(False)
        hist_ax = axes[r, n]
        hist_ax.hist(data.numpy().ravel(), bins=50, color="steelblue", edgecolor="none")
        hist_ax.set_xlim(vmin, vmax)
        hist_ax.tick_params(axis="both", labelsize=7)
        hist_ax.set_yticks([])

    fig.suptitle(f"EM step {em_step}", fontsize=12)
    plt.tight_layout()
    wandb.log({"em/reconstruction": wandb.Image(fig)}, step=em_step)
    plt.close(fig)
