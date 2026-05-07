import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import matplotlib.pyplot as plt
from typing import Tuple

from si import loss_func, sample
from corruption import forward_channel
from model import IMAGE_SIZE

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

def train_estep(model: nn.Module, x_pool: torch.Tensor,
                noise_std: float, p_drop: float, corruption: str, style: str,
                epochs: int, batch_size: int, lr: float,
                global_step: list,
                device: torch.device,
                z_pool: torch.Tensor | None = None,
                coupled_fraction: float = 0.0):
    '''
    Args:
        model: nn.Module, the model to train
        x_pool: torch.Tensor, the pool of samples to train on
        noise_std: float, the noise standard deviation
        p_drop: float, the probability of removing a pixel from the image (0.0 to 1.0)
        corruption: str, the corruption type
        style: str, the style of the interpolant ("linear" or "gvp")
        device: torch.device, the device to train on
        epochs: int, the number of epochs to train for
        batch_size: int, the batch size
        lr: float, the learning rate
        global_step: list, the global step
        z_pool: optional paired Z tensor from update_prior, shape (N, 1, H, W)
        coupled_fraction: fraction of each batch that uses paired Z from z_pool
    '''
    if z_pool is not None and coupled_fraction > 0.0:
        dataset = TensorDataset(x_pool, z_pool)
        has_z = True
    else:
        dataset = TensorDataset(x_pool)
        has_z = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(loader,
                          desc=f"  E-step epoch {epoch}/{epochs}",
                          leave=False):
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

            # Generate fresh Y from current prior samples
            y_batch = forward_channel(x_batch, noise_std=noise_std,
                                      p_drop=p_drop,
                                      corruption=corruption)

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
def update_prior(model: nn.Module, y_obs: torch.Tensor,
                 n_steps: int = 50, batch_size: int = 256, method: str = "euler",
                 device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    N = y_obs.size(0)
    chunks = []
    z_chunks = []
    y_gpu = y_obs.to(device)
    for start in tqdm(range(0, N, batch_size), desc="  M-step", leave=False):
        end = min(start + batch_size, N)
        initial_state = torch.randn(end - start, 1, IMAGE_SIZE, IMAGE_SIZE, device=y_gpu.device)
        x_batch = sample(model, initial_state, y_gpu[start:end], n_steps=n_steps, method=method)
        chunks.append(x_batch.cpu())
        z_chunks.append(initial_state.cpu())

    result = torch.cat(chunks, dim=0)
    z_pool = torch.cat(z_chunks, dim=0)
    print(f"    prior range=[{result.min():.3f}, {result.max():.3f}]"
          f"  mean={result.mean():.4f}  std={result.std():.4f}")

    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    return result, z_pool


########################################################
# Logging
########################################################

def log_em_step_wandb(x_gt: torch.Tensor, y_obs: torch.Tensor,
                      x_pool: torch.Tensor, em_step: int, n: int = 8):
    y_slice = y_obs[:n, 0].cpu()
    vmin = float(y_slice.min())
    vmax = float(y_slice.max())
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8

    # n image columns + 1 histogram column
    fig, axes = plt.subplots(3, n + 1, figsize=(2 * (n + 1), 6))
    rows = [
        (x_gt[:n, 0].cpu(),   "GT  X"),
        (y_obs[:n, 0].cpu(),  "Obs F(X)"),
        (x_pool[:n, 0].cpu(), f"π({em_step})"),
    ]
    for r, (data, label) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=10)
        for j in range(n):
            axes[r, j].imshow(data[j].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            for spine in axes[r, j].spines.values():
                spine.set_visible(False)

        # Histogram over all pixels in this row's n samples
        hist_ax = axes[r, n]
        hist_ax.hist(data.numpy().ravel(), bins=50, color="steelblue", edgecolor="none")
        hist_ax.set_xlim(vmin, vmax)
        hist_ax.tick_params(axis="both", labelsize=7)
        hist_ax.set_yticks([])

    fig.suptitle(f"EM step {em_step}", fontsize=12)
    plt.tight_layout()
    wandb.log({"em/reconstruction": wandb.Image(fig)}, step=em_step)
    plt.close(fig)