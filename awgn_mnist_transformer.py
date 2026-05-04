"""
SCSI (Self-Consistent Stochastic Interpolants) — MNIST inverse problems.
=========================================================================
Based on: https://arxiv.org/abs/2512.10857

Interpolant:  I_t = (1-t)*Z + t*X,   t in [0, 1]
  Z ~ N(0,I),  X = clean image
Velocity target:  dI_t/dt = X - Z
Inference:  integrate dx/dt = v_theta(x, t, Y) from t=0 to t=1

Supported channels
  awgn: Y = X + sigma*w
  mra:  Y = T(X) + sigma*w   (random 2-D periodic shift)

EM algorithm (SCSI):
  pi^(0) = Y_obs  (bootstrap)
  For k = 0, 1, ...:
    E-step: train v_theta(I_t, t, Y) with X ~ pi^(k), Y = F(X)
    M-step: push Y_obs through sampler -> pi^(k+1)

Requirements:
    pip install torch torchvision diffusers accelerate tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from diffusers import DiTTransformer2DModel
import matplotlib.pyplot as plt
import wandb

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

IMAGE_SIZE = 32


# ══════════════════════════════════════════════════════════════════════
# 1.  Forward channel
# ══════════════════════════════════════════════════════════════════════
def forward_channel(x: torch.Tensor, noise_std: float = 0.3,
                    corruption: str = "awgn") -> torch.Tensor:
    if corruption == "awgn":
        return x + noise_std * torch.randn_like(x)
    if corruption == "mra":
        B, C, H, W = x.shape
        rows = torch.randint(0, H, (B,))
        cols = torch.randint(0, W, (B,))
        translated = torch.stack([
            torch.roll(x[i], shifts=(rows[i].item(), cols[i].item()), dims=(-2, -1))
            for i in range(B)
        ])
        return translated + noise_std * torch.randn_like(x)
    raise ValueError(f"Unknown corruption: {corruption}")


# ══════════════════════════════════════════════════════════════════════
# 2.  Model: DiT with channel-concat conditioning
# ══════════════════════════════════════════════════════════════════════
class ConditionalDiT(nn.Module):
    """
    Input:  cat([I_t, Y], dim=1)  ->  2 channels
    Output: velocity prediction   ->  1 channel
    t is continuous in [0,1], scaled to [0,999] for DiT's ada-norm.
    """
    def __init__(self, image_size=IMAGE_SIZE, patch_size=4,
                 hidden=192, depth=6, heads=6):
        super().__init__()
        self.dit = DiTTransformer2DModel(
            sample_size=image_size,
            patch_size=patch_size,
            in_channels=2,
            out_channels=1,
            num_layers=depth,
            num_attention_heads=heads,
            attention_head_dim=hidden // heads,
            num_embeds_ada_norm=1000,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        # x_t:  (B, 1, H, W)  interpolated sample I_t
        # t:    (B,)           integer in [0, 999]
        # cond: (B, 1, H, W)  observation Y
        inp = torch.cat([x_t, cond], dim=1)
        dummy = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t, class_labels=dummy).sample


# ══════════════════════════════════════════════════════════════════════
# 3.  Flow-matching loss & sampling
# ══════════════════════════════════════════════════════════════════════
def flow_matching_loss(model: nn.Module, x: torch.Tensor,
                       y: torch.Tensor) -> torch.Tensor:
    """Stochastic interpolant loss with I_t = (1-t)*Z + t*X."""
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    z = torch.randn_like(x)
    t4 = t[:, None, None, None]
    x_t = (1.0 - t4) * z + t4 * x     # I_t
    velocity = x - z                    # dI_t/dt = X - Z
    t_dit = (t * 999).long()
    pred = model(x_t, t_dit, y)
    return F.mse_loss(pred, velocity)

@torch.no_grad()
def sample(model: nn.Module, y: torch.Tensor,
           n_steps: int = 50, method: str = "euler") -> torch.Tensor:
    model.eval()
    B = y.size(0)
    if method == "euler":
        return sample_euler(model, y, n_steps)
    elif method == "midpoint":
        return sample_midpoint(model, y, n_steps)
    else:
        raise ValueError(f"Unknown method: {method}")

@torch.no_grad()
def sample_euler(model: nn.Module, y: torch.Tensor,
                 n_steps: int = 50) -> torch.Tensor:
    model.eval()
    B = y.size(0)
    x = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE, device=y.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt 
        t1 = torch.full((B,), t_val * 999, device=y.device).long()
        v1 = model(x, t1, y)
        x = x + v1 * dt
    return x

@torch.no_grad()
def sample_midpoint(model: nn.Module, y: torch.Tensor,
                    n_steps: int = 50) -> torch.Tensor:
    """Midpoint-rule ODE integration from t=0 (noise) to t=1 (data)."""
    model.eval()
    B = y.size(0)
    x = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE, device=y.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t1 = torch.full((B,), t_val * 999, device=y.device).long()
        v1 = model(x, t1, y)
        x_mid = x + v1 * (dt / 2.0)
        t2 = torch.full((B,), (t_val + dt / 2.0) * 999, device=y.device).long()
        v2 = model(x_mid, t2, y)
        x = x + v2 * dt
    return x


# ══════════════════════════════════════════════════════════════════════
# 4.  E-step: train conditional velocity field
# ══════════════════════════════════════════════════════════════════════
def train_estep(model: nn.Module, x_pool: torch.Tensor,
                noise_std: float, corruption: str,
                epochs: int = 10, batch_size: int = 256, lr: float = 3e-4,
                global_step: list = None):
    loader = DataLoader(
        TensorDataset(x_pool),
        batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for (x_batch,) in tqdm(loader,
                                desc=f"  E-step epoch {epoch}/{epochs}",
                                leave=False):
            x_batch = x_batch.to(device)
            # Generate fresh Y from current prior samples
            y_batch = forward_channel(x_batch, noise_std=noise_std,
                                      corruption=corruption)

            loss = flow_matching_loss(model, x_batch, y_batch)
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


# ══════════════════════════════════════════════════════════════════════
# 5.  M-step: push fixed observations through the model
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def update_prior(model: nn.Module, y_obs: torch.Tensor,
                 n_steps: int = 50, batch_size: int = 256, method: str = "euler") -> torch.Tensor:
    model.eval()
    N = y_obs.size(0)
    chunks = []
    y_gpu = y_obs.to(device)
    for start in tqdm(range(0, N, batch_size), desc="  M-step", leave=False):
        end = min(start + batch_size, N)
        x_batch = sample(model, y_gpu[start:end], n_steps=n_steps, method=method)
        chunks.append(x_batch.cpu())

    result = torch.cat(chunks, dim=0)
    print(f"    prior range=[{result.min():.3f}, {result.max():.3f}]"
          f"  mean={result.mean():.4f}  std={result.std():.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════
# 6.  Visualisation
# ══════════════════════════════════════════════════════════════════════

def log_em_step_wandb(x_gt: torch.Tensor, y_obs: torch.Tensor,
                      x_pool: torch.Tensor, em_step: int, n: int = 8):
    y_slice = y_obs[:n, 0].cpu()
    vmin = float(y_slice.min())
    vmax = float(y_slice.max())
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    rows = [
        (x_gt[:n, 0].cpu(),   "GT  X"),
        (y_obs[:n, 0].cpu(),  "Obs F(X)"),
        (x_pool[:n, 0].cpu(), f"π({em_step})"),
    ]
    for r, (data, label) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=10)
        for j in range(n):
            axes[r, j].imshow(data[j].numpy(), cmap="gray", vmin=vmin, vmax=vmax)

            # Hide tick marks and tick labels instead of the whole axis
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            
            # Optional: If you also want to remove the black box/border 
            # around the image to perfectly mimic axis("off"):
            for spine in axes[r, j].spines.values():
                spine.set_visible(False)

    fig.suptitle(f"EM step {em_step}", fontsize=12)
    plt.tight_layout()
    wandb.log({"em/reconstruction": wandb.Image(fig)}, step=em_step)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Problem parameter
    corruption   = "awgn"
    noise_std    = 0.3
    n_obs        = 10_000 # Number of observations, instead of full dataset

    # SCSI parameters
    n_em_steps   = 100
    epochs_per_em = 2
    epochs_first_pass = 10
    sample_method = "euler" # "euler" or "midpoint"
    sample_steps = 50
    fresh_model_every_em_step = False

    # Training parameters
    batch_size   = 256
    lr           = 3e-4
    
    print(f"Device: {device}")
    print(f"Channel: {corruption},  noise_std={noise_std}")
    print(f"EM steps: {n_em_steps},  epochs/step: {epochs_per_em}\n")

    wandb.init(
        project="scsi-mnist",
        config=dict(
            corruption=corruption,
            noise_std=noise_std,
            n_obs=n_obs,
            n_em_steps=n_em_steps,
            epochs_per_em=epochs_per_em,
            epochs_first_pass=epochs_first_pass,
            sample_method=sample_method,
            sample_steps=sample_steps,
            batch_size=batch_size,
            lr=lr,
            fresh_model_every_em_step=fresh_model_every_em_step,
        ),
    )
    global_step = [0]

    # ── Load MNIST ────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # -> [-1, 1]
    ])
    dataset = datasets.MNIST("./data", train=True, download=True,
                             transform=transform)
    loader = DataLoader(dataset, batch_size=n_obs, shuffle=True)
    x_gt_all, _ = next(iter(loader))          # (n_obs, 1, 32, 32)

    # ── Generate fixed observations ───────────────────────────────────
    y_obs = forward_channel(x_gt_all, noise_std=noise_std, corruption=corruption)
    print(f"GT  range=[{x_gt_all.min():.2f}, {x_gt_all.max():.2f}]")
    print(f"Obs range=[{y_obs.min():.2f}, {y_obs.max():.2f}]\n")

    # ── Bootstrap: pi^(0) = Y_obs ────────────────────────────────────
    x_pool = y_obs.clone()
    prior_history = [x_pool[:8].clone()]

    ckpt_dir = Path(f"checkpoints_{corruption}_mnist_transformer")
    prior_dir = Path(f"priors_{corruption}_mnist_transformer")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    # ── EM loop ───────────────────────────────────────────────────────
    for k in range(n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k}")
        print("=" * 60)

        # Save prior for diagnostics
        torch.save(x_pool, prior_dir / f"prior_em{k:02d}.pt")

        # Fresh model each EM step (following SCSI paper)
        if k==0 or fresh_model_every_em_step:
            model = ConditionalDiT().to(device)
            print(f"Using fresh model")
        else:
            model = model
        
        if k == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}\n")

        # E-step
        epochs = epochs_first_pass if k == 0 else epochs_per_em
        train_estep(model, x_pool, noise_std=noise_std, corruption=corruption,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    global_step=global_step)
        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:02d}.pt")
        print(f"  ✓ saved checkpoint")

        # M-step
        print(f"\n  M-step: sampling π({k+1}) ...")
        x_pool = update_prior(model, y_obs, n_steps=sample_steps,
                              batch_size=batch_size*3, method=sample_method)
        prior_history.append(x_pool[:8].clone())
        log_em_step_wandb(x_gt_all, y_obs, x_pool, em_step=k)

    wandb.finish()
    print("Done.")
