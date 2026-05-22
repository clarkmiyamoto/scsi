"""
toy_3d/scsi_simple.py — minimal SCSI for CryoEM 3D.

No ground-truth volumes during training — only fixed 2D observations y_obs = F(x_gt).
EM loop:
  E-step: train velocity field on (x ~ π(k), y = F(x)) pairs
  M-step: sample π(k+1) by running ODE conditioned on y_obs
Bootstrap: π(0) from y_obs via tile / noise / revolve.

x_gt is loaded only for visualization (GT column in eval panels).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, TensorDataset

try:
    from diffusers import UNet3DConditionModel
except ImportError:
    raise ImportError("Missing `diffusers` package.")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

INTEGRATION_SCALE = 999
MAX_ALPHA = 0.85  # voxels never become fully opaque


# ── 1. Shape generation ────────────────────────────────────────────────────────

def _make_grid(grid_size: int, device="cpu"):
    lin = torch.linspace(-1.0, 1.0, grid_size, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    return xx, yy, zz


def _sphere(grid_size, radius, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    return ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 <= radius**2).float()


def _cylinder(grid_size, radius, half_height, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    return (
        ((xx - cx)**2 + (yy - cy)**2 <= radius**2) &
        ((zz - cz).abs() <= half_height)
    ).float()


def _torus(grid_size, R, r, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    dist_from_ring = (torch.sqrt((xx - cx)**2 + (yy - cy)**2) - R)**2 + (zz - cz)**2
    return (dist_from_ring <= r**2).float()


def _helix(grid_size, R, pitch, n_turns, tube_radius, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    n_samples = max(400, grid_size ** 2)
    t = torch.linspace(0.0, 2.0 * np.pi * n_turns, n_samples, device=device)
    hx = R * torch.cos(t) + cx
    hy = (pitch * t / (2.0 * np.pi) - pitch * n_turns / 2.0 + cy)
    hz = R * torch.sin(t) + cz
    dx = xx.unsqueeze(-1) - hx
    dy = yy.unsqueeze(-1) - hy
    dz = zz.unsqueeze(-1) - hz
    dist_sq = (dx**2 + dy**2 + dz**2).min(dim=-1).values
    return (dist_sq <= tube_radius**2).float()


ALL_SHAPES = ["sphere", "cylinder", "torus", "helix"]
DEFAULT_SHAPES = ["torus", "helix", "cylinder", "sphere"]


_SHAPE_PARAMS = {
    "sphere":   dict(radius=0.42),
    "cylinder": dict(radius=0.35, half_height=0.45),
    "torus":    dict(R=0.44, r=0.085),
    "helix":    dict(R=0.40, pitch=0.47, n_turns=2.5, tube_radius=0.11),
}


def _make_volume(shape: str, grid_size: int) -> torch.Tensor:
    """Single canonical instance of a shape, centred at origin."""
    p = _SHAPE_PARAMS[shape]
    if shape == "sphere":
        return _sphere(grid_size, p["radius"], 0, 0, 0)
    elif shape == "cylinder":
        return _cylinder(grid_size, p["radius"], p["half_height"], 0, 0, 0)
    elif shape == "torus":
        return _torus(grid_size, p["R"], p["r"], 0, 0, 0)
    elif shape == "helix":
        return _helix(grid_size, p["R"], p["pitch"], p["n_turns"], p["tube_radius"], 0, 0, 0)
    raise ValueError(f"Unknown shape: {shape!r}")


def generate_toy_dataset(
    n_per_class: int = 2000,
    grid_size: int = 16,
    shapes: list[str] | None = None,
) -> torch.Tensor:
    """Return (N, 1, D, H, W) float32 in [-1, 1] with N = len(shapes) * n_per_class.

    Every sample within a class is an identical canonical volume — no size or
    position jitter. The only variation across samples comes from the random
    SO(3) projection in the forward channel.
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES

    volumes = []
    for shape in shapes:
        vol = _make_volume(shape, grid_size)
        volumes.extend([vol] * n_per_class)

    x = torch.stack(volumes, dim=0).unsqueeze(1).float()
    return x * 2.0 - 1.0


# ── 2. CryoEM forward channel ──────────────────────────────────────────────────

def _apply_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    grid = torch.remainder(grid + 1.0, 2.0) - 1.0
    return F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="border")


def random_so3_rotate(x: torch.Tensor) -> torch.Tensor:
    B = x.size(0)
    R_np = Rotation.random(B).as_matrix().astype(np.float32)
    R = torch.from_numpy(R_np).to(x.device)
    zeros = torch.zeros(B, 3, 1, device=x.device)
    theta = torch.cat([R, zeros], dim=2)
    return _apply_rotation(x, theta)


def forward_channel(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """CryoEM: SO(3) rotation → sum projection → AWGN. (B,1,D,H,W) → (B,1,H,W)."""
    x_rot = random_so3_rotate(x)
    proj = x_rot.sum(dim=2)
    return proj + noise_std * torch.randn_like(proj)


# ── 3. Bootstrap π(0) ─────────────────────────────────────────────────────────

def make_bootstrap(mode: str, y_obs: torch.Tensor, vol_size: int) -> torch.Tensor:
    """Build the initial prior π(0) from 2D observations. Returns (N,1,D,H,W)."""
    N = y_obs.size(0)
    if mode == "tile":
        return y_obs.unsqueeze(2).expand(-1, -1, vol_size, -1, -1).clone()
    elif mode == "noise":
        return torch.randn(N, 1, vol_size, vol_size, vol_size)
    elif mode == "revolve":
        # Revolve each 2D projection around the y-axis.
        # Voxel (z, y, x) samples y_obs at (y, r) where r = sqrt(x^2 + z^2).
        device = y_obs.device
        D = vol_size
        flat = y_obs.reshape(N, -1)
        mn = flat.min(dim=1).values.view(N, 1, 1, 1)
        mx = flat.max(dim=1).values.view(N, 1, 1, 1)
        obs_norm = (y_obs - mn) / (mx - mn + 1e-8)

        lin = torch.linspace(-1.0, 1.0, D, device=device)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        r_vox = torch.sqrt(xx**2 + zz**2)

        grid = torch.stack(
            [r_vox.reshape(D * D, D), yy.reshape(D * D, D)], dim=-1
        ).unsqueeze(0).expand(N, -1, -1, -1)

        sampled = F.grid_sample(
            obs_norm, grid, align_corners=True,
            mode="bilinear", padding_mode="zeros",
        )
        return sampled.reshape(N, 1, D, D, D) * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown bootstrap mode: {mode!r}")


# ── 4. Velocity-field model ────────────────────────────────────────────────────

class ConditionalUNet3D(nn.Module):
    def __init__(
        self,
        vol_size: int = 16,
        block_out_channels: tuple = (32, 64, 128),
        layers_per_block: int = 1,
        norm_num_groups: int = 8,
    ):
        super().__init__()
        self.vol_size = vol_size
        self.unet = UNet3DConditionModel(
            sample_size=vol_size,
            in_channels=2,
            out_channels=1,
            down_block_types=tuple("DownBlock3D" for _ in block_out_channels),
            up_block_types=tuple("UpBlock3D" for _ in block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=block_out_channels[0],
            attention_head_dim=8,
            norm_num_groups=norm_num_groups,
        )

    def forward(self, x_t, t, cond):
        B, _, D, H, W = x_t.shape
        y_3d = cond.unsqueeze(2).expand(-1, -1, D, -1, -1)
        inp = torch.cat([x_t, y_3d], dim=1)
        dummy = torch.zeros(
            B, 1, self.unet.config.cross_attention_dim,
            device=x_t.device, dtype=x_t.dtype,
        )
        return self.unet(inp, timestep=t, encoder_hidden_states=dummy).sample


# ── 5. Stochastic interpolant loss ────────────────────────────────────────────

def si_loss(model, x, y):
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    z = torch.randn_like(x)
    t5 = t[:, None, None, None, None]
    I_t     = (1.0 - t5) * z + t5 * x
    I_dot_t = -z + x
    t_int = (t * INTEGRATION_SCALE).long()
    pred = model(I_t, t_int, y)
    return F.mse_loss(pred, I_dot_t)


# ── 6. ODE sampler (Euler) ────────────────────────────────────────────────────

@torch.no_grad()
def sample_euler(model, y, vol_size, n_steps=50):
    model.eval()
    B = y.size(0)
    x = torch.randn(B, 1, vol_size, vol_size, vol_size, device=y.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t_int = torch.full((B,), t_val * INTEGRATION_SCALE, device=y.device, dtype=torch.long)
        v = model(x, t_int, y)
        x = x + v * dt
    return x


# ── 7. E-step ─────────────────────────────────────────────────────────────────

def train_estep(
    model: nn.Module,
    x_pool: torch.Tensor,
    noise_std: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    global_step: list,
    use_wandb: bool,
) -> None:
    loader = DataLoader(
        TensorDataset(x_pool),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for (x_batch,) in tqdm(loader, desc=f"    E-step epoch {epoch}/{epochs}", leave=False):
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            y_batch = forward_channel(x_batch, noise_std)

            with torch.autocast(device.type, enabled=use_amp):
                B = x_batch.size(0)
                t   = torch.rand(B, device=device)
                z   = torch.randn_like(x_batch)
                t5  = t[:, None, None, None, None]
                I_t     = (1.0 - t5) * z + t5 * x_batch
                I_dot_t = -z + x_batch
                t_int   = (t * INTEGRATION_SCALE).long()
                pred    = model(I_t, t_int, y_batch)
                loss    = F.mse_loss(pred, I_dot_t)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()

            if use_wandb:
                wandb.log({"train/loss": loss.item()}, step=global_step[0])
            global_step[0] += 1

        sched.step()
        print(f"      epoch {epoch:3d}  loss={running / max(len(loader), 1):.5f}")


# ── 8. M-step ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_prior(
    model: nn.Module,
    y_obs: torch.Tensor,
    vol_size: int,
    n_steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample π(k+1) from model conditioned on y_obs. Returns x_pool on CPU."""
    model.eval()
    N = y_obs.size(0)
    chunks = []
    y_gpu = y_obs.to(device)

    for start in tqdm(range(0, N, batch_size), desc="  M-step", leave=False):
        end  = min(start + batch_size, N)
        x    = torch.randn(end - start, 1, vol_size, vol_size, vol_size, device=device)
        y_ch = y_gpu[start:end]
        dt   = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i * dt
            t_int = torch.full((end - start,), t_val * INTEGRATION_SCALE,
                               device=device, dtype=torch.long)
            x = x + model(x, t_int, y_ch) * dt
        chunks.append(x.cpu())

    x_pool = torch.cat(chunks, dim=0)
    print(f"    prior  range=[{x_pool.min():.3f}, {x_pool.max():.3f}]"
          f"  mean={x_pool.mean():.4f}  std={x_pool.std():.4f}")
    return x_pool


# ── 9. Visualization ──────────────────────────────────────────────────────────

def _show_voxels(ax, vol: torch.Tensor) -> None:
    """Render a (D,H,W) volume as translucent voxels; float intensity → alpha, capped at MAX_ALPHA."""
    vol_np = vol.float().cpu().numpy()
    vmin, vmax = vol_np.min(), vol_np.max()
    vol_norm = (vol_np - vmin) / (vmax - vmin + 1e-8)  # [0, 1]

    filled = vol_norm > 0.1  # skip near-background voxels

    rgba = np.zeros((*vol_np.shape, 4))
    rgba[filled, 0] = 0.27  # steelblue
    rgba[filled, 1] = 0.51
    rgba[filled, 2] = 0.71
    rgba[filled, 3] = np.clip(vol_norm[filled] * MAX_ALPHA, 0.0, MAX_ALPHA)

    ax.voxels(filled, facecolors=rgba, edgecolors="none", shade=False)
    ax.view_init(elev=25, azim=45)
    ax.set_axis_off()


def log_em_step(
    x_gt: torch.Tensor,
    y_obs: torch.Tensor,
    x_pool: torch.Tensor,
    noise_std: float,
    em_step: int,
    use_wandb: bool,
    n_cols: int = 6,
    global_step: int | None = None,
) -> None:
    """
    4-row panel with a rightmost histogram column showing intensity distribution:
      Row 0 — GT voxels             [visual reference only]
      Row 1 — CryoEM observation y_obs
      Row 2 — Pool sample x_pool[i] voxels
      Row 3 — F(x_pool[i])          (consistency check)
    """
    n = min(n_cols, x_gt.size(0))
    with torch.no_grad():
        y_pool = forward_channel(x_pool[:n].cpu(), noise_std)

    row_labels = ["GT (3D)", "Obs y_obs", f"π({em_step}) (3D)", "F(pool)"]
    n_rows = len(row_labels)
    _3D_ROWS = {0, 2}
    n_cols_total = n + 1  # sample columns + 1 histogram column

    # plt.subplots can't mix projections — build axes manually
    fig = plt.figure(figsize=(2.5 * n_cols_total, 2.5 * n_rows))
    axes = [
        [
            fig.add_subplot(n_rows, n_cols_total, r * n_cols_total + j + 1,
                            projection="3d" if r in _3D_ROWS else None)
            for j in range(n)
        ] + [fig.add_subplot(n_rows, n_cols_total, r * n_cols_total + n + 1)]
        for r in range(n_rows)
    ]

    def _show(ax, img2d):
        data = img2d if isinstance(img2d, np.ndarray) else img2d.float().cpu().numpy()
        vmin, vmax = data.min(), data.max()
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")

    for j in range(n):
        _show_voxels(axes[0][j], x_gt[j, 0])
        _show(axes[1][j], y_obs[j, 0].float().cpu().numpy())
        _show_voxels(axes[2][j], x_pool[j, 0])
        _show(axes[3][j], y_pool[j, 0].float().cpu().numpy())

    # Histograms: aggregate all sample values across the n columns per row
    row_data = [
        np.concatenate([x_gt[j, 0].float().cpu().numpy().ravel()   for j in range(n)]),
        np.concatenate([y_obs[j, 0].float().cpu().numpy().ravel()   for j in range(n)]),
        np.concatenate([x_pool[j, 0].float().cpu().numpy().ravel()  for j in range(n)]),
        np.concatenate([y_pool[j, 0].float().cpu().numpy().ravel()  for j in range(n)]),
    ]
    for r, data in enumerate(row_data):
        ax_h = axes[r][n]
        ax_h.hist(data, bins=50, color="steelblue", alpha=0.8, density=True)
        ax_h.set_xlabel("intensity", fontsize=7)
        ax_h.set_ylabel("density", fontsize=7)
        ax_h.tick_params(labelsize=6)
        ax_h.spines[["top", "right"]].set_visible(False)

    for r, label in enumerate(row_labels):
        ax0 = axes[r][0]
        kwargs = dict(transform=ax0.transAxes, fontsize=9,
                      va="center", ha="right", fontweight="bold")
        if r in _3D_ROWS:
            ax0.text2D(-0.05, 0.5, label, **kwargs)
        else:
            ax0.text(-0.05, 0.5, label, **kwargs)

    fig.suptitle(f"EM step {em_step}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if use_wandb:
        step = global_step if global_step is not None else em_step
        wandb.log({"em/panel": wandb.Image(fig), "em/step": em_step}, step=step)
    else:
        out = Path("toy3d_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_step_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── 10. CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="toy_3d SCSI: unsupervised CryoEM 3D via EM")
    p.add_argument("--n_per_class",       type=int,   default=2000)
    p.add_argument("--shapes",            nargs="+",  default=DEFAULT_SHAPES,
                   choices=ALL_SHAPES, metavar="SHAPE",
                   help=f"Shape classes. Choices: {ALL_SHAPES}. Default: {DEFAULT_SHAPES}")
    p.add_argument("--vol_size",          type=int,   default=16)
    p.add_argument("--n_em_steps",        type=int,   default=50)
    p.add_argument("--epochs_first_pass", type=int,   default=10,
                   help="E-step epochs for EM iteration 0")
    p.add_argument("--epochs_per_em",     type=int,   default=2,
                   help="E-step epochs for EM iterations 1+")
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--noise_std",         type=float, default=0.3)
    p.add_argument("--sample_steps",      type=int,   default=50)
    p.add_argument("--n_eval",            type=int,   default=5,
                   help="Number of eval samples (one per shape class)")
    p.add_argument("--bootstrap",         type=str,   default="tile",
                   choices=["tile", "noise", "revolve"],
                   help="π(0): tile y_obs along depth | Gaussian noise | revolve around y-axis")
    p.add_argument("--no_wandb",          action="store_true")
    p.add_argument("--debug",             action="store_true",
                   help="2 EM steps, tiny model, quick smoke test")
    return p.parse_args()


# ── 11. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        args.n_em_steps        = 2
        args.n_per_class       = 4
        args.batch_size        = 4
        args.epochs_first_pass = 1
        args.epochs_per_em     = 1
        args.sample_steps      = 5
        args.n_eval            = 4

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def _empty_cache():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Device: {device}  amp={use_amp}")

    # Dataset
    print("Generating toy dataset ...")
    print(f"  shapes={args.shapes}  n_per_class={args.n_per_class}")
    x_gt = generate_toy_dataset(n_per_class=args.n_per_class, grid_size=args.vol_size,
                                shapes=args.shapes)
    N = x_gt.size(0)
    print(f"  {N} volumes  shape={tuple(x_gt.shape)}")

    # Fixed observations
    print("Generating fixed observations y_obs = F(x_gt) ...")
    with torch.no_grad():
        y_obs = forward_channel(x_gt, noise_std=args.noise_std)
    print(f"  y_obs shape={tuple(y_obs.shape)}"
          f"  range=[{y_obs.min():.2f}, {y_obs.max():.2f}]")

    # Eval slice: one sample per shape class
    n_classes     = len(args.shapes)
    class_indices = [c * args.n_per_class for c in range(n_classes)][:args.n_eval]
    x_eval = x_gt[class_indices]
    y_eval = y_obs[class_indices]

    # Bootstrap π(0)
    print(f"Bootstrap π(0): {args.bootstrap} ...")
    x_pool = make_bootstrap(args.bootstrap, y_obs, args.vol_size)
    print(f"  x_pool shape={tuple(x_pool.shape)}")

    # Model
    small_channels = (32, 64, 128) if not args.debug else (16, 32)
    model = ConditionalUNet3D(vol_size=args.vol_size,
                              block_out_channels=small_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # W&B
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="scsi-cryoem-toy3d-simple",
            config=vars(args) | {"n_params": n_params, "N": N},
        )

    ckpt_dir = Path("toy3d_scsi_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    global_step = [0]

    # EM loop
    for k in range(args.n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k} / {args.n_em_steps}")
        print("=" * 60)

        epochs = args.epochs_first_pass if k == 0 else args.epochs_per_em

        # E-step
        train_estep(
            model, x_pool,
            noise_std=args.noise_std,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            global_step=global_step,
            use_wandb=use_wandb,
        )
        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:04d}.pt")
        print(f"  checkpoint saved → {ckpt_dir}/model_em{k:04d}.pt")

        # M-step
        print(f"\n  M-step: sampling π({k+1}) ...")
        x_pool = update_prior(
            model, y_obs,
            vol_size=args.vol_size,
            n_steps=args.sample_steps,
            batch_size=args.batch_size * 3,
            device=device,
        )
        _empty_cache()

        log_em_step(
            x_eval, y_eval, x_pool[class_indices],
            noise_std=args.noise_std,
            em_step=k,
            use_wandb=use_wandb,
            n_cols=len(class_indices),
            global_step=global_step[0],
        )

    if use_wandb:
        wandb.finish()
    print("Done.")
