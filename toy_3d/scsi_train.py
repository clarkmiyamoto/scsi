"""
toy_3d/scsi_train.py — SCSI (Self-Consistent Stochastic Interpolant) for CryoEM 3D.

The key difference from train.py (supervised):
  - No access to ground-truth volumes during training.
  - Only the fixed 2D observations y_obs = F(x_gt) are used.
  - Training proceeds via EM:
      E-step: train velocity field on (x ~ π(k), y = F(x)) pairs
      M-step: sample π(k+1) by running ODE conditioned on y_obs
  - Bootstrap: π(0) = y_obs tiled along depth axis.

x_gt is loaded only for visualization (GT column in eval panels).
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import matplotlib.pyplot as plt
import numpy as np
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

try:
    import imageio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False

INTEGRATION_SCALE = 999


# ── 1. Shape generation ────────────────────────────────────────────────────────

def _make_grid(grid_size: int, device="cpu"):
    lin = torch.linspace(-1.0, 1.0, grid_size, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    return xx, yy, zz


def _sphere(grid_size, radius, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    return ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 <= radius**2).float()


def _cube(grid_size, half_side, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    return (
        ((xx - cx).abs() <= half_side) &
        ((yy - cy).abs() <= half_side) &
        ((zz - cz).abs() <= half_side)
    ).float()


def _cylinder(grid_size, radius, half_height, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    return (
        ((xx - cx)**2 + (yy - cy)**2 <= radius**2) &
        ((zz - cz).abs() <= half_height)
    ).float()


def _ellipsoid(grid_size, ra, rb, rc, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    return (
        ((xx - cx) / ra)**2 + ((yy - cy) / rb)**2 + ((zz - cz) / rc)**2 <= 1.0
    ).float()


def _torus(grid_size, R, r, cx, cy, cz, device="cpu"):
    xx, yy, zz = _make_grid(grid_size, device)
    dist_from_ring = (torch.sqrt((xx - cx)**2 + (yy - cy)**2) - R)**2 + (zz - cz)**2
    return (dist_from_ring <= r**2).float()


def _torus_y_axis(grid_size, R, r, cx, cy, cz, device="cpu"):
    """Torus whose symmetry axis is Y (ring lies in XZ plane)."""
    xx, yy, zz = _make_grid(grid_size, device)
    dist_from_ring = (torch.sqrt((xx - cx)**2 + (zz - cz)**2) - R)**2 + (yy - cy)**2
    return (dist_from_ring <= r**2).float()



def _helix(grid_size, R, pitch, n_turns, tube_radius, cx, cy, cz, device="cpu"):
    """
    Helix coiled around the Y-axis.
    R: coil radius; pitch: height per full turn; n_turns: number of turns.
    Samples the helix curve densely, then marks voxels within tube_radius.
    """
    xx, yy, zz = _make_grid(grid_size, device)
    n_samples = max(400, grid_size ** 2)
    t = torch.linspace(0.0, 2.0 * np.pi * n_turns, n_samples, device=device)
    hx = R * torch.cos(t) + cx
    hy = (pitch * t / (2.0 * np.pi) - pitch * n_turns / 2.0 + cy)
    hz = R * torch.sin(t) + cz
    # (D,D,D,n_samples) distance field; min over curve samples
    dx = xx.unsqueeze(-1) - hx
    dy = yy.unsqueeze(-1) - hy
    dz = zz.unsqueeze(-1) - hz
    dist_sq = (dx**2 + dy**2 + dz**2).min(dim=-1).values
    return (dist_sq <= tube_radius**2).float()


def _linked_rings(grid_size, R, r, cx, cy, cz, device="cpu"):
    """
    Two interlocked tori sharing the same center.
    Ring 1: axis = Z (ring in XY plane).
    Ring 2: axis = Y (ring in XZ plane).
    Their main circles pass through each other's holes, topologically linking them.
    """
    t1 = _torus(grid_size, R, r, cx, cy, cz, device)
    t2 = _torus_y_axis(grid_size, R, r, cx, cy, cz, device)
    return (t1 + t2).clamp(0.0, 1.0)


def _separated_rings(grid_size, R, r, offset, cx, cy, cz, device="cpu"):
    """Two separate tori side-by-side along X, both with axis = Z."""
    t1 = _torus(grid_size, R, r, cx - offset, cy, cz, device)
    t2 = _torus(grid_size, R, r, cx + offset, cy, cz, device)
    return (t1 + t2).clamp(0.0, 1.0)


def _mickey_mouse(grid_size, R_head, R_ear, ear_dx, ear_dy, cx, cy, cz, device="cpu"):
    """
    Mickey Mouse head: large sphere (head) + two small spheres (ears) at top.
    cx, cy, cz: center of the head sphere.
    ear_dx: horizontal (X) offset of each ear from head center.
    ear_dy: vertical (Y) offset of each ear above head center.
    """
    head  = _sphere(grid_size, R_head, cx, cy, cz, device)
    ear_l = _sphere(grid_size, R_ear, cx - ear_dx, cy + ear_dy, cz, device)
    ear_r = _sphere(grid_size, R_ear, cx + ear_dx, cy + ear_dy, cz, device)
    return (head + ear_l + ear_r).clamp(0.0, 1.0)


ALL_SHAPES = [
    "sphere", "cube", "cylinder", "ellipsoid", "torus",
    "helix", "linked_rings", "separated_rings", "mickey_mouse",
]
DEFAULT_SHAPES = ["torus", "helix", "linked_rings", "separated_rings", "mickey_mouse"]


def generate_toy_dataset(
    n_per_class: int = 2000,
    grid_size: int = 16,
    seed: int = 42,
    shapes: list[str] | None = None,
) -> torch.Tensor:
    """Return (N, 1, D, H, W) float32 in [-1, 1] with N = len(shapes) * n_per_class."""
    if shapes is None:
        shapes = DEFAULT_SHAPES
    rng = np.random.default_rng(seed)
    jitter = 0.05

    def jit():
        return float(rng.uniform(-jitter, jitter))

    volumes = []

    for shape in shapes:
        for _ in range(n_per_class):
            if shape == "sphere":
                r = float(rng.uniform(0.30, 0.55))
                volumes.append(_sphere(grid_size, r, jit(), jit(), jit()))
            elif shape == "cube":
                a = float(rng.uniform(0.25, 0.50))
                volumes.append(_cube(grid_size, a, jit(), jit(), jit()))
            elif shape == "cylinder":
                r = float(rng.uniform(0.25, 0.45))
                h = float(rng.uniform(0.30, 0.60))
                volumes.append(_cylinder(grid_size, r, h, jit(), jit(), jit()))
            elif shape == "ellipsoid":
                ra, rb, rc = (float(rng.uniform(0.20, 0.55)) for _ in range(3))
                volumes.append(_ellipsoid(grid_size, ra, rb, rc, jit(), jit(), jit()))
            elif shape == "torus":
                R = float(rng.uniform(0.38, 0.50))
                r = float(rng.uniform(0.06, 0.11))
                volumes.append(_torus(grid_size, R, r, jit(), jit(), jit()))
            elif shape == "helix":
                R       = float(rng.uniform(0.35, 0.45))
                tube_r  = float(rng.uniform(0.09, 0.13))
                n_turns = float(rng.uniform(2.0, 3.0))
                pitch   = float(rng.uniform(0.40, 0.55))
                volumes.append(_helix(grid_size, R, pitch, n_turns, tube_r,
                                      jit(), jit(), jit()))
            elif shape == "linked_rings":
                R = float(rng.uniform(0.30, 0.40))
                r = float(rng.uniform(0.09, 0.13))
                volumes.append(_linked_rings(grid_size, R, r, jit(), jit(), jit()))
            elif shape == "separated_rings":
                R      = float(rng.uniform(0.20, 0.27))
                r      = float(rng.uniform(0.08, 0.11))
                offset = float(rng.uniform(0.32, 0.42))
                volumes.append(_separated_rings(grid_size, R, r, offset,
                                                jit(), jit(), jit()))
            elif shape == "mickey_mouse":
                R_head = float(rng.uniform(0.26, 0.32))
                R_ear  = float(rng.uniform(0.13, 0.17))
                ear_dx = float(rng.uniform(0.24, 0.30))
                ear_dy = float(rng.uniform(0.28, 0.34))
                # shift head center down so ears don't clip the top of the grid
                volumes.append(_mickey_mouse(grid_size, R_head, R_ear, ear_dx, ear_dy,
                                             jit(), jit() - 0.10, jit()))
            else:
                raise ValueError(f"Unknown shape: {shape!r}")

    x = torch.stack(volumes, dim=0).unsqueeze(1).float()
    x = x * 2.0 - 1.0
    return x


# ── 2. CryoEM corruption channel ──────────────────────────────────────────────

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


def bootstrap_revolve(y_obs: torch.Tensor, vol_size: int) -> torch.Tensor:
    """
    Revolve 2D projections around the y-axis to form 3D volumes.
    A 2D circle → 3D sphere; a filled square → cylinder.

    For each 3D voxel at (z, y, x), samples y_obs at (y, r) where
    r = sqrt(x^2 + z^2). r > 1 falls outside the image → empty (-1).

    y_obs  : (N, 1, H, W)
    returns: (N, 1, D, H, W)  values in [-1, 1]
    """
    N, _, H, W = y_obs.shape
    D = vol_size
    device = y_obs.device

    # Normalize per sample: background → 0, peak projection signal → 1
    flat = y_obs.reshape(N, -1)
    mn = flat.min(dim=1).values.view(N, 1, 1, 1)
    mx = flat.max(dim=1).values.view(N, 1, 1, 1)
    obs_norm = (y_obs - mn) / (mx - mn + 1e-8)  # (N, 1, H, W) in [0, 1]

    lin = torch.linspace(-1.0, 1.0, D, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")  # (D, D, D)

    # Radius from y-axis for each voxel
    r_vox = torch.sqrt(xx**2 + zz**2)  # (D, D, D), range [0, sqrt(2)]

    # Build sampling grid: flatten (z, y) dims → D*D rows, keep x as D cols
    # grid[..., 0] = column lookup (radius), grid[..., 1] = row lookup (height)
    grid = torch.stack(
        [r_vox.reshape(D * D, D), yy.reshape(D * D, D)],
        dim=-1,
    ).unsqueeze(0).expand(N, -1, -1, -1)  # (N, D*D, D, 2)

    # grid_sample: (N,1,H,W) × (N,D*D,D,2) → (N,1,D*D,D)
    sampled = F.grid_sample(
        obs_norm, grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",  # r > 1 → outside image → 0 → maps to -1
    )

    return sampled.reshape(N, 1, D, D, D) * 2.0 - 1.0


# ── 3. Conditional velocity-field model ───────────────────────────────────────

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


# ── 4. Stochastic interpolant loss ────────────────────────────────────────────

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


# ── 5. ODE sampler (Euler) ────────────────────────────────────────────────────

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


# ── 6. E-step ─────────────────────────────────────────────────────────────────

def train_estep_3d(
    model: nn.Module,
    x_pool: torch.Tensor,      # (N, 1, D, H, W) current prior — CPU
    noise_std: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    global_step: list,
    use_wandb: bool,
    z_pool: torch.Tensor | None = None,   # (N, 1, D, H, W) paired noise — CPU
    coupled_fraction: float = 0.0,
) -> None:
    if z_pool is not None and coupled_fraction > 0.0:
        dataset = TensorDataset(x_pool, z_pool)
        has_z = True
    else:
        dataset = TensorDataset(x_pool)
        has_z = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True,
                        pin_memory=(device.type == "cuda"))
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"    E-step epoch {epoch}/{epochs}", leave=False):
            if has_z:
                x_batch, z_batch_coupled = batch
                x_batch        = x_batch.to(device, non_blocking=(device.type == "cuda"))
                z_batch_coupled = z_batch_coupled.to(device, non_blocking=(device.type == "cuda"))

                B = x_batch.size(0)
                n_coupled = int(round(coupled_fraction * B))
                n_random  = B - n_coupled
                if n_coupled == B:
                    z_batch = z_batch_coupled
                elif n_coupled == 0:
                    z_batch = torch.randn_like(x_batch)
                else:
                    z_random = torch.randn(n_random, *x_batch.shape[1:], device=device)
                    z_batch  = torch.cat([z_batch_coupled[:n_coupled], z_random], dim=0)
            else:
                (x_batch,) = batch
                x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
                z_batch = None

            y_batch = forward_channel(x_batch, noise_std)

            with torch.autocast(device.type, enabled=use_amp):
                B_b = x_batch.size(0)
                t   = torch.rand(B_b, device=device)
                z   = torch.randn_like(x_batch) if z_batch is None else z_batch
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
        n_batches = max(len(loader), 1)
        print(f"      epoch {epoch:3d}  loss={running / n_batches:.5f}")


# ── 7. M-step ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_prior_3d(
    model: nn.Module,
    y_obs: torch.Tensor,   # (N, 1, H, W) fixed observations — CPU
    vol_size: int,
    n_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample π(k+1) from model conditioned on y_obs. Returns (x_pool, z_pool) on CPU."""
    model.eval()
    N = y_obs.size(0)
    x_chunks, z_chunks = [], []
    y_gpu = y_obs.to(device)

    for start in tqdm(range(0, N, batch_size), desc="  M-step", leave=False):
        end  = min(start + batch_size, N)
        z    = torch.randn(end - start, 1, vol_size, vol_size, vol_size, device=device)
        y_ch = y_gpu[start:end]
        dt   = 1.0 / n_steps
        x    = z.clone()
        for i in range(n_steps):
            t_val = i * dt
            t_int = torch.full((end - start,), t_val * INTEGRATION_SCALE,
                               device=device, dtype=torch.long)
            v = model(x, t_int, y_ch)
            x = x + v * dt
        x_chunks.append(x.cpu())
        z_chunks.append(z.cpu())

    x_pool = torch.cat(x_chunks, dim=0)
    z_pool = torch.cat(z_chunks, dim=0)
    print(f"    prior  range=[{x_pool.min():.3f}, {x_pool.max():.3f}]"
          f"  mean={x_pool.mean():.4f}  std={x_pool.std():.4f}")
    return x_pool, z_pool


# ── 8. Visualization helpers ──────────────────────────────────────────────────

_VIEW_R = Rotation.from_euler("yx", [35, 20], degrees=True)
_VIEW_THETA = torch.tensor(
    np.concatenate([_VIEW_R.as_matrix().astype(np.float32),
                    np.zeros((3, 1), dtype=np.float32)], axis=1)[None],
)


def _make_z_rot_theta(phi: float) -> torch.Tensor:
    c, s = float(np.cos(phi)), float(np.sin(phi))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    R  = _VIEW_R.as_matrix().astype(np.float32) @ Rz
    return torch.tensor(
        np.concatenate([R, np.zeros((3, 1), dtype=np.float32)], axis=1)[None]
    )


def _project_from_pov(vol: torch.Tensor, theta: torch.Tensor) -> np.ndarray:
    vol5    = vol.float().unsqueeze(0).unsqueeze(0).cpu()
    vol_rot = _apply_rotation(vol5, theta)
    return vol_rot[0, 0].sum(dim=0).numpy()


def render_volume(vol: torch.Tensor, theta: torch.Tensor | None = None) -> np.ndarray:
    if theta is None:
        theta = _VIEW_THETA
    vol5    = vol.float().unsqueeze(0).unsqueeze(0).cpu()
    vol_rot = _apply_rotation(vol5, theta)
    occ     = (vol_rot[0, 0] > 0.0).numpy()
    D, H, W = occ.shape

    any_occ = occ.any(axis=0)
    first_z = np.argmax(occ, axis=0).astype(float)

    rows       = np.arange(H)[:, None] * np.ones((1, W))
    height_norm = 1.0 - rows / max(H - 1, 1)

    gy, gx = np.gradient(first_z)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx, ny, nz = nx / mag, ny / mag, nz / mag
    light = np.array([0.5, 0.7, 1.0])
    light /= np.linalg.norm(light)
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    shading = 0.25 + 0.75 * diffuse

    cmap = plt.get_cmap("plasma")
    rgb  = cmap(height_norm)[:, :, :3] * shading[:, :, None]
    rgb  = np.clip(rgb, 0.0, 1.0)
    bg   = np.ones((H, W, 3))
    return np.where(any_occ[:, :, None], rgb, bg)


def _show_2d(ax, img2d: torch.Tensor, label: str | None = None):
    data = img2d.float().cpu().numpy()
    vmin, vmax = data.min(), data.max()
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    if label:
        ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=60, va="center")


def log_em_step(
    x_gt: torch.Tensor,      # (N, 1, D, H, W) ground-truth (for visual reference)
    y_obs: torch.Tensor,     # (N, 1, H, W)    fixed 2D observations
    x_pool: torch.Tensor,    # (N, 1, D, H, W) current prior samples
    noise_std: float,
    em_step: int,
    use_wandb: bool,
    n_cols: int = 6,
) -> None:
    """
    4-row panel:
      Row 0 — GT volume (3-D render)     [visual reference only]
      Row 1 — CryoEM observation y_obs   (2-D)
      Row 2 — Pool sample x_pool[i]      (3-D render)
      Row 3 — F(x_pool[i])               (2-D; consistency check)
    """
    n = min(n_cols, x_gt.size(0))
    cpu = torch.device("cpu")

    with torch.no_grad():
        y_pool = forward_channel(x_pool[:n].to(cpu), noise_std)

    row_labels = ["GT (3-D)", "Obs y_obs", f"π({em_step}) sample", "F(pool)"]
    n_rows = len(row_labels)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.2 * n, 2.2 * n_rows), squeeze=False)

    for j in range(n):
        axes[0, j].imshow(render_volume(x_gt[j, 0].cpu()),   interpolation="nearest")
        axes[0, j].axis("off")
        _show_2d(axes[1, j], y_obs[j, 0])
        axes[2, j].imshow(render_volume(x_pool[j, 0].cpu()), interpolation="nearest")
        axes[2, j].axis("off")
        _show_2d(axes[3, j], y_pool[j, 0])

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=65, va="center")

    fig.suptitle(f"EM step {em_step}", fontsize=11, y=1.01)
    plt.tight_layout()

    if use_wandb:
        wandb.log({"em/panel": wandb.Image(fig)}, step=em_step)
    else:
        out = Path("toy3d_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"em_step_{em_step:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── 9. Rotation movie (reused from train.py) ──────────────────────────────────

def make_rotation_movie(
    x_gt: torch.Tensor,
    y_obs: torch.Tensor,
    x_pool: torch.Tensor,
    em_step: int,
    use_wandb: bool,
    n_frames: int = 72,
    fps: int = 15,
) -> None:
    N = x_gt.size(0)

    gt_projs_np = [y_obs[i, 0].float().cpu().numpy() for i in range(N)]
    gt_ranges = []
    for p in gt_projs_np:
        lo, hi = float(p.min()), float(p.max())
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        gt_ranges.append((lo, hi))

    def _to_uint8_gray(arr, vmin=None, vmax=None):
        if vmin is None:
            vmin = arr.min()
        if vmax is None:
            vmax = arr.max()
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        gray = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb  = np.stack([gray, gray, gray], axis=-1)
        return (rgb * 255).astype(np.uint8)

    col_labels = ["GT", "Pool sample"]
    n_rows     = 2 * N

    out_dir = Path("toy3d_eval")
    out_dir.mkdir(exist_ok=True)

    if _IMAGEIO_AVAILABLE:
        out_path = out_dir / f"em_step_{em_step:04d}_rotation.gif"
        writer   = imageio.get_writer(str(out_path), fps=fps)
    else:
        frame_dir = out_dir / f"em_step_{em_step:04d}_rotation_frames"
        frame_dir.mkdir(exist_ok=True)
        writer = None

    for i in tqdm(range(n_frames), desc="  rotation movie", leave=False):
        phi   = 2.0 * np.pi * i / n_frames
        theta = _make_z_rot_theta(phi)

        fig, axes = plt.subplots(n_rows, 2, figsize=(4.4, 2.2 * n_rows), squeeze=False)

        for s in range(N):
            r_render = 2 * s
            r_proj   = 2 * s + 1

            renders = [
                render_volume(x_gt[s, 0].cpu(),    theta),
                render_volume(x_pool[s, 0].cpu(),  theta),
            ]
            lo, hi = gt_ranges[s]
            projs = [
                _to_uint8_gray(gt_projs_np[s], lo, hi),
                _to_uint8_gray(_project_from_pov(x_pool[s, 0].cpu(), theta)),
            ]

            for j in range(2):
                if s == 0:
                    axes[r_render, j].set_title(col_labels[j], fontsize=8)
                axes[r_render, j].imshow(renders[j], interpolation="nearest")
                axes[r_render, j].axis("off")
                axes[r_proj, j].imshow(projs[j], cmap="gray", vmin=0, vmax=255,
                                       interpolation="nearest")
                axes[r_proj, j].axis("off")

            axes[r_render, 0].set_ylabel(f"Shape {s+1}\n3-D", fontsize=7,
                                         rotation=0, labelpad=50, va="center")
            axes[r_proj,   0].set_ylabel("Proj", fontsize=7,
                                         rotation=0, labelpad=50, va="center")

        deg = int(round(np.degrees(phi)))
        fig.suptitle(f"EM step {em_step}  |  z-rotation {deg:3d}°", fontsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72)
        buf.seek(0)
        frame_u8 = (plt.imread(buf)[:, :, :3] * 255).astype(np.uint8)
        buf.close()
        plt.close(fig)

        if writer is not None:
            writer.append_data(frame_u8)
        else:
            plt.imsave(str(frame_dir / f"frame_{i:03d}.png"), frame_u8)

    if writer is not None:
        writer.close()
        if use_wandb:
            wandb.log({"em/rotation_movie": wandb.Video(str(out_path), fps=fps)},
                      step=em_step)


# ── 10. CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="toy_3d SCSI: unsupervised CryoEM 3D via EM")
    p.add_argument("--n_per_class",       type=int,   default=2000)
    p.add_argument("--shapes",            nargs="+",  default=DEFAULT_SHAPES,
                   choices=ALL_SHAPES,
                   metavar="SHAPE",
                   help=("Ordered list of shape classes to include. "
                         f"Choices: {ALL_SHAPES}. "
                         f"Default: {DEFAULT_SHAPES}"))
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
    p.add_argument("--movie_every",       type=int,   default=10,
                   help="Save rotation movie every N EM steps")
    p.add_argument("--n_frames",          type=int,   default=72)
    p.add_argument("--coupled_fraction",  type=float, default=0.0,
                   help="Fraction of E-step batch using paired z from M-step")
    p.add_argument("--bootstrap",         type=str,   default="tile",
                   choices=["tile", "noise", "revolve"],
                   help="π(0): tile y_obs along depth axis | Gaussian noise | revolve y_obs around y-axis (circle→sphere)")
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
        args.n_frames          = 8
        args.movie_every       = 1
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

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("Generating toy dataset ...")
    print(f"  shapes={args.shapes}  n_per_class={args.n_per_class}")
    x_gt = generate_toy_dataset(
        n_per_class=args.n_per_class,
        grid_size=args.vol_size,
        shapes=args.shapes,
    )
    N    = x_gt.size(0)
    print(f"  {N} volumes  shape={tuple(x_gt.shape)}")

    # Fixed observations (never updated during SCSI)
    print("Generating fixed observations y_obs = F(x_gt) ...")
    with torch.no_grad():
        y_obs = forward_channel(x_gt, noise_std=args.noise_std)   # (N, 1, H, W)
    print(f"  y_obs shape={tuple(y_obs.shape)}"
          f"  range=[{y_obs.min():.2f}, {y_obs.max():.2f}]")

    # Evaluation slice: one sample per shape class
    n_classes     = len(args.shapes)
    class_indices = [c * args.n_per_class for c in range(n_classes)]
    class_indices = class_indices[:args.n_eval]
    x_eval = x_gt[class_indices]
    y_eval = y_obs[class_indices]

    # ── Bootstrap π(0) ─────────────────────────────────────────────────────────
    if args.bootstrap == "tile":
        # Tile each 2D observation along the depth axis to form a 3D volume
        x_pool = y_obs.unsqueeze(2).expand(-1, -1, args.vol_size, -1, -1).clone()
    elif args.bootstrap == "revolve":
        x_pool = bootstrap_revolve(y_obs, args.vol_size)
    else:
        x_pool = torch.randn(N, 1, args.vol_size, args.vol_size, args.vol_size)
    z_pool = None
    print(f"  Bootstrap π(0): {args.bootstrap}  shape={tuple(x_pool.shape)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    small_channels = (32, 64, 128) if not args.debug else (16, 32)
    model = ConditionalUNet3D(
        vol_size=args.vol_size,
        block_out_channels=small_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="scsi-cryoem-toy3d-scsi",
            config=vars(args) | {"n_params": n_params, "N": N},
        )

    ckpt_dir = Path("toy3d_scsi_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    global_step = [0]

    # ── EM loop ───────────────────────────────────────────────────────────────
    for k in range(args.n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k} / {args.n_em_steps}")
        print("=" * 60)

        epochs = args.epochs_first_pass if k == 0 else args.epochs_per_em

        # E-step
        train_estep_3d(
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
            z_pool=z_pool,
            coupled_fraction=args.coupled_fraction,
        )
        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:04d}.pt")
        print(f"  checkpoint saved → {ckpt_dir}/model_em{k:04d}.pt")

        # M-step
        print(f"\n  M-step: sampling π({k+1}) ...")
        x_pool, z_pool = update_prior_3d(
            model, y_obs,
            vol_size=args.vol_size,
            n_steps=args.sample_steps,
            batch_size=args.batch_size * 3,
            device=device,
        )
        _empty_cache()

        if use_wandb:
            wandb.log({"em/step": k}, step=k)

        # Visualize
        log_em_step(
            x_eval, y_eval, x_pool[class_indices],
            noise_std=args.noise_std,
            em_step=k,
            use_wandb=use_wandb,
            n_cols=len(class_indices),
        )

        if k % args.movie_every == 0:
            make_rotation_movie(
                x_eval, y_eval, x_pool[class_indices],
                em_step=k,
                use_wandb=use_wandb,
                n_frames=args.n_frames,
            )

    if use_wandb:
        wandb.finish()
    print("Done.")
