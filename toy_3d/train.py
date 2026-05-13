"""
toy_3d/train.py — Supervised diffusion model for CryoEM 3D reconstruction.

Setup:
  - Dataset: finite pool of simple synthetic 3D shapes (sphere, cube, cylinder,
    ellipsoid, torus) generated analytically on a VOL_SIZE³ voxel grid.
  - Corruption channel: random SO(3) rotation → sum projection (Radon) → AWGN.
  - Model: stochastic interpolant (flow matching) with a conditional UNet3D
    velocity field, conditioned on the 2D CryoEM observation.
  - Training: supervised — (x, y=F(x)) pairs, fresh random rotation each batch.
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display server on HPC; must set before pyplot import
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

INTEGRATION_SCALE = 999   # integer time range for UNet timestep embedding


# ── 1. Shape generation ────────────────────────────────────────────────────────

def _make_grid(grid_size: int, device="cpu"):
    """Return (D,H,W) coordinate tensors in [-1, 1]."""
    lin = torch.linspace(-1.0, 1.0, grid_size, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    return xx, yy, zz   # each (D, H, W)


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


def generate_toy_dataset(
    n_per_class: int = 50,
    grid_size: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    """
    Return (N, 1, D, H, W) float32 in [-1, 1] with N = 5 * n_per_class.

    Classes (in order): sphere, cube, cylinder, ellipsoid, torus.
    Each instance has randomized shape parameters drawn from a seeded RNG.
    """
    rng = np.random.default_rng(seed)
    jitter = 0.05   # max center offset

    def jit():
        return float(rng.uniform(-jitter, jitter))

    volumes = []

    # Sphere: radius in [0.30, 0.55]
    for _ in range(n_per_class):
        r = float(rng.uniform(0.30, 0.55))
        v = _sphere(grid_size, r, jit(), jit(), jit())
        volumes.append(v)

    # Cube: half-side in [0.25, 0.50]
    for _ in range(n_per_class):
        a = float(rng.uniform(0.25, 0.50))
        v = _cube(grid_size, a, jit(), jit(), jit())
        volumes.append(v)

    # Cylinder: radius in [0.25, 0.45], half-height in [0.30, 0.60]
    for _ in range(n_per_class):
        r = float(rng.uniform(0.25, 0.45))
        h = float(rng.uniform(0.30, 0.60))
        v = _cylinder(grid_size, r, h, jit(), jit(), jit())
        volumes.append(v)

    # Ellipsoid: each semi-axis in [0.20, 0.55]
    for _ in range(n_per_class):
        ra, rb, rc = (float(rng.uniform(0.20, 0.55)) for _ in range(3))
        v = _ellipsoid(grid_size, ra, rb, rc, jit(), jit(), jit())
        volumes.append(v)

    # Torus: major R in [0.35, 0.50], minor r in [0.10, 0.20]
    for _ in range(n_per_class):
        R = float(rng.uniform(0.35, 0.50))
        r = float(rng.uniform(0.10, 0.20))
        v = _torus(grid_size, R, r, jit(), jit(), jit())
        volumes.append(v)

    x = torch.stack(volumes, dim=0).unsqueeze(1).float()   # (N,1,D,H,W)
    x = x * 2.0 - 1.0                                      # {0,1} -> {-1,1}
    return x


# ── 2. CryoEM corruption channel ──────────────────────────────────────────────

def _apply_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate volumes using affine matrix theta with periodic boundary conditions.

    Grid coordinates outside [-1, 1] are wrapped via modular arithmetic so that
    the volume tiles periodically rather than being zero-padded. This avoids
    edge artifacts when voxels near the grid boundary are queried after rotation.
    """
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    grid = torch.remainder(grid + 1.0, 2.0) - 1.0   # wrap to [-1, 1] periodically
    return F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="border")


def random_so3_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply independent Haar-random SO(3) rotations to each volume in the batch."""
    B = x.size(0)
    R_np = Rotation.random(B).as_matrix().astype(np.float32)
    R = torch.from_numpy(R_np).to(x.device)                 # (B, 3, 3)
    zeros = torch.zeros(B, 3, 1, device=x.device)
    theta = torch.cat([R, zeros], dim=2)                     # (B, 3, 4)
    return _apply_rotation(x, theta)


def forward_channel(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """CryoEM: SO(3) rotation → sum projection → AWGN. (B,1,D,H,W) → (B,1,H,W)."""
    x_rot = random_so3_rotate(x)
    proj = x_rot.sum(dim=2)                                  # (B,1,H,W)
    return proj + noise_std * torch.randn_like(proj)


# ── 3. Conditional velocity-field model ───────────────────────────────────────

class ConditionalUNet3D(nn.Module):
    """
    Velocity field v(I_t, t, y) for stochastic interpolant ODE.

    Concatenates the interpolated volume I_t with the 2D observation y
    (tiled along the depth axis) and feeds them to a UNet3DConditionModel.
    """

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

    def forward(
        self,
        x_t: torch.Tensor,    # (B, 1, D, H, W)
        t: torch.Tensor,       # (B,) long in [0, INTEGRATION_SCALE]
        cond: torch.Tensor,    # (B, 1, H, W)
    ) -> torch.Tensor:
        B, _, D, H, W = x_t.shape
        y_3d = cond.unsqueeze(2).expand(-1, -1, D, -1, -1)   # (B,1,D,H,W)
        inp = torch.cat([x_t, y_3d], dim=1)                   # (B,2,D,H,W)
        dummy = torch.zeros(
            B, 1, self.unet.config.cross_attention_dim,
            device=x_t.device, dtype=x_t.dtype,
        )
        return self.unet(inp, timestep=t, encoder_hidden_states=dummy).sample


# ── 4. Stochastic interpolant (linear) ────────────────────────────────────────

def si_loss(
    model: nn.Module,
    x: torch.Tensor,    # (B, 1, D, H, W)  ground-truth 3D volume
    y: torch.Tensor,    # (B, 1, H, W)     2D observation
) -> torch.Tensor:
    B = x.size(0)
    t = torch.rand(B, device=x.device)
    z = torch.randn_like(x)
    t5 = t[:, None, None, None, None]

    I_t     = (1.0 - t5) * z + t5 * x     # linear interpolant
    I_dot_t = -z + x                        # dI/dt (constant in linear case)

    t_int = (t * INTEGRATION_SCALE).long()
    pred = model(I_t, t_int, y)
    return F.mse_loss(pred, I_dot_t)


# ── 5. ODE sampler (Euler) ────────────────────────────────────────────────────

@torch.no_grad()
def sample_euler(
    model: nn.Module,
    y: torch.Tensor,       # (B, 1, H, W)
    vol_size: int,
    n_steps: int = 50,
) -> torch.Tensor:
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


# ── 6. Visualization helpers ──────────────────────────────────────────────────

# Fixed viewing rotation applied before ray-casting (azimuth 35°, elevation 20°).
_VIEW_R = Rotation.from_euler("yx", [35, 20], degrees=True)
_VIEW_THETA = torch.tensor(
    np.concatenate([_VIEW_R.as_matrix().astype(np.float32),
                    np.zeros((3, 1), dtype=np.float32)], axis=1)[None],  # (1,3,4)
)


def _make_z_rot_theta(phi: float) -> torch.Tensor:
    """(1,3,4) rotation matrix: z-axis rotation by phi composed with the view rotation."""
    c, s = float(np.cos(phi)), float(np.sin(phi))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    R = _VIEW_R.as_matrix().astype(np.float32) @ Rz
    return torch.tensor(
        np.concatenate([R, np.zeros((3, 1), dtype=np.float32)], axis=1)[None]
    )


def _project_from_pov(vol: torch.Tensor, theta: torch.Tensor) -> np.ndarray:
    """Sum-project a (D,H,W) volume along depth after rotating by theta. Returns (H,W)."""
    vol5 = vol.float().unsqueeze(0).unsqueeze(0).cpu()   # (1,1,D,H,W)
    vol_rot = _apply_rotation(vol5, theta)
    return vol_rot[0, 0].sum(dim=0).numpy()              # (H, W)


def render_volume(vol: torch.Tensor, theta: torch.Tensor | None = None) -> np.ndarray:
    """
    Render a (D, H, W) voxel volume as an (H, W, 3) RGB image.

    Steps:
      1. Rotate by a fixed viewing angle so the shape looks 3-D.
      2. Ray-cast along depth: find the first occupied voxel for each pixel.
      3. Color by image-row height (plasma: bottom=purple, top=yellow).
      4. Shade with surface normals derived from the depth buffer (Lambert).
    """
    if theta is None:
        theta = _VIEW_THETA
    # 1. Viewing rotation (periodic BC via _apply_rotation)
    vol5 = vol.float().unsqueeze(0).unsqueeze(0).cpu()   # (1,1,D,H,W)
    vol_rot = _apply_rotation(vol5, theta)
    occ = (vol_rot[0, 0] > 0.0).numpy()   # (D, H, W)
    D, H, W = occ.shape

    # 2. Depth buffer: index of first occupied voxel along z (axis 0)
    any_occ = occ.any(axis=0)                          # (H, W)
    first_z = np.argmax(occ, axis=0).astype(float)     # (H, W); 0 where empty too

    # 3. Height-based color: row 0 = top of image → warm, row H-1 = bottom → cool
    rows = np.arange(H)[:, None] * np.ones((1, W))
    height_norm = 1.0 - rows / max(H - 1, 1)   # (H, W)  top→1, bottom→0

    # 4. Surface normals from depth-buffer gradient + Lambert shading
    gy, gx = np.gradient(first_z)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx, ny, nz = nx / mag, ny / mag, nz / mag
    light = np.array([0.5, 0.7, 1.0])
    light /= np.linalg.norm(light)
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    shading = 0.25 + 0.75 * diffuse   # (H, W)

    # 5. Compose: colormap × shading, white background for empty pixels
    cmap = plt.get_cmap("plasma")
    rgb = cmap(height_norm)[:, :, :3] * shading[:, :, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    bg = np.ones((H, W, 3))
    return np.where(any_occ[:, :, None], rgb, bg)


def _show_2d(ax, img2d: torch.Tensor, label: str | None = None):
    """Normalize and display a 2-D grayscale image."""
    data = img2d.float().cpu().numpy()
    vmin, vmax = data.min(), data.max()
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    if label:
        ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=60, va="center")


def log_samples(
    x_gt: torch.Tensor,       # (N, 1, D, H, W)
    y_obs: torch.Tensor,       # (N, 1, H, W)   — corruption of GT
    x_hat1: torch.Tensor,      # (N, 1, D, H, W) — realization 1
    x_hat2: torch.Tensor,      # (N, 1, D, H, W) — realization 2
    noise_std: float,
    epoch: int,
    use_wandb: bool,
    n_cols: int = 6,
) -> None:
    """
    6 rows × n_cols columns:
      Row 0 — GT volume (3-D render)
      Row 1 — CryoEM observation fed to the model (2-D)
      Row 2 — Realization 1 predicted by the diffusion model (3-D render)
      Row 3 — CryoEM corruption of realization 1 (2-D)
      Row 4 — Realization 2 predicted by the diffusion model (3-D render)
      Row 5 — CryoEM corruption of realization 2 (2-D)
    """
    n = min(n_cols, x_gt.size(0))
    cpu = torch.device("cpu")

    # Compute corruptions of the two reconstructions (fresh random rotation each)
    with torch.no_grad():
        y_hat1 = forward_channel(x_hat1[:n].to(cpu), noise_std)   # (n,1,H,W)
        y_hat2 = forward_channel(x_hat2[:n].to(cpu), noise_std)

    row_labels = [
        "GT (3-D)",
        "Obs y = F(GT)",
        "Recon 1 (3-D)",
        "F(Recon 1)",
        "Recon 2 (3-D)",
        "F(Recon 2)",
    ]
    n_rows = len(row_labels)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.2 * n, 2.2 * n_rows),
                             squeeze=False)

    for j in range(n):
        gt_vol  = x_gt[j, 0].cpu()
        h1_vol  = x_hat1[j, 0].cpu()
        h2_vol  = x_hat2[j, 0].cpu()

        # Row 0: GT 3-D render
        axes[0, j].imshow(render_volume(gt_vol), interpolation="nearest")
        axes[0, j].axis("off")

        # Row 1: GT observation (2-D)
        _show_2d(axes[1, j], y_obs[j, 0])

        # Row 2: Realization 1 (3-D render)
        axes[2, j].imshow(render_volume(h1_vol), interpolation="nearest")
        axes[2, j].axis("off")

        # Row 3: Corruption of realization 1 (2-D)
        _show_2d(axes[3, j], y_hat1[j, 0])

        # Row 4: Realization 2 (3-D render)
        axes[4, j].imshow(render_volume(h2_vol), interpolation="nearest")
        axes[4, j].axis("off")

        # Row 5: Corruption of realization 2 (2-D)
        _show_2d(axes[5, j], y_hat2[j, 0])

    # Row labels on the leftmost column only
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=65, va="center")

    fig.suptitle(f"Epoch {epoch}", fontsize=11, y=1.01)
    plt.tight_layout()

    if use_wandb:
        wandb.log({"eval/reconstruction": wandb.Image(fig)}, step=epoch)
    else:
        out = Path("toy3d_eval")
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"epoch_{epoch:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── 7. Rotation movie ────────────────────────────────────────────────────────

def make_rotation_movie(
    x_gt: torch.Tensor,    # (N, 1, D, H, W)
    y_obs: torch.Tensor,   # (N, 1, H, W)     — fixed GT observations
    x_hat1: torch.Tensor,  # (N, 1, D, H, W)
    x_hat2: torch.Tensor,  # (N, 1, D, H, W)
    epoch: int,
    use_wandb: bool,
    n_frames: int = 72,
    fps: int = 15,
) -> None:
    """
    Save a movie of one full z-axis revolution for N shapes.

    Layout per frame (2*N rows × 3 cols):
      For each shape i:
        Row 2i:   GT render   | Recon1 render | Recon2 render  (rotating)
        Row 2i+1: GT proj     | Recon1 POV    | Recon2 POV     (GT fixed)
    """
    N = x_gt.size(0)

    # Pre-compute per-sample GT projection normalisation (stable across frames)
    gt_projs_np = [y_obs[i, 0].float().cpu().numpy() for i in range(N)]
    gt_ranges = []
    for p in gt_projs_np:
        lo, hi = float(p.min()), float(p.max())
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        gt_ranges.append((lo, hi))

    def _to_uint8_gray(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
        if vmin is None:
            vmin = arr.min()
        if vmax is None:
            vmax = arr.max()
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        gray = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb = np.stack([gray, gray, gray], axis=-1)
        return (rgb * 255).astype(np.uint8)

    col_labels = ["GT", "Recon 1", "Recon 2"]
    n_rows = 2 * N

    out_dir = Path("toy3d_eval")
    out_dir.mkdir(exist_ok=True)

    if _IMAGEIO_AVAILABLE:
        out_path = out_dir / f"epoch_{epoch:04d}_rotation.gif"
        writer = imageio.get_writer(str(out_path), fps=fps)
    else:
        frame_dir = out_dir / f"epoch_{epoch:04d}_rotation_frames"
        frame_dir.mkdir(exist_ok=True)
        writer = None

    for i in tqdm(range(n_frames)):
        phi = 2.0 * np.pi * i / n_frames
        theta = _make_z_rot_theta(phi)

        fig, axes = plt.subplots(n_rows, 3, figsize=(6.6, 2.2 * n_rows), squeeze=False)

        for s in range(N):
            r_render = 2 * s       # render row for shape s
            r_proj   = 2 * s + 1   # projection row for shape s

            renders = [
                render_volume(x_gt[s, 0].cpu(),   theta),
                render_volume(x_hat1[s, 0].cpu(), theta),
                render_volume(x_hat2[s, 0].cpu(), theta),
            ]
            lo, hi = gt_ranges[s]
            projs = [
                _to_uint8_gray(gt_projs_np[s], lo, hi),
                _to_uint8_gray(_project_from_pov(x_hat1[s, 0].cpu(), theta)),
                _to_uint8_gray(_project_from_pov(x_hat2[s, 0].cpu(), theta)),
            ]

            for j in range(3):
                if s == 0:
                    axes[r_render, j].set_title(col_labels[j], fontsize=8)
                axes[r_render, j].imshow(renders[j], interpolation="nearest")
                axes[r_render, j].axis("off")
                axes[r_proj, j].imshow(projs[j], cmap="gray", vmin=0, vmax=255,
                                       interpolation="nearest")
                axes[r_proj, j].axis("off")

            axes[r_render, 0].set_ylabel(f"Shape {s+1}\n3-D", fontsize=7,
                                         rotation=0, labelpad=50, va="center")
            axes[r_proj, 0].set_ylabel(f"Proj", fontsize=7,
                                       rotation=0, labelpad=50, va="center")

        deg = int(round(np.degrees(phi)))
        fig.suptitle(f"Epoch {epoch}  |  z-rotation {deg:3d}°", fontsize=9)
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
            wandb.log({"eval/rotation_movie": wandb.Video(str(out_path), fps=fps)},
                      step=epoch)


# ── 8. Training ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="toy_3d: supervised CryoEM 3D diffusion")
    p.add_argument("--n_per_class",  type=int,   default=50)
    p.add_argument("--vol_size",     type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--noise_std",    type=float, default=0.3)
    p.add_argument("--eval_every",   type=int,   default=10)
    p.add_argument("--n_eval",       type=int,   default=8,
                   help="Number of held-out samples to evaluate")
    p.add_argument("--sample_steps", type=int,   default=50)
    p.add_argument("--n_frames",     type=int,   default=72,
                   help="Frames in rotation movie (reduced to 8 in --debug)")
    p.add_argument("--no_wandb",     action="store_true")
    p.add_argument("--debug",        action="store_true",
                   help="2 epochs, tiny model, quick smoke test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        args.epochs       = 2
        args.n_per_class  = 4
        args.batch_size   = 4
        args.eval_every   = 1
        args.n_eval       = 4
        args.sample_steps = 5
        args.n_frames     = 8

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

    # Mixed precision: CUDA uses autocast + GradScaler; MPS/CPU use full precision.
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}  amp={use_amp}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("Generating toy dataset ...")
    x_gt = generate_toy_dataset(n_per_class=args.n_per_class, grid_size=args.vol_size)
    N = x_gt.size(0)
    print(f"  {N} volumes  shape={tuple(x_gt.shape)}  "
          f"range=[{x_gt.min():.2f}, {x_gt.max():.2f}]")

    # Fixed held-out observations — one sample per class (sphere/cube/cylinder/ellipsoid/torus)
    class_indices = [c * args.n_per_class for c in range(5)]
    x_eval = x_gt[class_indices].to(device)
    y_eval = forward_channel(x_eval, noise_std=args.noise_std)

    loader = DataLoader(
        TensorDataset(x_gt), batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    small_channels = (32, 64, 128) if not args.debug else (16, 32)
    model = ConditionalUNet3D(
        vol_size=args.vol_size,
        block_out_channels=small_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="scsi-cryoem-toy3d",
            config=vars(args) | {"n_params": n_params},
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for (x_batch,) in tqdm(loader, desc=f"epoch {epoch:4d}", leave=False):
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            y_batch = forward_channel(x_batch, noise_std=args.noise_std)

            with torch.autocast(device.type, enabled=use_amp):
                loss = si_loss(model, x_batch, y_batch)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        sched.step()
        n_batches = len(loader)
        avg_loss = running / max(n_batches, 1)
        print(f"epoch {epoch:4d}  loss={avg_loss:.5f}")
        if use_wandb:
            wandb.log({"train/loss": avg_loss, "train/lr": sched.get_last_lr()[0]},
                      step=epoch)

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                x_hat1 = sample_euler(model, y_eval, args.vol_size, n_steps=args.sample_steps)
                x_hat2 = sample_euler(model, y_eval, args.vol_size, n_steps=args.sample_steps)
            log_samples(x_eval, y_eval, x_hat1, x_hat2,
                        noise_std=args.noise_std, epoch=epoch, use_wandb=use_wandb)
            make_rotation_movie(
                x_eval.cpu(), y_eval.cpu(),
                x_hat1.cpu(), x_hat2.cpu(),
                epoch=epoch, use_wandb=use_wandb,
                n_frames=args.n_frames,
            )
            _empty_cache()

    if use_wandb:
        wandb.finish()
    print("Done.")
