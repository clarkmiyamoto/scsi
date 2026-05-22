"""Quick visualization of the 'revolve' bootstrap for a sphere."""
import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

# ── copy the minimal functions needed ─────────────────────────────────────────

GRID_SIZE = 16
MAX_ALPHA  = 0.85

def _make_grid(g):
    lin = torch.linspace(-1.0, 1.0, g)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    return xx, yy, zz

def make_sphere(g=GRID_SIZE, r=0.42):
    xx, yy, zz = _make_grid(g)
    vol = ((xx**2 + yy**2 + zz**2) <= r**2).float()
    return (vol * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)   # (1,1,D,H,W)

def random_so3_rotate(x):
    B = x.size(0)
    R_np = Rotation.random(B).as_matrix().astype(np.float32)
    R = torch.from_numpy(R_np)
    zeros = torch.zeros(B, 3, 1)
    theta = torch.cat([R, zeros], dim=2)
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    return F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="border")

def forward_channel(x, noise_std=0.3):
    return random_so3_rotate(x).sum(dim=2) + noise_std * torch.randn(x.shape[0], 1, GRID_SIZE, GRID_SIZE)

def revolve(y_obs, vol_size=GRID_SIZE):
    N = y_obs.size(0)
    D = vol_size
    flat = y_obs.reshape(N, -1)
    mn = flat.min(1).values.view(N,1,1,1)
    mx = flat.max(1).values.view(N,1,1,1)
    obs_norm = (y_obs - mn) / (mx - mn + 1e-8)

    lin = torch.linspace(-1.0, 1.0, D)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    r_vox = torch.sqrt(xx**2 + zz**2)

    grid = torch.stack(
        [r_vox.reshape(D*D, D), yy.reshape(D*D, D)], dim=-1
    ).unsqueeze(0).expand(N, -1, -1, -1)

    sampled = F.grid_sample(obs_norm, grid, align_corners=True,
                             mode="bilinear", padding_mode="zeros")
    return sampled.reshape(N, 1, D, D, D) * 2.0 - 1.0

# ── show_voxels helper ────────────────────────────────────────────────────────

def show_voxels(ax, vol):
    v = vol.float().numpy()
    vmin, vmax = v.min(), v.max()
    vn = (v - vmin) / (vmax - vmin + 1e-8)
    filled = vn > 0.15
    rgba = np.zeros((*v.shape, 4))
    rgba[filled, 0] = 0.27
    rgba[filled, 1] = 0.51
    rgba[filled, 2] = 0.71
    rgba[filled, 3] = np.clip(vn[filled] * MAX_ALPHA, 0, MAX_ALPHA)
    ax.voxels(filled, facecolors=rgba, edgecolors="none", shade=False)
    ax.view_init(elev=25, azim=45)
    ax.set_axis_off()

def show_img(ax, img, title=""):
    d = img.float().numpy()
    ax.imshow(d, cmap="gray", vmin=d.min(), vmax=d.max(), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")

def show_slice(ax, vol, dim, idx=None, title=""):
    v = vol.float().numpy()
    mid = v.shape[dim] // 2 if idx is None else idx
    sl = np.take(v, mid, axis=dim)
    ax.imshow(sl, cmap="gray", vmin=v.min(), vmax=v.max(), interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.axis("off")

# ── generate data ─────────────────────────────────────────────────────────────

torch.manual_seed(42)
N = 4  # show 4 independent random projections

x_sphere = make_sphere().expand(N, -1, -1, -1, -1)   # (N,1,D,H,W)
y_obs     = forward_channel(x_sphere, noise_std=0.3)   # (N,1,H,W)
x_rev     = revolve(y_obs)                             # (N,1,D,H,W)

# ── figure ────────────────────────────────────────────────────────────────────
#   Rows:   [0] GT sphere (3D) | [1] y_obs projection | [2] revolved 3D | [3] revolved slices XY/XZ/YZ
#   Columns: one per sample

n_samples_shown = N
n_slice_cols = 3  # XY, XZ, YZ per sample
total_cols = n_samples_shown

fig = plt.figure(figsize=(3.2 * total_cols, 12))
n_rows = 4

# Row 0 – GT sphere (same for all columns, 3D)
for j in range(n_samples_shown):
    ax = fig.add_subplot(n_rows, total_cols, j + 1, projection="3d")
    show_voxels(ax, x_sphere[0, 0])
    if j == 0:
        ax.set_title("GT sphere (3D)", fontsize=8, pad=2)

# Row 1 – y_obs: random SO(3) projection + noise
for j in range(n_samples_shown):
    ax = fig.add_subplot(n_rows, total_cols, total_cols + j + 1)
    show_img(ax, y_obs[j, 0], title=f"y_obs sample {j}" if j == 0 else f"sample {j}")

# Row 2 – revolved volume (3D)
for j in range(n_samples_shown):
    ax = fig.add_subplot(n_rows, total_cols, 2 * total_cols + j + 1, projection="3d")
    show_voxels(ax, x_rev[j, 0])
    if j == 0:
        ax.set_title("Revolve π(0) (3D)", fontsize=8, pad=2)

# Row 3 – revolved volume mid-slices (XY / XZ / YZ) for first sample only, rest empty
# Use a sub-grid approach: place 3 slim axes per column
for j in range(n_samples_shown):
    vol = x_rev[j, 0]    # (D,H,W)
    # Subdivide this column's subplot slot into 3 horizontal strips via inset_axes
    ax_parent = fig.add_subplot(n_rows, total_cols, 3 * total_cols + j + 1)
    ax_parent.axis("off")
    bbox = ax_parent.get_position()
    w = bbox.width / 3.1
    h = bbox.height * 0.85
    y0 = bbox.y0 + bbox.height * 0.075

    for k, (dim, lbl) in enumerate([(0, "XY"), (1, "XZ"), (2, "YZ")]):
        left = bbox.x0 + k * (bbox.width / 3.0)
        ax_s = fig.add_axes([left, y0, w, h])
        show_slice(ax_s, vol, dim=dim, title=lbl if j == 0 else "")

# ── labels ────────────────────────────────────────────────────────────────────
row_labels = ["GT sphere", "y_obs\n(SO3+proj+noise)", "Revolve π(0)\n(3D voxels)", "Revolve π(0)\n(mid-slices)"]
for r, lbl in enumerate(row_labels):
    # place label to the left of first subplot in each row
    ax0_idx = r * total_cols + 1
    ax0 = fig.axes[ax0_idx - 1]
    if hasattr(ax0, "text2D"):
        ax0.text2D(-0.12, 0.5, lbl, transform=ax0.transAxes,
                   fontsize=8, va="center", ha="right", fontweight="bold")
    else:
        ax0.text(-0.12, 0.5, lbl, transform=ax0.transAxes,
                 fontsize=8, va="center", ha="right", fontweight="bold")

fig.suptitle("Bootstrap 'revolve' — sphere", fontsize=11, y=0.99)
plt.tight_layout(rect=[0.06, 0, 1, 0.97])

out = "toy_3d/revolve_sphere_viz.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
