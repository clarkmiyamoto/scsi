"""
EMNIST-under-3D-CryoET dataset assembly: extrude each 2D digit into a voxel volume, then run it
through the 3D->2D tilt-series channel (corruption.corruption_channel). The 3D counterpart of
cryoet_mnist/data.py -- same shape (a Config dataclass, build_observations -> TensorDataset, a
separate build_viz_pool, a __main__ visualizer), with `image_size` becoming `vol_size` and the
digit gaining a depth extent. Digits come from the EMNIST "digits" split (24k train / 4k test
per class, ~4x MNIST) and are de-transposed back to MNIST orientation on load; the 2D sibling
cryoet_mnist/ still uses plain MNIST, so the dataset is one point where the two dirs diverge.
The __main__ here opens an interactive 3D window (rotatable
marching-cubes isosurfaces, GT vs build_warmup reconstruction) rather than a 2D slice gallery.

build_warmup here wraps pseudoinverse.py (3D weighted backprojection: radial 2D |k| ramp
filter, -vol_size DC pedestal, ~120 deg single-axis coverage) to build a classical-recon warm
start from build_observations' output -- x_hat is the filtered backprojection of the OBSERVED
tilt series y, never the MNIST digit or its label, and never a fresh MNIST-through-channel
draw. The tilt-series rotations are not observed, so build_warmup resamples its own from the
channel config, exactly like the 2D cryoet_mnist/build_warmup. In 3D that costs more than in
2D: a resampled tilt series differs from the true one by a rotation on the specimen side of
every pose, which backprojection does not absorb into a global reorientation (see
pseudoinverse.py's docstring), so the 3D warm start is genuinely pose-blind and only roughly
right -- a symmetry-breaking seed for EM, not an upper bound.
"""

from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

from corruption import corruption_channel
from rotation import sample_tilt_series_rotations_so3
from pseudoinverse import pseudoinverse
# pseudoinverse.py does NOT import from this file at module level (see the NOTE there) -- that's
# what keeps this import non-circular.

VOL_SIZE = 32

# corruption_channel expands each volume to (B*num_tilts, 1, D, H, W) internally for one
# grid_sample, so the channel is applied in small chunks rather than over the whole pool at once.
_CHANNEL_BATCH = 32


@dataclass
class Config_Dataset_MNIST:
    # Dataset
    n_images_per_class: int = 23_000     # per-digit draw from EMNIST "digits" (24k train / 4k test per class)
    vol_size: int = VOL_SIZE
    inplane_size: int | None = None      # digit load resolution; default round(vol_size * 0.65)
    depth_extent: int | None = None      # depth band the digit is extruded across; default round(vol_size * 0.25)
    digit_classes: list[int] | None = None  # e.g. [3, 7]. None -> all 10 digits.

    # Corruption channel
    num_tilts: int = 16
    tilt_increment_deg: float = 7.5
    noise_std: float = 3.0
    tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Warmup (classical-recon pseudoinverse)
    filtered: bool = True
    filter_type: str = "hann"  # "hann" or "ramp"

    # Misc
    seed: int = 42
    train: bool = True


# EMNIST stores its digits transposed relative to MNIST (a 90-deg rotation + horizontal mirror).
# Undo it with a main-diagonal reflection BEFORE Resize/ToTensor so every digit lands in MNIST
# orientation -- skip this and the whole pool trains sideways-and-mirrored.
_EMNIST_DEORIENT = transforms.Lambda(lambda img: img.transpose(Image.Transpose.TRANSPOSE))


def _load_mnist_digits(config: Config_Dataset_MNIST, image_size: int) -> torch.Tensor:
    """
    n_images_per_class random EMNIST digits ("digits" split) from EACH class in digit_classes,
    de-transposed to MNIST orientation, resized to `image_size`, normalized to [-1, 1],
    concatenated in digit_classes order.

    EMNIST "digits" is class-balanced: 24_000 train / 4_000 test per digit. A per-class request
    larger than the split (e.g. the 23_000 default under --test_split) is silently truncated to
    whatever the split holds.

    Returns:
        (N, 1, image_size, image_size)
    """
    digit_classes = config.digit_classes if config.digit_classes is not None else list(range(10))

    transform = transforms.Compose([
        _EMNIST_DEORIENT,
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.EMNIST("./data", split="digits", train=config.train, download=True,
                              transform=transform)
    generator = torch.Generator().manual_seed(config.seed) if config.seed is not None else None

    chunks = []
    for digit_class in digit_classes:
        class_idx = (dataset.targets == digit_class).nonzero(as_tuple=True)[0]
        loader = DataLoader(Subset(dataset, class_idx), batch_size=config.n_images_per_class,
                            shuffle=True, generator=generator)
        x_c, _ = next(iter(loader))
        chunks.append(x_c)
    return torch.cat(chunks, dim=0)


def load_mnist_volumes(config: Config_Dataset_MNIST) -> torch.Tensor:
    """
    Extrude each 2D MNIST digit uniformly across a central depth band, leaving empty (-1) space
    at the boundary of ALL THREE axes so a generic SO(3) rotation doesn't clip the object
    against the cube's corners -- a 90-degree rotation swaps depth extent into in-plane extent,
    so depth-only margin is not enough on its own. Binding constraint: for in-plane ink radius
    r_xy and half-depth extent d, need sqrt(r_xy^2 + d^2) <= vol_size/2 - eps. The
    inplane_size / depth_extent defaults are a starting point for that, checked by
    mass-invariance under rotation (see __main__), not proven.

    Returns:
        (N, 1, vol_size, vol_size, vol_size) in [-1, 1], background -1.
    """
    V = config.vol_size
    inplane = config.inplane_size if config.inplane_size is not None else round(V * 0.65)
    depth = config.depth_extent if config.depth_extent is not None else round(V * 0.25)

    x2d = _load_mnist_digits(config, image_size=inplane)          # (N, 1, s, s)
    N = x2d.size(0)

    pad = V - inplane
    pad_lo, pad_hi = pad // 2, pad - pad // 2
    x2d = F.pad(x2d, (pad_lo, pad_hi, pad_lo, pad_hi), value=-1.0)  # (N, 1, V, V)

    margin = (V - depth) // 2
    vol = torch.full((N, 1, V, V, V), -1.0, dtype=x2d.dtype)
    vol[:, :, margin:margin + depth] = x2d.unsqueeze(2).expand(-1, -1, depth, -1, -1)
    return vol


def build_observations(config: Config_Dataset_MNIST) -> Dataset:
    """
    Load the MNIST volume pool and apply the forward model once per volume, in chunks.

    Returns:
        TensorDataset of a single (N, num_tilts, 1, vol_size, vol_size) observation tensor.
    """
    vol_gt = load_mnist_volumes(config)

    observations = []
    for i in range(0, vol_gt.size(0), _CHANNEL_BATCH):
        y_obs = corruption_channel(vol_gt[i:i + _CHANNEL_BATCH],
                                   num_tilts=config.num_tilts,
                                   tilt_increment_deg=config.tilt_increment_deg,
                                   noise_std=config.noise_std,
                                   tilt_axis=config.tilt_axis)
        observations.append(y_obs)

    return TensorDataset(torch.cat(observations, dim=0))


def _renorm_unit(vol: torch.Tensor) -> torch.Tensor:
    """
    Per-volume affine rescale to the [-1, 1] / background = -1 convention, then clamp. Raw WBP
    already lands close to this -- the ramp filter's exact DC null pins the background near -1 --
    but the ink gain is an unmodeled O(1) constant and edge ringing undershoots below -1
    (pseudoinverse.py's scale note). Anchor the per-volume median to -1 and the 99.9th
    percentile to +1, then clamp the ringing. Both the M-step interpolant and the E-step
    re-corruption (rotate_3d's +1 shift and zero padding) assume this scale.

    The median stands in for the background level because extruded-MNIST ink is sparse (a thin
    strokes-only digit across ~depth_extent/V of the volume); a much thicker extrusion where
    ink exceeds half the voxels would pull the median into the ink and map true background above
    -1. Keep this per-chunk, not over the whole pool -- torch.quantile caps out near 2^24
    elements per reduction.

    Args:
        vol: (B, 1, D, H, W)
    """
    flat = vol.flatten(1)
    bg = flat.median(dim=1).values.view(-1, 1, 1, 1, 1)
    hi = torch.quantile(flat, 0.999, dim=1).view(-1, 1, 1, 1, 1)
    return (2.0 * (vol - bg) / (hi - bg).clamp_min(1e-8) - 1.0).clamp(-1.0, 1.0)


def build_warmup(observations: Dataset, config: Config_Dataset_MNIST) -> Dataset:
    """
    Classical-reconstruction warm start for mstep_lifted -- the 3D counterpart of
    cryoet_mnist/data.py::build_warmup. Returns a TensorDataset of (x_hat, y) pairs where x_hat
    is pseudoinverse.py's filtered backprojection of the OBSERVED tilt series y.

    Consumes build_observations' output directly -- x_hat is the backprojection of the SAME y
    the E-step re-corrupts against, not a fresh MNIST-through-channel draw. The tilt-series
    rotations are NOT observed, so (like the 2D sibling) build_warmup resamples its own from the
    channel config rather than receiving them. In 3D that resampled series differs from the true
    one by a specimen-side Haar rotation backprojection cannot absorb into a global reorientation
    (pseudoinverse.py docstring), so the warm start is genuinely pose-blind and only roughly
    right -- a symmetry-breaking seed for the EM loop, not the upper bound a pose-supervised
    start would give. Still identity/label-blind: pseudoinverse() only ever sees (y, rotations).

    Returns:
        TensorDataset of (x_warmup, y_warmup): x_warmup (N, 1, V, V, V) in [-1, 1],
        y_warmup (N, num_tilts, 1, V, V) -- the same tensor `observations` holds, shared.
    """
    (y_all,) = observations.tensors
    tilt_increment_rad = config.tilt_increment_deg * torch.pi / 180.0

    xhat_chunks = []
    for i in range(0, y_all.size(0), _CHANNEL_BATCH):
        y = y_all[i:i + _CHANNEL_BATCH]
        rotations = sample_tilt_series_rotations_so3(
            y.size(0), config.num_tilts, tilt_increment_rad, tilt_axis=config.tilt_axis)
        x_hat = pseudoinverse(y, rotations, config.filtered, config.filter_type)
        xhat_chunks.append(_renorm_unit(x_hat))

    return TensorDataset(torch.cat(xhat_chunks, dim=0), y_all)


def build_viz_pool(config: Config_Dataset_MNIST, n_pool: int, viz_seed: int) -> dict:
    """
    Small diagnostic pool for wandb viz: n_pool volumes at deterministic positions in the SAME
    pool build_observations() draws from, with a fixed x0 / rotations / y seeded independently by
    viz_seed (comparable across sweeps). Runs on CPU with the RNG state saved/restored, so it has
    no effect on the rest of the run's RNG stream.

    Returns a dict with `rotations` (B, num_tilts, 3, 3) -- the 3D analogue of the 2D pool's
    scalar `theta` -- named to match corruption_channel's kwarg.
    """
    vol_gt = load_mnist_volumes(config)
    idx = torch.linspace(0, vol_gt.size(0) - 1, n_pool).round().long()
    x_gt = vol_gt[idx]

    rng_state = torch.get_rng_state()
    torch.manual_seed(viz_seed)
    V = config.vol_size
    x0 = torch.randn(x_gt.size(0), 1, V, V, V)
    tilt_increment_rad = config.tilt_increment_deg * torch.pi / 180.0
    rotations = sample_tilt_series_rotations_so3(x_gt.size(0), config.num_tilts,
                                                tilt_increment_rad, tilt_axis=config.tilt_axis)
    y = corruption_channel(x_gt, rotations=rotations, noise_std=config.noise_std)
    torch.set_rng_state(rng_state)

    return {"x_gt": x_gt, "x0": x0, "rotations": rotations, "y": y}


if __name__ == "__main__":
    # Interactive 3D viewer: marching-cubes isosurfaces of each GT digit volume (top row) and
    # its build_warmup pseudoinverse reconstruction (bottom row), in one rotatable matplotlib
    # figure. All 3D axes share a single view, so dragging any panel orbits every panel together
    # for a direct GT-vs-recon comparison; scroll to zoom. `--save PATH` also writes a static
    # PNG (headless-friendly -- there plt.show() is a no-op). A mass-invariance check on
    # load_mnist_volumes and a warmup-vs-GT scale line print first.
    import argparse

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  -- registers the "3d" projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    from rotation import rotate_3d, sample_uniform_rotation_so3

    parser = argparse.ArgumentParser(
        description="Interactive 3D view: GT volumes vs build_warmup reconstructions")
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None)
    parser.add_argument("--n_images_per_class", type=int, default=2)
    parser.add_argument("--vol_size", type=int, default=32)
    parser.add_argument("--inplane_size", type=int, default=None)
    parser.add_argument("--depth_extent", type=int, default=None)
    parser.add_argument("--num_tilts", type=int, default=16)
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5)
    parser.add_argument("--noise_std", type=float, default=3.0)
    parser.add_argument("--filter_type", type=str, default="hann", choices=["hann", "ramp"])
    parser.add_argument("--no_filtered", dest="filtered", action="store_false",
                        help="Plain (unfiltered) backprojection for the reconstruction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", action="store_true")
    parser.add_argument("--n_display", type=int, default=4,
                        help="How many (GT, reconstruction) example pairs to show.")
    parser.add_argument("--level", type=float, default=0.0,
                        help="Isosurface threshold in [-1, 1] (background -1, ink +1). Lower "
                             "picks up more of the noisy reconstruction lobes.")
    parser.add_argument("--save", type=str, default=None,
                        help="Also write a static PNG here (for headless runs).")
    parser.add_argument("--no_show", action="store_true",
                        help="Skip the interactive window (use with --save).")
    args = parser.parse_args()

    config = Config_Dataset_MNIST(
        n_images_per_class=args.n_images_per_class, vol_size=args.vol_size,
        inplane_size=args.inplane_size, depth_extent=args.depth_extent,
        digit_classes=args.digit_classes, num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg, noise_std=args.noise_std,
        filtered=args.filtered, filter_type=args.filter_type,
        seed=args.seed, train=not args.test_split,
    )
    digit_classes = config.digit_classes or list(range(10))

    vol_gt = load_mnist_volumes(config)
    torch.manual_seed(config.seed)   # build_warmup resamples its tilt series from the global RNG; seed for a repro recon
    observations = build_observations(config)
    x_hat_all = build_warmup(observations, config).tensors[0]
    N, V = vol_gt.size(0), config.vol_size        # (N, 1, V, V, V), renormed to [-1, 1]

    # Mass invariance: total ink (in the shifted x+1 frame) should survive random SO(3) rotation
    # up to interpolation blur. A large drop means the digit is clipping the cube corners.
    probe = vol_gt[:min(16, N)]
    ink0 = (probe + 1.0).sum().item()
    ink_rot = (rotate_3d(probe, sample_uniform_rotation_so3(probe.size(0))) + 1.0).sum().item()
    print(f"load_mnist_volumes mass check: ink {ink0:.0f} -> {ink_rot:.0f} after SO(3) "
          f"({100 * (ink_rot - ink0) / ink0:+.1f}%);  V={V} "
          f"inplane={config.inplane_size or round(V * 0.65)} "
          f"depth={config.depth_extent or round(V * 0.25)}")
    # Warmup lands on [-1, 1] by construction (_renorm_unit); the informative check is whether
    # the background actually sits at -1 -- a corner voxel should, for both x_hat and GT.
    print(f"build_warmup vs GT  |  x_hat min/mean/max "
          f"{x_hat_all.min():+.2f}/{x_hat_all.mean():+.2f}/{x_hat_all.max():+.2f}   "
          f"GT {vol_gt.min():+.2f}/{vol_gt.mean():+.2f}/{vol_gt.max():+.2f}   "
          f"bg-corner x_hat {x_hat_all[:, 0, 0, 0, 0].mean():+.2f}  GT {vol_gt[:, 0, 0, 0, 0].mean():+.2f}")

    n_display = min(args.n_display, N)
    idx = torch.linspace(0, N - 1, n_display).round().long().tolist()

    def _draw_isosurface(ax, vol: torch.Tensor, level: float, color: str) -> None:
        """Marching-cubes isosurface of a (V, V, V) [-1, 1] volume onto a 3D axis."""
        v = vol.numpy()
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1), zoom=1.6)   # fills the axis; mpl 3D is whitespace-heavy
        if not (float(v.min()) < level < float(v.max())):
            ax.text2D(0.5, 0.5, "empty at\nthis level", ha="center", va="center",
                      transform=ax.transAxes, fontsize=8)
            return
        verts, faces, _, _ = marching_cubes(v, level=level)
        mesh = Poly3DCollection(verts[faces], alpha=0.6, linewidths=0.0, facecolor=color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, v.shape[0]); ax.set_ylim(0, v.shape[1]); ax.set_zlim(0, v.shape[2])

    fig = plt.figure(figsize=(3.1 * n_display, 5.2))
    axes3d = []
    for col, i in enumerate(idx):
        ax_gt = fig.add_subplot(2, n_display, col + 1, projection="3d")
        ax_hat = fig.add_subplot(2, n_display, n_display + col + 1, projection="3d")
        _draw_isosurface(ax_gt, vol_gt[i, 0], args.level, "#4c72b0")
        _draw_isosurface(ax_hat, x_hat_all[i, 0], args.level, "#c44e52")
        ax_gt.set_title(f"GT  idx={i}", fontsize=9)
        ax_hat.set_title(f"reconstruction  idx={i}", fontsize=9)
        axes3d += [ax_gt, ax_hat]

    axes3d[0].view_init(elev=18, azim=-60)
    for ax in axes3d[1:]:
        ax.shareview(axes3d[0])          # drag any panel -> every panel orbits together

    fig.suptitle(f"GT vs build_warmup reconstruction (POSE-BLIND: recon at an arbitrary "
                 f"orientation, geometry smeared by the unknown mount)  |  isosurface @ "
                 f"{args.level:+.2f}  digits={digit_classes}  T={config.num_tilts}  "
                 f"noise_std={config.noise_std}  filter={config.filter_type}", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.02, hspace=0.12)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    if not args.no_show:
        backend = plt.get_backend().lower()
        if backend in ("agg", "pdf", "ps", "svg", "template"):
            print(f"matplotlib backend {backend!r} is non-interactive; "
                  f"re-run with --save PATH for a static image.")
        else:
            print("Interactive 3D window: drag to rotate (all panels linked), scroll to zoom, "
                  "close to exit.")
            plt.show()
