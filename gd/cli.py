"""Command-line interface: `python -m gd ...` -- direct gradient-descent
reconstruction of a single 3D shape from a finite CryoET tilt-series dataset."""
from __future__ import annotations

import argparse

from .device import resolve_device
from .shapes import available_shapes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gd", description=__doc__)

    p.add_argument("--shape", choices=available_shapes(), default="torus")
    p.add_argument("--param", choices=["pointcloud", "voxel"], default="pointcloud",
                   help="how X is parameterized")
    p.add_argument("--method", choices=["distributional", "joint_pose"], default="distributional",
                   help="how simulated projections of X are compared to the real dataset")
    p.add_argument("--dist-metric", choices=["sliced_wasserstein", "mmd"], default="sliced_wasserstein",
                   help="[method=distributional] distance between simulated/real projection batches")
    p.add_argument("--splat-kernel", choices=["gaussian", "ball", "cube"], default="gaussian",
                   help="projection footprint of each rendered point")
    p.add_argument("--splat-size", type=float, default=0.08,
                   help="kernel radius/half-width (gaussian: sigma; ball/cube: radius/half-width)")

    p.add_argument("--n-tilts", type=int, default=11, help="T: tilts per particle")
    p.add_argument("--tilt-step", type=float, default=12.0, help="degrees between consecutive tilts")
    p.add_argument("--tilt-axis", choices=["x", "y"], default="y")
    p.add_argument("--n-objects", type=int, default=128, help="N: dataset size")
    p.add_argument("--image-size", type=int, default=32, help="p: projection resolution")
    p.add_argument("--noise-std", type=float, default=3.0, help="AWGN std added per projection")
    p.add_argument("--extent", type=float, default=2.0, help="world half-extent mapped to the image")

    p.add_argument("--n-points", type=int, default=512, help="[param=pointcloud] points in X")
    p.add_argument("--vol-size", type=int, default=16, help="[param=voxel] grid resolution of X")
    p.add_argument("--gt-resolution", type=int, default=None,
                   help="ground-truth resolution: default 4x n_points for pointcloud (point count "
                        "scales linearly), or exactly vol_size for voxel (point count scales as "
                        "vol_size^3, so a naive '4x' would mean 64x more points -- the render "
                        "cost is O(n_objects * n_tilts * resolution * image_size), which for "
                        "voxel grids gets very large very fast)")
    p.add_argument("--tomo-vol", type=int, default=32, help="backprojection carve-grid resolution")

    p.add_argument("--init", choices=["backproject", "random"], default="backproject")
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr-shape", type=float, default=1e-2)
    p.add_argument("--lr-pose", type=float, default=1e-2, help="[method=joint_pose] pose learning rate")
    p.add_argument("--lr-schedule", choices=["none", "cosine", "linear", "exponential"], default="none",
                   help="decay shape applied to both lr-shape and lr-pose over n-steps")
    p.add_argument("--lr-min-frac", type=float, default=0.01,
                   help="[lr-schedule != none] final LR as a fraction of the initial LR")
    p.add_argument("--l1-weight", type=float, default=0.0,
                   help="[param=voxel] L1 penalty coefficient on voxel occupancy weights "
                        "(mean of sigmoid(logits)) -- discourages background 'salt-and-pepper' "
                        "drift over long runs")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-points", type=int, default=1024, help="points used for the Chamfer eval metric")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--wandb-project", default="gd-cryoet")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--out-dir", default="gd_runs", help="where PNG snapshots are written")
    p.add_argument("--debug", action="store_true", help="tiny smoke-test config")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    def _default_gt_resolution(args) -> int:
        # Point count scales linearly with n_points but as vol_size^3 for voxels,
        # so voxel ground truth is NOT multiplied -- see --gt-resolution help.
        return 4 * args.n_points if args.param == "pointcloud" else args.vol_size

    if args.gt_resolution is None:
        args.gt_resolution = _default_gt_resolution(args)

    if args.debug:
        args.n_objects, args.n_tilts = 8, 5
        args.n_points, args.vol_size = 128, 12
        args.gt_resolution = _default_gt_resolution(args)
        args.tomo_vol = 12
        args.n_steps, args.log_every, args.batch = 20, 5, 4

    device = resolve_device(args.device)
    from .optimize import run

    run(args, device)


if __name__ == "__main__":
    main()
