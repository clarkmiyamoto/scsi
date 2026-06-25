"""Command-line interface: `pointcloud-fm train ...` and `pointcloud-fm sample ...`."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import available_shapes, torus_surface_residual
from .device import resolve_device
from .flow import ModelConfig, load_checkpoint, sample, save_checkpoint, train
from .tracking import Tracker


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dim", type=int, default=128, help="feature width")
    p.add_argument("--depth", type=int, default=6, help="number of set-attention blocks")
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument("--n-points", type=int, default=512, help="points per cloud (N)")


def _add_wandb_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="pointcloud-fm")
    p.add_argument("--wandb-name", default=None, help="W&B run name")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pointcloud-fm", description=__doc__)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="accelerator (default: auto-detect CUDA > MPS > CPU)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train the flow-matching velocity field")
    _add_model_args(pt)
    pt.add_argument("--steps", type=int, default=4000)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--log-every", type=int, default=200)
    pt.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    pt.add_argument("--compile", action="store_true", help="use torch.compile")
    pt.add_argument("--out", default="checkpoint.pt", help="where to save weights")
    pt.add_argument(
        "--sample-every", type=int, default=0,
        help="log generated 3D clouds to W&B every N steps (0 = off)",
    )
    pt.add_argument("--sample-n", type=int, default=4, help="clouds per W&B log")
    pt.add_argument("--sample-steps", type=int, default=50, help="ODE steps for logging")
    pt.add_argument(
        "--ball-radius", type=float, default=0.05,
        help="radius of the ball placed at each point in the logged mesh (0=off)",
    )
    pt.add_argument("--ball-subdivisions", type=int, default=1, help="icosphere detail")
    _add_wandb_args(pt)

    ps = sub.add_parser("sample", help="generate point clouds from a checkpoint")
    ps.add_argument("--ckpt", default="checkpoint.pt")
    ps.add_argument("--n", type=int, default=4, help="number of clouds to generate")
    ps.add_argument("--n-points", type=int, default=None, help="override N at sampling")
    ps.add_argument("--steps", type=int, default=100, help="Euler ODE steps")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out-npy", default="samples.npy")
    ps.add_argument("--out-png", default="samples.png", help="'' to skip plotting")
    _add_wandb_args(ps)

    # ---- balls: place a solid ball at every point and visualize the mesh ----
    pb = sub.add_parser(
        "balls", help="place a ball at every point -> solid mesh (.obj) + preview"
    )
    src = pb.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="generate clouds from this checkpoint")
    src.add_argument("--npy", help="load clouds from a .npy file (N,3) or (M,N,3)")
    pb.add_argument("--radius", type=float, default=0.05, help="ball radius")
    pb.add_argument("--subdivisions", type=int, default=1, help="icosphere detail")
    pb.add_argument("--n", type=int, default=4, help="clouds (when sampling from --ckpt)")
    pb.add_argument("--n-points", type=int, default=None, help="override N when sampling")
    pb.add_argument("--steps", type=int, default=100, help="ODE steps when sampling")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--out-obj", default="balls.obj", help="mesh path (per-cloud suffix)")
    pb.add_argument("--out-png", default="balls.png", help="'' to skip PNG preview")
    pb.add_argument("--png-max-balls", type=int, default=200, help="balls per PNG panel")
    _add_wandb_args(pb)

    # ---- scsi: lifted SCSI -- recover a 3D cloud prior from CryoET projections ----
    pe = sub.add_parser(
        "scsi",
        help="lifted SCSI: recover a 3D point-cloud prior from CryoET tilt-series "
             "projections via tomo bootstrap + supervised pretraining + EM",
    )
    _add_model_args(pe)  # --dim --depth --heads --n-points
    pe.add_argument("--image-size", type=int, default=32, help="projection / conditioning P")
    pe.add_argument("--patch-size", type=int, default=4, help="image-encoder patch size")
    pe.add_argument(
        "--supervised", action="store_true",
        help="debug oracle: train directly on (x, F(x)) with unlimited fresh ground "
             "truth instead of EM (ignores --n-objects/--em-steps/--epochs-per-em/"
             "--pretrain-steps/--coupled-fraction; uses --steps/--eval-every/--shape)",
    )
    pe.add_argument("--steps", type=int, default=4000, help="[supervised] gradient steps")
    pe.add_argument("--eval-every", type=int, default=500, help="[supervised] eval panel cadence")
    pe.add_argument("--n-objects", type=int, default=128, help="number of observations (|p_corrupted|)")
    pe.add_argument("--em-steps", type=int, default=30)
    pe.add_argument("--epochs-per-em", type=int, default=2, help="E-step epochs per EM iteration")
    pe.add_argument("--batch", type=int, default=32)
    pe.add_argument("--lr", type=float, default=2e-4)
    pe.add_argument("--radius", type=float, default=0.08, help="ball radius for the channel F")
    pe.add_argument("--noise-std", type=float, default=0.1,
                    help="Z: white Gaussian image noise std (added to the projection)")
    pe.add_argument("--coord-noise-std", type=float, default=0.0,
                    help="W: AWGN std on the 3D point coordinates (iid per coordinate per "
                         "particle), added before rotation; the full channel is P G (X+W)+Z")
    pe.add_argument("--extent", type=float, default=2.0, help="world half-extent mapped to the image")
    pe.add_argument(
        "--n-tilts", type=int, default=11,
        help="number of projections K in the CryoET tilt series",
    )
    pe.add_argument(
        "--tilt-step", type=float, default=12.0,
        help="degrees between consecutive tilts (K tilts centred at 0)",
    )
    pe.add_argument(
        "--tilt-axis", choices=["x", "y"], default="y",
        help="tilt axis (out-of-plane; 'z' would be in-plane/degenerate)",
    )
    pe.add_argument(
        "--tomo-vol", type=int, default=48,
        help="tomo bootstrap back-projection grid resolution (vol^3)",
    )
    pe.add_argument(
        "--tomo-quantile", type=float, default=0.15,
        help="tomo bootstrap space-carving quantile over tilts "
             "(0.0 = strict intersection/min, 0.5 = median)",
    )
    pe.add_argument("--sample-steps", type=int, default=50, help="M-step Euler ODE steps")
    pe.add_argument("--coupled-fraction", type=float, default=0.0,
                    help="fraction of E-step batch using paired z from the M-step")
    pe.add_argument(
        "--shape", nargs="+", choices=available_shapes(), default=["torus"],
        metavar="SHAPE",
        help="dataset shape(s) as a uniform mixture, e.g. --shape cylinder torus",
    )
    pe.add_argument(
        "--dataset", choices=["iid", "template"], default="iid",
        help="ground-truth construction: 'iid' = N independent fresh shape samples; "
             "'template' = N bounded perturbations of fixed canonical template(s) "
             "(cryo-ET / subtomogram-averaging dataset; see --dataset-eps)",
    )
    pe.add_argument(
        "--dataset-eps", type=float, default=0.0,
        help="[--dataset template] max per-point perturbation epsilon "
             "(||delta_n|| <= eps); 0 = identical copies of the template",
    )
    pe.add_argument(
        "--pretrain-steps", type=int, default=2000,
        help="epochs of supervised pretraining on the tomo bootstrap X_boot before EM",
    )
    pe.add_argument("--n-eval", type=int, default=4, help="objects shown in eval panels")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    pe.add_argument("--out", default="scsi_checkpoint.pt", help="final checkpoint path")
    pe.add_argument("--eval-dir", default="toy3d_pc_eval", help="where eval PNGs are written")
    pe.add_argument("--viz-ball-radius", type=float, default=0.05,
                    help="ball radius for logged W&B meshes (0=off)")
    pe.add_argument(
        "--ema-decay", type=float, default=0.999,
        help="EMA decay for the M-step sampler (per optimizer step); 0.0 disables EMA",
    )
    pe.add_argument("--debug", action="store_true", help="tiny smoke-test config")
    _add_wandb_args(pe)
    return p


def _indexed(path: str, i: int, n: int) -> str:
    """balls.obj -> balls_0.obj when there is more than one cloud."""
    if n == 1:
        return path
    stem, dot, ext = path.rpartition(".")
    return f"{stem}_{i}.{ext}" if dot else f"{path}_{i}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    if args.cmd == "train":
        cfg = ModelConfig(
            dim=args.dim, depth=args.depth, heads=args.heads, n_points=args.n_points
        )
        tracker = Tracker(
            enabled=args.wandb,
            project=args.wandb_project,
            name=args.wandb_name,
            config={**vars(args), "device": device.type},
            job_type="train",
        )
        with tracker:
            model = train(
                device=device,
                cfg=cfg,
                steps=args.steps,
                batch=args.batch,
                lr=args.lr,
                use_amp=not args.no_amp,
                compile_model=args.compile,
                seed=args.seed,
                log_every=args.log_every,
                tracker=tracker,
                sample_every=args.sample_every,
                sample_n=args.sample_n,
                sample_steps=args.sample_steps,
                ball_radius=args.ball_radius,
                ball_subdivisions=args.ball_subdivisions,
            )
            save_checkpoint(args.out, model, cfg)
            print(f"[train] saved checkpoint -> {args.out}")

    elif args.cmd == "sample":
        model, cfg = load_checkpoint(args.ckpt, device)
        n_points = args.n_points or cfg.n_points
        clouds = sample(
            model, device, n_clouds=args.n, n_points=n_points,
            n_steps=args.steps, seed=args.seed,
        )
        residual = torus_surface_residual(clouds).item()
        print(
            f"[sample] {tuple(clouds.shape)}  surface residual={residual:.4f} "
            f"(torus target r^2=0.16)"
        )
        np.save(args.out_npy, clouds.float().cpu().numpy())
        print(f"[sample] wrote {args.out_npy}")
        if args.out_png:
            from .plot import save_scatter

            save_scatter(clouds, args.out_png)

        if args.wandb:
            with Tracker(
                enabled=True,
                project=args.wandb_project,
                name=args.wandb_name,
                config={**vars(args), "device": device.type},
                job_type="sample",
            ) as tracker:
                tracker.log_clouds("samples_3d", clouds)
                tracker.log({"surface_residual": residual})
                if args.out_png:
                    tracker.log_image("samples_png", args.out_png)

    elif args.cmd == "scsi":
        from .scsi import ConditionalModelConfig, scsi_train, train_supervised

        if args.debug:
            args.dim, args.depth, args.heads = 64, 2, 4
            args.n_objects, args.n_points = 8, 128
            args.em_steps, args.epochs_per_em = 2, 1
            args.batch, args.sample_steps, args.n_eval = 4, 5, 4
            args.steps, args.eval_every = 20, 10
            args.pretrain_steps = 10
            args.n_tilts = min(args.n_tilts, 5)

        cfg = ConditionalModelConfig(
            dim=args.dim, depth=args.depth, heads=args.heads,
            n_points=args.n_points, image_size=args.image_size, patch_size=args.patch_size,
            in_channels=args.n_tilts,
        )
        tracker = Tracker(
            enabled=args.wandb,
            project=args.wandb_project,
            name=args.wandb_name,
            config={**vars(args), "device": device.type},
            job_type="supervised" if args.supervised else "scsi",
        )
        with tracker:
            if args.supervised:
                train_supervised(
                    device=device, cfg=cfg,
                    steps=args.steps, batch=args.batch, lr=args.lr,
                    radius=args.radius, noise_std=args.noise_std, extent=args.extent,
                    sample_steps=args.sample_steps, n_eval=args.n_eval,
                    eval_every=args.eval_every,
                    use_amp=not args.no_amp, seed=args.seed, shapes=args.shape,
                    tracker=tracker, out=args.out, eval_dir=args.eval_dir,
                    viz_ball_radius=args.viz_ball_radius,
                    coord_noise_std=args.coord_noise_std,
                    n_tilts=args.n_tilts, tilt_step=args.tilt_step, tilt_axis=args.tilt_axis,
                )
            else:
                scsi_train(
                    device=device, cfg=cfg,
                    n_objects=args.n_objects, em_steps=args.em_steps,
                    epochs_per_em=args.epochs_per_em,
                    batch=args.batch, lr=args.lr,
                    radius=args.radius, noise_std=args.noise_std, extent=args.extent,
                    sample_steps=args.sample_steps, coupled_fraction=args.coupled_fraction,
                    shapes=args.shape, pretrain_steps=args.pretrain_steps,
                    n_eval=args.n_eval,
                    use_amp=not args.no_amp, seed=args.seed,
                    tracker=tracker, out=args.out, eval_dir=args.eval_dir,
                    viz_ball_radius=args.viz_ball_radius,
                    coord_noise_std=args.coord_noise_std,
                    n_tilts=args.n_tilts, tilt_step=args.tilt_step, tilt_axis=args.tilt_axis,
                    tomo_vol=args.tomo_vol, tomo_quantile=args.tomo_quantile,
                    dataset=args.dataset, dataset_eps=args.dataset_eps,
                    ema_decay=args.ema_decay,
                )

    elif args.cmd == "balls":
        from .balls import render_balls_png, save_balls_obj

        if args.ckpt:
            model, cfg = load_checkpoint(args.ckpt, device)
            n_points = args.n_points or cfg.n_points
            clouds = sample(
                model, device, n_clouds=args.n, n_points=n_points,
                n_steps=args.steps, seed=args.seed,
            ).cpu().numpy()
        else:
            clouds = np.load(args.npy)
        if clouds.ndim == 2:
            clouds = clouds[None]
        n = clouds.shape[0]

        obj_paths = []
        for i in range(n):
            path = _indexed(args.out_obj, i, n)
            v, f = save_balls_obj(clouds[i], path, args.radius, args.subdivisions)
            print(f"[balls] wrote {path}  ({v} verts, {f} faces, radius={args.radius})")
            obj_paths.append(path)

        if args.out_png:
            render_balls_png(
                list(clouds), args.out_png, args.radius, args.subdivisions,
                max_balls=args.png_max_balls,
            )

        if args.wandb:
            with Tracker(
                enabled=True,
                project=args.wandb_project,
                name=args.wandb_name,
                config={**vars(args), "device": device.type},
                job_type="balls",
            ) as tracker:
                tracker.log_clouds("samples_3d", torch.from_numpy(clouds))
                tracker.log_meshes("samples_balls", obj_paths)
                if args.out_png:
                    tracker.log_image("samples_balls_png", args.out_png)


if __name__ == "__main__":
    main()
