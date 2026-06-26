"""Command-line interface: ``python -m toy_3d_pc scsi ...``.

Runs lifted SCSI for CryoET in the point-cloud representation: recover a generative prior
over clean 3D point clouds from only their corrupted tilt-series projections, via
F-dagger bootstrap + warm-start + the literal self-consistent EM loop. ``--supervised``
runs the oracle baseline instead. W&B logging is ON by default (``--no-wandb`` to disable).
"""
from __future__ import annotations

import argparse

from .data import available_shapes
from .device import resolve_device
from .model import ConditionalModelConfig
from .tracking import Tracker


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toy_3d_pc", description=__doc__)
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="accelerator (default: auto-detect CUDA > MPS > CPU)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("scsi", help="lifted SCSI (CryoET, point clouds)")

    # Model.
    pe.add_argument("--dim", type=int, default=128, help="feature width")
    pe.add_argument("--depth", type=int, default=6, help="number of set-attention blocks")
    pe.add_argument("--heads", type=int, default=4, help="attention heads")
    pe.add_argument("--n-points", type=int, default=512, help="points per cloud (N)")
    pe.add_argument("--image-size", type=int, default=32, help="projection / conditioning P")
    pe.add_argument("--patch-size", type=int, default=4, help="image-encoder patch size")

    # Forward model F.
    pe.add_argument("--radius", type=float, default=0.08, help="blob radius/sigma for G")
    pe.add_argument("--splat", choices=["gaussian", "ball"], default="gaussian",
                    help="G kernel: gaussian splat, or solid/filled ball (filled disk in projection)")
    pe.add_argument("--noise-std", type=float, default=0.1, help="Z: white image-noise std")
    pe.add_argument("--coord-noise-std", type=float, default=0.0,
                    help="W: AWGN std on 3D point coords before rotation")
    pe.add_argument("--extent", type=float, default=2.0, help="world half-extent mapped to the image")
    pe.add_argument("--n-tilts", type=int, default=32, help="number of projections K in the tilt series")
    pe.add_argument("--tilt-step", type=float, default=5.0, help="degrees between consecutive tilts")
    pe.add_argument("--tilt-axis", choices=["x", "y"], default="y", help="tilt axis (out-of-plane)")
    pe.add_argument("--tomo-vol", type=int, default=48, help="F-dagger back-projection grid (vol^3)")
    pe.add_argument("--tomo-quantile", type=float, default=0.15,
                    help="F-dagger space-carving quantile over tilts (0=strict min, 0.5=median)")

    # SCSI loop.
    pe.add_argument("--n-objects", type=int, default=2048, help="number of observations (|mu|)")
    pe.add_argument("--em-steps", type=int, default=100, help="outer EM iterations K")
    pe.add_argument("--training-steps", type=int, default=200,
                    help="inner SGD training steps per EM iteration (T_tr)")
    pe.add_argument("--batch", type=int, default=128)
    pe.add_argument("--lr", type=float, default=2e-4)
    pe.add_argument("--sample-steps", type=int, default=64, help="Euler steps in the transport ODE")
    pe.add_argument("--alpha-z", type=float, default=0.05, help="noise-coupling prob (z = z')")
    pe.add_argument("--alpha-y", type=float, default=0.05, help="obs-coupling prob (y-hat = y)")
    pe.add_argument("--ema-decay", type=float, default=0.999,
                    help="gamma: EMA decay over the outer EM loop")
    pe.add_argument("--pretrain-steps", type=int, default=2000, help="warm-start SGD steps")
    pe.add_argument("--interpolant-style", choices=["linear", "gvp"], default="gvp")

    # Data / eval.
    pe.add_argument("--shape", nargs="+", choices=available_shapes(), default=["torus"],
                    metavar="SHAPE", help="dataset shape(s) as a uniform mixture")
    pe.add_argument("--dataset", choices=["iid", "template"], default="iid",
                    help="iid = fresh shape samples; template = perturbed copies of a fixed template")
    pe.add_argument("--dataset-eps", type=float, default=0.0,
                    help="[--dataset template] max per-point perturbation (||delta|| <= eps)")
    pe.add_argument("--n-eval", type=int, default=6, help="objects shown in eval panels")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--out", default="toy_3d_pc_checkpoint.pt", help="final checkpoint path")
    pe.add_argument("--eval-dir", default="toy_3d_pc_eval", help="where eval PNGs are written")

    # Supervised oracle.
    pe.add_argument("--supervised", action="store_true",
                    help="debug oracle: train on (x, F(x)) with fresh GT instead of EM")
    pe.add_argument("--steps", type=int, default=4000, help="[supervised] gradient steps")
    pe.add_argument("--eval-every", type=int, default=500, help="[supervised] eval cadence")

    # Infra.
    pe.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    pe.add_argument("--no-wandb", action="store_true", help="disable W&B (on by default)")
    pe.add_argument("--wandb-project", default="toy3d-pc-scsi")
    pe.add_argument("--wandb-name", default=None, help="W&B run name")
    pe.add_argument("--debug", action="store_true", help="tiny smoke-test config")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    if args.debug:
        args.dim, args.depth, args.heads = 64, 2, 4
        args.n_objects, args.n_points = 8, 128
        args.em_steps, args.training_steps = 2, 5
        args.batch, args.sample_steps, args.n_eval = 4, 5, 4
        args.steps, args.eval_every = 20, 10
        args.pretrain_steps = 10
        args.n_tilts = min(args.n_tilts, 5)

    cfg = ConditionalModelConfig(
        dim=args.dim, depth=args.depth, heads=args.heads, n_points=args.n_points,
        image_size=args.image_size, patch_size=args.patch_size, in_channels=args.n_tilts,
    )
    tracker = Tracker(
        enabled=not args.no_wandb,
        project=args.wandb_project, name=args.wandb_name,
        config={**vars(args), "device": device.type},
        job_type="supervised" if args.supervised else "scsi",
    )
    with tracker:
        if args.supervised:
            from .supervised import train_supervised

            train_supervised(
                device=device, cfg=cfg,
                steps=args.steps, batch=args.batch, lr=args.lr,
                radius=args.radius, noise_std=args.noise_std, extent=args.extent,
                sample_steps=args.sample_steps, n_eval=args.n_eval, eval_every=args.eval_every,
                use_amp=not args.no_amp, seed=args.seed, style=args.interpolant_style,
                shapes=args.shape, tracker=tracker, out=args.out, eval_dir=args.eval_dir,
                coord_noise_std=args.coord_noise_std,
                n_tilts=args.n_tilts, tilt_step=args.tilt_step, tilt_axis=args.tilt_axis,
                splat=args.splat,
            )
        else:
            from .scsi import scsi_train

            scsi_train(
                device=device, cfg=cfg,
                n_objects=args.n_objects, em_steps=args.em_steps,
                training_steps=args.training_steps, batch=args.batch, lr=args.lr,
                radius=args.radius, noise_std=args.noise_std, extent=args.extent,
                sample_steps=args.sample_steps, alpha_z=args.alpha_z, alpha_y=args.alpha_y,
                ema_decay=args.ema_decay, pretrain_steps=args.pretrain_steps,
                style=args.interpolant_style, shapes=args.shape, n_eval=args.n_eval,
                use_amp=not args.no_amp, seed=args.seed, tracker=tracker, out=args.out,
                eval_dir=args.eval_dir,
                coord_noise_std=args.coord_noise_std, n_tilts=args.n_tilts,
                tilt_step=args.tilt_step, tilt_axis=args.tilt_axis, splat=args.splat,
                tomo_vol=args.tomo_vol, tomo_quantile=args.tomo_quantile,
                dataset=args.dataset, dataset_eps=args.dataset_eps,
            )


if __name__ == "__main__":
    main()
