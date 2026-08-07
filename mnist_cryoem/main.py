import sys
import torch
import argparse
from pathlib import Path

from data import load_mnist_subset, build_observations
from corruption import forward_channel, sample_uniform_angle
from em import run_em_loop
from overfit import overfit_single_batch
from warmup import run_classical_recon_warmup
from model import ConditionalVelocityCryoEM, IMAGE_SIZE

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCSI on a CryoEM-style 2D->1D MNIST channel "
                    "(random in-plane rotation + 1D projection + AWGN)"
    )
    # Dataset / channel
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) to draw GT digits from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_images_per_class", type=int, default=2,
                        help="Number of ground-truth MNIST digits to use PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--corruptions_per_object", type=int, default=200,
                        help="Number of independent tilt-series ACQUISITIONS per GT object "
                             "(F_CryoET is applied this many times per object; each "
                             "application draws its own random tilt-series start offset)")
    parser.add_argument("--n_tilts", type=int, default=16,
                        help="T: number of tilts (1D projections) per acquisition's tilt "
                             "series, at angles evenly spaced by --tilt_increment_deg from "
                             "that acquisition's random start offset. Default 1 reproduces "
                             "the old single-random-rotation-per-observation behavior.")
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5,
                        help="Degrees between consecutive tilts within one acquisition's "
                             "series (unused when --n_tilts=1)")
    parser.add_argument("--noise_std", type=float, default=3.0)
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "test"],
                        help="Which MNIST split to draw GT digits from. Default 'test' preserves "
                             "the disjointness guarantee vs. pretrain.py's hardcoded train split "
                             "(see mnist_cryoem/CLAUDE.md's dataset-split gotcha) -- change this "
                             "only when there's no separate pretrain.py pool to stay disjoint "
                             "from, e.g. an in-process --warmup_steps run with no --init_ckpt, or "
                             "when the test split's smaller per-class count (892-1135) is itself "
                             "the limiting factor for --n_images_per_class. CAUTION: combining "
                             "--dataset_split train with --init_ckpt pointing at a pretrain.py "
                             "checkpoint (which always trains on the train split) reintroduces "
                             "exactly the overlap that guarantee exists to prevent.")

    # EM Loop
    parser.add_argument("--init_ckpt", type=str, default=None,
                            help="Path to a pretrained state_dict (e.g. from pretrain.py) to "
                                 "load as Theta^(0) before the EM loop starts")
    parser.add_argument("--ckpt_dir", type=str, default="mnist_cryoem_checkpoints",
                        help="Directory EM-loop checkpoints are written to")
    parser.add_argument("--n_em_steps", type=int, default=200,
                        help="K: number of outer EM iterations")
    parser.add_argument("--steps_per_em", type=int, default=200,
                        help="T_tr: M-step SGD steps taken per Phi^(k-1) pool generation "
                             "before the next E-step refreshes it. "
                             "1 = literal pseudocode (a completely fresh generation every "
                             "SGD step); >1 amortizes the ODE solve over more gradient steps.")
    parser.add_argument("--steps_first_em", type=int, default=None,
                        help="Override --steps_per_em for EM iteration 0 only "
                             "(default: same as --steps_per_em)")
    parser.add_argument("--sample_steps", type=int, default=50,
                            help="Euler steps for the joint ODE (Phi) in the E-step")

    # Warmup (--warmup_steps > 0): supervised stochastic-interpolant warm-start on classical
    # (filtered-backprojection) reconstructions of THIS run's own D_Y (y_obs/theta_star built
    # above), run in-process right before the EM loop -- see warmup.py::run_classical_recon_warmup.
    # Unlike pretrain.py's separate --warmstart_target classical_recon script, there's no separate
    # pool geometry here: --corruptions_per_object/--n_tilts/--tilt_increment_deg/--noise_std
    # above size BOTH this pool and the EM loop's pool, since they're the same D_Y.
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Supervised SGD steps on a classical-recon reconstruction of this "
                             "run's own D_Y, run once before the EM loop starts. 0 (default) "
                             "disables the warmup phase entirely.")
    parser.add_argument("--warmup_recon_filter_type", type=str, default="hann",
                        choices=["hann", "ramp"],
                        help="Only used when --warmup_steps > 0. Passed to "
                             "classical_recon.backproject's ramp filter.")
    parser.add_argument("--warmup_recon_calibration", type=str, default="affine_clamp",
                        choices=["affine_clamp", "none"],
                        help="Only used when --warmup_steps > 0. 'affine_clamp' (default): "
                             "rescale the whole reconstruction pool to the GT pool's mean/std "
                             "(raw backprojection output is not on MNIST's [-1,1] scale), then "
                             "clamp to [-1,1]. 'none': use raw backproject() output as-is.")
    parser.add_argument("--warmup_checkpoint_every", type=int, default=100,
                        help="Only used when --warmup_steps > 0. Save a safety checkpoint "
                             "(model_warmup.pt) every N warmup steps.")
    parser.add_argument("--warmup_plot_every", type=int, default=100,
                        help="Only used when --warmup_steps > 0. Log a qualitative "
                             "reconstruction panel to wandb every N warmup steps (0 disables "
                             "it; the final warmup step always logs one).")

    # Model
    parser.add_argument("--arch", type=str, default="dit", choices=["dit", "unet"],
                        help="Image branch architecture: DiTTransformer2DModel or "
                             "UNet2DModel (both from diffusers)")

    # Stochastic Interpolant / Training
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"],
                        help="Image branch only — the pose branch always uses a geodesic "
                             "(constant angular velocity) schedule")
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)

    # Etc.
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Tiny run: quick smoke test")
    parser.add_argument("--overfit", action="store_true",
                        help="Sanity check only: verify the model can overfit a single fixed "
                             "batch (see overfit.py), log loss/grad_norm to wandb, then exit "
                             "WITHOUT running the EM loop.")
    parser.add_argument("--overfit_steps", type=int, default=1000,
                        help="SGD steps for the --overfit check")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        # Only fill in tiny defaults for flags the user didn't explicitly pass — --debug
        # combined with an explicit --steps_per_em (e.g. to smoke-test literal mode on a
        # tiny dataset) must not silently override that choice.
        def _explicit(flag):
            return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

        if not _explicit("--digit_classes"):          args.digit_classes = [0]
        if not _explicit("--n_images_per_class"):     args.n_images_per_class = 16
        if not _explicit("--corruptions_per_object"): args.corruptions_per_object = 2
        if not _explicit("--n_em_steps"):             args.n_em_steps = 2
        if not _explicit("--steps_per_em"):           args.steps_per_em = 4
        if not _explicit("--steps_first_em"):         args.steps_first_em = 4
        if not _explicit("--sample_steps"):           args.sample_steps = 5
        if not _explicit("--batch_size"):             args.batch_size = 8
        if not _explicit("--overfit_steps"):          args.overfit_steps = 8
        # --warmup_steps itself is left alone (stays 0 unless the user opts in) -- --debug
        # should not silently turn the warmup phase on. These just keep it fast if they did.
        if not _explicit("--warmup_checkpoint_every"): args.warmup_checkpoint_every = 2
        if not _explicit("--warmup_plot_every"):       args.warmup_plot_every = 2

    steps_first_em = args.steps_first_em if args.steps_first_em is not None else args.steps_per_em
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb

    print(f"Device: {device}")
    if args.steps_per_em == 1:
        print("Mode: LITERAL pseudocode (--steps_per_em=1) — a completely fresh "
              "Phi^(k-1)-generated (x_hat, R_hat) batch every single SGD step")
    else:
        print(f"Mode: amortized (--steps_per_em={args.steps_per_em}) — one Phi^(k-1) pool "
              f"generation is reused across {args.steps_per_em} SGD steps before refreshing")

    # ── Load dataset ─────────────────────────────────────────────────────
    x_gt = load_mnist_subset(args.n_images_per_class, digit_classes=args.digit_classes,
                             train=(args.dataset_split == "train"))
    y_obs, theta_star, image_idx, acq_idx = build_observations(
        x_gt, corruptions_per_object=args.corruptions_per_object,
        n_tilts=args.n_tilts, tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std,
    )
    N_obs = y_obs.size(0)
    print(f"GT digits: {x_gt.size(0)} ({args.n_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))}, split={args.dataset_split})   "
          f"observations: {N_obs} "
          f"({args.corruptions_per_object} acquisitions x {args.n_tilts} tilts per digit)")
    print(f"GT  range=[{x_gt.min():.2f}, {x_gt.max():.2f}]")
    print(f"Obs range=[{y_obs.min():.2f}, {y_obs.max():.2f}]\n")

    # ── Model / optimizer ────────────────────────────────────────────────
    # The optimizer is created once and persists across every EM outer iteration (see
    # scsi.py::train_mstep docstring) — required for --steps_per_em as low as 1 to be a fair
    # test of the literal pseudocode rather than being crippled by constantly-reset Adam state.
    model = ConditionalVelocityCryoEM(image_size=IMAGE_SIZE, arch=args.arch).to(device)
    if args.init_ckpt is not None:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        print(f"Loaded initial teacher weights <- {args.init_ckpt}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    # ── --overfit: sanity check only, exits before the EM loop ──────────────
    if args.overfit:
        n_batch = min(args.batch_size, x_gt.size(0))
        x_batch = x_gt[:n_batch].to(device)
        theta_batch = sample_uniform_angle(n_batch, device)
        y_batch, _ = forward_channel(x_batch, noise_std=args.noise_std, theta=theta_batch)
        print(f"Overfit check: batch_size={n_batch}  steps={args.overfit_steps}")

        if use_wandb:
            wandb.init(
                project="scsi-cryoem-mnist-overfit",
                config=vars(args) | {"n_params": n_params, "batch_size": n_batch,
                                     "dataset_split": args.dataset_split},
            )

        final_loss = overfit_single_batch(
            model, x_batch, theta_batch, y_batch,
            n_steps=args.overfit_steps, lr=args.lr,
            style=args.interpolant_style, pose_loss_weight=args.pose_loss_weight,
            use_wandb=use_wandb,
        )
        print(f"Overfit check done: final loss={final_loss:.5f}")

        if use_wandb:
            wandb.finish()
        sys.exit(0)

    if use_wandb:
        wandb.init(
            project="scsi-cryoem-mnist",
            config=vars(args) | {
                "n_params": n_params, "N_obs": N_obs, "steps_first_em": steps_first_em,
                "dataset_split": args.dataset_split,
            },
        )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ── Warmup (--warmup_steps > 0): supervised classical-recon warm-start on this run's own
    # D_Y, sharing this same wandb run with the EM loop below (see warmup.py's docstring).
    # Composable with --init_ckpt: loading a checkpoint and then further warming it up on this
    # run's own data is intentional, not an error.
    warmup_global_step = 0
    if args.warmup_steps > 0:
        print(f"Warmup: {args.warmup_steps} classical-recon-supervised steps on this run's own "
              f"D_Y ({N_obs} observations, {args.corruptions_per_object} acquisitions/object)")
        warmup_global_step = run_classical_recon_warmup(
            model, opt, x_gt, y_obs, theta_star, image_idx, acq_idx,
            args=args, device=device, use_wandb=use_wandb, ckpt_dir=ckpt_dir,
        )

    run_em_loop(
        model, opt, y_obs, x_gt, theta_star, image_idx, acq_idx,
        args=args, device=device, use_wandb=use_wandb, ckpt_dir=ckpt_dir,
        global_step_start=warmup_global_step,
    )

    if use_wandb:
        wandb.finish()
    print("Done.")
