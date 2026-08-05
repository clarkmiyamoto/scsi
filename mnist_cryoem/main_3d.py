import sys
import torch
import argparse
from pathlib import Path

from data import load_mnist_volumes_3d, build_observations_3d
from corruption import forward_channel_3d
from em import run_em_loop_3d
from overfit import overfit_single_batch
from scsi import loss_func_joint_3d
from model import ConditionalVelocityCryoET3D, VOL_SIZE

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
        description="SCSI on a CryoET-style 3D->2D extruded-MNIST channel "
                    "(SO(3) rotation, mount+fixed-axis tilt series, + 2D projection + AWGN)"
    )
    # Dataset / channel
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) to draw GT digits from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_images_per_class", type=int, default=2,
                        help="Number of ground-truth MNIST digits to use PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--vol_size", type=int, default=VOL_SIZE,
                        help="p: side length of the R^{p^3} volume cube")
    parser.add_argument("--inplane_size", type=int, default=None,
                        help="MNIST digit is loaded at THIS resolution before zero-padding up "
                             "to --vol_size in H,W (default: round(vol_size*0.65))")
    parser.add_argument("--depth_extent", type=int, default=None,
                        help="Thickness (voxels) of the depth band the digit is extruded "
                             "across, centered in D (default: round(vol_size*0.55))")
    parser.add_argument("--corruptions_per_object", type=int, default=50,
                        help="Number of independent tilt-series ACQUISITIONS per GT object "
                             "(F_CryoET is applied this many times per object; each "
                             "application draws its own random mount orientation + tilt-series "
                             "start offset)")
    parser.add_argument("--n_tilts", type=int, default=16,
                        help="T: number of tilts (2D projections) per acquisition's tilt "
                             "series, at angles evenly spaced by --tilt_increment_deg about "
                             "the FIXED --tilt_axis, from that acquisition's random start "
                             "offset.")
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5,
                        help="Degrees between consecutive tilts within one acquisition's "
                             "series")
    parser.add_argument("--tilt_axis", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                        help="The FIXED physical tilt axis, in lab-frame coordinates "
                             "(3 floats, need not be normalized)")
    parser.add_argument("--noise_std", type=float, default=0.5,
                        help="AWGN std for the 2D-image observation (different scale than the "
                             "2D pipeline's 1D-projection channel — don't reuse that default)")

    # EM Loop
    parser.add_argument("--init_ckpt", type=str, default=None,
                            help="Path to a pretrained state_dict (e.g. from pretrain_3d.py) "
                                 "to load as Theta^(0) before the EM loop starts")
    parser.add_argument("--n_em_steps", type=int, default=200,
                        help="K: number of outer EM iterations")
    parser.add_argument("--steps_per_em", type=int, default=200,
                        help="T_tr: M-step SGD steps taken per Phi^(k-1) pool generation "
                             "before the next E-step refreshes it. NOTE: unlike the 2D "
                             "pipeline, --steps_per_em 1 (literal pseudocode, a fresh "
                             "Phi^(k-1) draw every SGD step) is computationally IMPRACTICAL "
                             "at 3D scale — the E-step's cost is pool_size * sample_steps "
                             "full 3D UNet passes per pool refresh. Amortized mode "
                             "(tens-to-hundreds) is the only practical regime here.")
    parser.add_argument("--steps_first_em", type=int, default=None,
                        help="Override --steps_per_em for EM iteration 0 only "
                             "(default: same as --steps_per_em)")
    parser.add_argument("--sample_steps", type=int, default=20,
                            help="Euler steps for the joint ODE (Phi) in the E-step")

    # Model
    parser.add_argument("--arch", type=str, default="unet3d", choices=["unet3d"],
                        help="Volume branch architecture (only diffusers' UNet3DConditionModel "
                             "is implemented — a DiT3D analogue is dead code elsewhere in this "
                             "repo, not resurrected here)")

    # Stochastic Interpolant / Training
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"],
                        help="Shared by BOTH branches — the pose branch is the flat 6D "
                             "representation (si.py), so it reuses si.interpolant directly, "
                             "unlike the 2D pipeline's SO(2) pose branch which always uses a "
                             "separate geodesic schedule regardless of this flag")
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
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
        if not _explicit("--n_images_per_class"):     args.n_images_per_class = 4
        if not _explicit("--vol_size"):               args.vol_size = 16
        if not _explicit("--corruptions_per_object"): args.corruptions_per_object = 2
        if not _explicit("--n_tilts"):                args.n_tilts = 4
        if not _explicit("--n_em_steps"):             args.n_em_steps = 2
        if not _explicit("--steps_per_em"):           args.steps_per_em = 2
        if not _explicit("--steps_first_em"):         args.steps_first_em = 2
        if not _explicit("--sample_steps"):           args.sample_steps = 3
        if not _explicit("--batch_size"):             args.batch_size = 2
        if not _explicit("--overfit_steps"):          args.overfit_steps = 4

    steps_first_em = args.steps_first_em if args.steps_first_em is not None else args.steps_per_em
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb

    print(f"Device: {device}")
    if args.steps_per_em == 1:
        print("WARNING: --steps_per_em=1 (literal pseudocode) is computationally impractical "
              "at 3D scale — see --help. Proceeding anyway since it was explicitly requested.")
    print(f"Mode: amortized (--steps_per_em={args.steps_per_em}) — one Phi^(k-1) pool "
          f"generation is reused across {args.steps_per_em} SGD steps before refreshing")

    # ── Load dataset ─────────────────────────────────────────────────────
    x_gt = load_mnist_volumes_3d(args.n_images_per_class, vol_size=args.vol_size,
                                 inplane_size=args.inplane_size, depth_extent=args.depth_extent,
                                 digit_classes=args.digit_classes, train=False)
    y_obs, R_star, image_idx, acq_idx = build_observations_3d(
        x_gt, corruptions_per_object=args.corruptions_per_object,
        n_tilts=args.n_tilts, tilt_increment_deg=args.tilt_increment_deg,
        noise_std=args.noise_std, tilt_axis=tuple(args.tilt_axis),
    )
    N_obs = y_obs.size(0)
    print(f"GT volumes: {x_gt.size(0)} ({args.n_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))}, vol_size={args.vol_size}, "
          f"split=test)   observations: {N_obs} "
          f"({args.corruptions_per_object} acquisitions x {args.n_tilts} tilts per volume)")
    print(f"GT  range=[{x_gt.min():.2f}, {x_gt.max():.2f}]")
    print(f"Obs range=[{y_obs.min():.2f}, {y_obs.max():.2f}]\n")

    # ── Model / optimizer ────────────────────────────────────────────────
    # The optimizer is created once and persists across every EM outer iteration (see
    # scsi.py::train_mstep_3d docstring) — required for --steps_per_em as low as 1 to be a fair
    # test of the literal pseudocode rather than being crippled by constantly-reset Adam state.
    model = ConditionalVelocityCryoET3D(vol_size=args.vol_size, arch=args.arch).to(device)
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
        y_batch, pose_batch = forward_channel_3d(x_batch, noise_std=args.noise_std)
        print(f"Overfit check: batch_size={n_batch}  steps={args.overfit_steps}")

        if use_wandb:
            wandb.init(
                project="scsi-cryoet-mnist3d-overfit",
                config=vars(args) | {"n_params": n_params, "batch_size": n_batch,
                                     "dataset_split": "test"},
            )

        final_loss = overfit_single_batch(
            model, x_batch, pose_batch, y_batch,
            n_steps=args.overfit_steps, lr=args.lr,
            style=args.interpolant_style, pose_loss_weight=args.pose_loss_weight,
            use_wandb=use_wandb, loss_fn=loss_func_joint_3d,
        )
        print(f"Overfit check done: final loss={final_loss:.5f}")

        if use_wandb:
            wandb.finish()
        sys.exit(0)

    if use_wandb:
        wandb.init(
            project="scsi-cryoet-mnist3d",
            config=vars(args) | {
                "n_params": n_params, "N_obs": N_obs, "steps_first_em": steps_first_em,
                "dataset_split": "test",
            },
        )

    ckpt_dir = Path("mnist_cryoet_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    run_em_loop_3d(
        model, opt, y_obs, x_gt, R_star, image_idx, acq_idx,
        args=args, device=device, use_wandb=use_wandb, ckpt_dir=ckpt_dir,
    )

    if use_wandb:
        wandb.finish()
    print("Done.")
