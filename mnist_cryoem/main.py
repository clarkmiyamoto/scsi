import sys
import torch
import argparse
from pathlib import Path

from data import load_mnist_subset, build_observations
from em import run_em_loop
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
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) to draw GT digits from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_images_per_class", type=int, default=2,
                        help="Number of ground-truth MNIST digits to use PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="Path to a pretrained state_dict (e.g. from pretrain.py) to "
                             "load as Theta^(0) before the EM loop starts")
    parser.add_argument("--corruptions_per_image", type=int, default=200,
                        help="Number of independent corrupted observations per GT digit "
                             "(the channel is applied this many times per photo)")
    parser.add_argument("--noise_std", type=float, default=1.0)
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
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"],
                        help="Image branch only — the pose branch always uses a geodesic "
                             "(constant angular velocity) schedule")
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="Euler steps for the joint ODE (Phi) in the E-step")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Tiny run: quick smoke test")
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
        if not _explicit("--corruptions_per_image"): args.corruptions_per_image = 2
        if not _explicit("--n_em_steps"):             args.n_em_steps = 2
        if not _explicit("--steps_per_em"):           args.steps_per_em = 4
        if not _explicit("--steps_first_em"):         args.steps_first_em = 4
        if not _explicit("--sample_steps"):           args.sample_steps = 5
        if not _explicit("--batch_size"):             args.batch_size = 8

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
    x_gt = load_mnist_subset(args.n_images_per_class, digit_classes=args.digit_classes)
    y_obs, theta_star, image_idx = build_observations(
        x_gt, corruptions_per_image=args.corruptions_per_image, noise_std=args.noise_std,
    )
    N_obs = y_obs.size(0)
    print(f"GT digits: {x_gt.size(0)} ({args.n_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))})   observations: {N_obs} "
          f"({args.corruptions_per_image} per digit)")
    print(f"GT  range=[{x_gt.min():.2f}, {x_gt.max():.2f}]")
    print(f"Obs range=[{y_obs.min():.2f}, {y_obs.max():.2f}]\n")

    # ── Model / optimizer ────────────────────────────────────────────────
    # The optimizer is created once and persists across every EM outer iteration (see
    # scsi.py::train_mstep docstring) — required for --steps_per_em as low as 1 to be a fair
    # test of the literal pseudocode rather than being crippled by constantly-reset Adam state.
    model = ConditionalVelocityCryoEM(image_size=IMAGE_SIZE).to(device)
    if args.init_ckpt is not None:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        print(f"Loaded initial teacher weights <- {args.init_ckpt}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    if use_wandb:
        wandb.init(
            project="scsi-cryoem-mnist",
            config=vars(args) | {
                "n_params": n_params, "N_obs": N_obs, "steps_first_em": steps_first_em,
            },
        )

    ckpt_dir = Path("mnist_cryoem_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    run_em_loop(
        model, opt, y_obs, x_gt, theta_star, image_idx,
        args=args, device=device, use_wandb=use_wandb, ckpt_dir=ckpt_dir,
    )

    if use_wandb:
        wandb.finish()
    print("Done.")
