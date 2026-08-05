import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import load_mnist_subset_mra
from model import ConditionalVelocityMRA, IMAGE_SIZE
from corruption import forward_channel_mra, sample_uniform_angle
from scsi import loss_func_joint
from wandb_logging import log_train_step, log_pretrain_reconstruction_mra

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

########################################################
# Supervised warm-start for Theta^(0) -- rotational-MRA analogue of pretrain.py. Same flat-SGD
# structure: ordinary stochastic-interpolant training on scsi.py::loss_func_joint (reused
# UNMODIFIED — it never hardcodes a channel call or inspects y's shape, see
# mnist_cryoem/CLAUDE.md), just with corruption.forward_channel_mra building y instead of
# corruption.forward_channel, and the GT pool loaded via data.load_mnist_subset_mra
# (disk-masked — REQUIRED, see corruption.mask_to_disk's docstring / the MRA gotcha in
# mnist_cryoem/CLAUDE.md) instead of data.load_mnist_subset.
#
# x_hat is deliberately left canonical (not rotated) — same canonical-target invariant as
# pretrain.py: the main EM loop's M-step computes y_hat = F_MRA(x_hat; R_hat), which
# RE-APPLIES the rotation, so x_hat has to be canonical for that reconstruction (and this
# checkpoint, loaded via main_mra_rotation.py's --init_ckpt) to be meaningful.
########################################################


########################################################
# Pluggable pool selection — add a new strategy by writing a function with this signature
# and registering it in SELECTION_STRATEGIES; nothing else in this file needs to change.
########################################################

def select_per_class(n_images_per_class: int, digit_classes: list[int] | None,
                     seed: int | None = None) -> torch.Tensor:
    """n_images_per_class GT digits from EACH class in digit_classes (default: all 10),
    disk-masked (data.load_mnist_subset_mra)."""
    return load_mnist_subset_mra(n_images_per_class, digit_classes=digit_classes, seed=seed)


SELECTION_STRATEGIES = {
    "per_class": select_per_class,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised pretraining of the first SCSI teacher Theta^(0) for the "
                    "rotational-MRA channel: ordinary stochastic-interpolant training on a "
                    "small pool of known GT MNIST digits, learning to flow noise to the "
                    "canonical digit conditioned on F_MRA(digit; R) = R∘digit + W for a "
                    "fresh random rotation R each step."
    )
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) the pretraining pool is drawn from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_pretrain_images_per_class", type=int, default=2,
                        help="How many GT images make up the supervised pool, PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--selection_strategy", type=str, default="per_class",
                        choices=list(SELECTION_STRATEGIES),
                        help="How the pretraining pool {x_i} is chosen — see "
                             "SELECTION_STRATEGIES")
    parser.add_argument("--noise_std", type=float, default=0.3,
                        help="AWGN std, added directly to the full [-1,1]-range rotated image "
                             "— matches main_mra_rotation.py's default (0.3), NOT "
                             "pretrain.py's 1.0 (calibrated for the 1D-projected channel).")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Flat supervised SGD steps (no outer/inner loop)")
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"])
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save a safety checkpoint every N steps (final state is always "
                             "saved too)")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Log a qualitative reconstruction panel to wandb every N steps "
                             "(0 disables it; the final step always logs one)")
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="Euler steps for the joint ODE used only by --plot_every's "
                             "reconstruction panel (not used in training itself)")
    parser.add_argument("--out_ckpt", type=str,
                        default="mnist_mra_rotation_checkpoints/pretrain_theta0.pt")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Tiny run: quick smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        def _explicit(flag):
            return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

        if not _explicit("--digit_classes"):              args.digit_classes = [0]
        if not _explicit("--n_pretrain_images_per_class"): args.n_pretrain_images_per_class = 2
        if not _explicit("--n_steps"):                    args.n_steps = 8
        if not _explicit("--batch_size"):                 args.batch_size = 8
        if not _explicit("--checkpoint_every"):            args.checkpoint_every = 4
        if not _explicit("--plot_every"):                  args.plot_every = 4
        if not _explicit("--sample_steps"):                args.sample_steps = 5

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    print(f"Device: {device}")

    # ── Pretraining pool ─────────────────────────────────────────────────
    select_fn = SELECTION_STRATEGIES[args.selection_strategy]
    image_pool = select_fn(args.n_pretrain_images_per_class, args.digit_classes).to(device)
    print(f"Pretrain pool: {image_pool.size(0)} images "
          f"({args.n_pretrain_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))}, "
          f"strategy={args.selection_strategy})")

    # ── Model / optimizer ────────────────────────────────────────────────
    model = ConditionalVelocityMRA(image_size=IMAGE_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    if use_wandb:
        wandb.init(
            project="scsi-mra-rotation-mnist-pretrain",
            config=vars(args) | {"n_params": n_params, "pool_size": image_pool.size(0)},
        )

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── Flat supervised stochastic-interpolant training ─────────────────
    N = image_pool.size(0)
    model.train()
    running_img, running_pose = 0.0, 0.0

    for step in tqdm(range(args.n_steps), desc="pretrain"):
        idx = torch.randint(0, N, (args.batch_size,))
        x_batch = image_pool[idx]                       # canonical GT, untouched
        theta_batch = sample_uniform_angle(args.batch_size, device)
        y_batch, _ = forward_channel_mra(x_batch, noise_std=args.noise_std, theta=theta_batch)

        loss, loss_img, loss_pose = loss_func_joint(
            model, x_batch, theta_batch, y_batch,
            style=args.interpolant_style, pose_loss_weight=args.pose_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_img += loss_img.item()
        running_pose += loss_pose.item()
        log_train_step(loss_img, loss_pose, grad_norm, step, use_wandb)

        if (step + 1) % args.checkpoint_every == 0:
            torch.save(model.state_dict(), out_ckpt)

        if args.plot_every > 0 and (step + 1) % args.plot_every == 0:
            log_pretrain_reconstruction_mra(
                model, image_pool, noise_std=args.noise_std,
                sample_steps=args.sample_steps, step=step, use_wandb=use_wandb,
            )

    if args.plot_every > 0 and args.n_steps % args.plot_every != 0:
        log_pretrain_reconstruction_mra(
            model, image_pool, noise_std=args.noise_std,
            sample_steps=args.sample_steps, step=args.n_steps - 1, use_wandb=use_wandb,
        )

    torch.save(model.state_dict(), out_ckpt)
    print(f"steps={args.n_steps}  loss_image={running_img / args.n_steps:.5f}"
          f"  loss_pose={running_pose / args.n_steps:.5f}")
    print(f"Theta^(0) saved -> {out_ckpt}")

    if use_wandb:
        wandb.finish()
    print("Done.")
