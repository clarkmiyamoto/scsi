import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import load_mnist_volumes_3d
from model import ConditionalVelocityCryoET3D, VOL_SIZE
from corruption import forward_channel_3d
from scsi import loss_func_joint_3d
from wandb_logging import log_train_step, log_pretrain_reconstruction_3d

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
# 3D/CryoET analogue of pretrain.py — supervised warm-start for Theta^(0), the first SCSI
# teacher, on extruded-MNIST volumes under the CryoET channel (SO(3) rotation -> 2D projection
# -> AWGN). Structurally identical to pretrain.py: ordinary stochastic-interpolant training on
# scsi.py::loss_func_joint_3d, no outer/inner loop, R drawn fresh each step, x_hat = the
# canonical (untouched) GT volume.
#
# x_hat is deliberately left canonical (not rotated) — see mnist_cryoem/CLAUDE.md for why: the
# main EM loop's M-step computes y_hat = F(x_hat; pose6_hat), which RE-APPLIES the rotation, so
# x_hat has to be canonical for that reconstruction (and this checkpoint, loaded via
# main_3d.py's --init_ckpt) to be meaningful. This is the single highest-value invariant to get
# right — it's a silent bug (no shape error) if broken, regardless of pose representation.
########################################################


########################################################
# Pluggable pool selection — same pattern as pretrain.py.
########################################################

def select_per_class(n_images_per_class: int, digit_classes: list[int] | None,
                     vol_size: int, inplane_size: int | None, depth_extent: int | None,
                     seed: int | None = None) -> torch.Tensor:
    """n_images_per_class GT volumes from EACH class in digit_classes (default: all 10)."""
    return load_mnist_volumes_3d(n_images_per_class, vol_size=vol_size,
                                 inplane_size=inplane_size, depth_extent=depth_extent,
                                 digit_classes=digit_classes, seed=seed)


SELECTION_STRATEGIES = {
    "per_class": select_per_class,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised pretraining of the first SCSI teacher Theta^(0) for the "
                    "3D/CryoET generalization: ordinary stochastic-interpolant training on a "
                    "small pool of known GT extruded-MNIST volumes, learning to flow noise to "
                    "the canonical volume conditioned on F(volume; R) for a fresh random "
                    "rotation R each step."
    )
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) the pretraining pool is drawn from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_pretrain_images_per_class", type=int, default=2,
                        help="How many GT volumes make up the supervised pool, PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--selection_strategy", type=str, default="per_class",
                        choices=list(SELECTION_STRATEGIES),
                        help="How the pretraining pool {x_i} is chosen — see "
                             "SELECTION_STRATEGIES")
    parser.add_argument("--vol_size", type=int, default=VOL_SIZE,
                        help="p: side length of the R^{p^3} volume cube")
    parser.add_argument("--inplane_size", type=int, default=None,
                        help="MNIST digit is loaded at THIS resolution before zero-padding up "
                             "to --vol_size in H,W (default: round(vol_size*0.65))")
    parser.add_argument("--depth_extent", type=int, default=None,
                        help="Thickness (voxels) of the depth band the digit is extruded "
                             "across, centered in D (default: round(vol_size*0.55))")
    parser.add_argument("--noise_std", type=float, default=0.5,
                        help="AWGN std for the 2D-image observation (different scale than the "
                             "2D pipeline's 1D-projection channel — don't reuse that default)")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Flat supervised SGD steps (no outer/inner loop)")
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"])
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save a safety checkpoint every N steps (final state is always "
                             "saved too)")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Log a qualitative reconstruction panel to wandb every N steps "
                             "(0 disables it; the final step always logs one)")
    parser.add_argument("--sample_steps", type=int, default=20,
                        help="Euler steps for the joint ODE used only by --plot_every's "
                             "reconstruction panel (not used in training itself)")
    parser.add_argument("--out_ckpt", type=str,
                        default="mnist_cryoet_checkpoints/pretrain_theta0.pt")
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
        if not _explicit("--vol_size"):                    args.vol_size = 16
        if not _explicit("--n_steps"):                    args.n_steps = 8
        if not _explicit("--batch_size"):                 args.batch_size = 4
        if not _explicit("--checkpoint_every"):            args.checkpoint_every = 4
        if not _explicit("--plot_every"):                  args.plot_every = 4
        if not _explicit("--sample_steps"):                args.sample_steps = 5

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    print(f"Device: {device}")

    # ── Pretraining pool ─────────────────────────────────────────────────
    select_fn = SELECTION_STRATEGIES[args.selection_strategy]
    volume_pool = select_fn(args.n_pretrain_images_per_class, args.digit_classes,
                            args.vol_size, args.inplane_size, args.depth_extent).to(device)
    print(f"Pretrain pool: {volume_pool.size(0)} volumes "
          f"({args.n_pretrain_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))}, "
          f"strategy={args.selection_strategy}, vol_size={args.vol_size})")

    # ── Model / optimizer ────────────────────────────────────────────────
    model = ConditionalVelocityCryoET3D(vol_size=args.vol_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    if use_wandb:
        wandb.init(
            project="scsi-cryoet-mnist3d-pretrain",
            config=vars(args) | {"n_params": n_params, "pool_size": volume_pool.size(0)},
        )

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── Flat supervised stochastic-interpolant training ─────────────────
    N = volume_pool.size(0)
    model.train()
    running_img, running_pose = 0.0, 0.0

    for step in tqdm(range(args.n_steps), desc="pretrain"):
        idx = torch.randint(0, N, (args.batch_size,))
        x_batch = volume_pool[idx]                       # canonical GT, untouched
        y_batch, pose_batch = forward_channel_3d(x_batch, noise_std=args.noise_std)

        loss, loss_img, loss_pose = loss_func_joint_3d(
            model, x_batch, pose_batch, y_batch,
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
            log_pretrain_reconstruction_3d(
                model, volume_pool, noise_std=args.noise_std,
                sample_steps=args.sample_steps, step=step, use_wandb=use_wandb,
            )

    if args.plot_every > 0 and args.n_steps % args.plot_every != 0:
        log_pretrain_reconstruction_3d(
            model, volume_pool, noise_std=args.noise_std,
            sample_steps=args.sample_steps, step=args.n_steps - 1, use_wandb=use_wandb,
        )

    torch.save(model.state_dict(), out_ckpt)
    print(f"steps={args.n_steps}  loss_image={running_img / args.n_steps:.5f}"
          f"  loss_pose={running_pose / args.n_steps:.5f}")
    print(f"Theta^(0) saved -> {out_ckpt}")

    if use_wandb:
        wandb.finish()
    print("Done.")
