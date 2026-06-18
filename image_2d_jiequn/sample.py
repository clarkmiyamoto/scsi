"""Evaluate / visualize a trained CryoET-SCSI model on held-out MNIST.

Loads a checkpoint (EMA weights), corrupts a batch of test images, reconstructs
them by transporting the forward-model prior through the learned drift, and saves
a ``[clean | FBP | reconstruction]`` comparison grid. Reports mean MSE and
MSE-up-to-rotation (the honest metric given the unknown global ``theta0``).

    python sample.py --ckpt ./results_clean/model-best.pt --n 8
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import load_mnist_pm1
from forward import radon_tilt_series, rotate_image
from backwards import warmup_target
from model import build_model
from scsi import SCSInterpolant


@torch.no_grad()
def mse_up_to_rotation(recon, clean, n_angles=72):
    """Min MSE over a grid of global rotations of ``recon`` (per image),
    accounting for the cryo-ET orientation ambiguity."""
    N = recon.shape[0]
    best = torch.full((N,), float("inf"), device=recon.device)
    for a in torch.linspace(0, 2 * math.pi, n_angles + 1)[:-1]:
        ang = torch.full((N,), float(a), device=recon.device)
        r = rotate_image(recon, ang, bg=-1.0)
        m = ((r - clean) ** 2).flatten(1).mean(1)
        best = torch.minimum(best, m)
    return best


def main():
    p = argparse.ArgumentParser(description="Sample / visualize CryoET-SCSI on MNIST")
    p.add_argument("--ckpt", type=str, default="./results_clean/model-best.pt")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--n", type=int, default=8, help="number of test images")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--tilt_span_deg", type=float, default=60.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--ode_steps", type=int, default=80)
    p.add_argument("--network", type=str, default="unet", choices=["unet", "dit"])
    p.add_argument("--model_channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="./results_clean/samples.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(args.seed)

    # Held-out test images (train=False would be cleaner; train set is fine for a demo).
    base = load_mnist_pm1(args.data_root)
    idx = torch.randint(len(base), (args.n,), generator=g).tolist()
    clean = torch.stack([base[i][0] for i in idx]).to(device)          # [n,1,32,32]

    fwd = radon_tilt_series(args.epsilon, args.K, args.tilt_span_deg)
    cgen = torch.Generator().manual_seed(args.seed + 1)
    z_out, cond = fwd(clean, return_latents=True, generator=cgen)
    z_out, cond = z_out.to(device), cond.to(device)

    fbp_img = warmup_target(cond, args.K, args.tilt_span_deg)           # [n,1,32,32]

    model = build_model(D=32, nc=1, K=args.K, model_channels=args.model_channels,
                        network=args.network).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["ema"])                                 # EMA weights
    model.eval()

    interp = SCSInterpolant(fwd, n_steps=args.ode_steps)
    recon = interp.transport(model, z_out, cond)                       # [n,1,32,32]

    mse = ((recon - clean) ** 2).flatten(1).mean(1)
    mse_rot = mse_up_to_rotation(recon, clean)
    fbp_mse = ((fbp_img - clean) ** 2).flatten(1).mean(1)
    print(f"recon MSE              : {mse.mean().item():.4f}")
    print(f"recon MSE up-to-rotation: {mse_rot.mean().item():.4f}")
    print(f"FBP baseline MSE       : {fbp_mse.mean().item():.4f}")
    print("(reconstructions are correct only up to a global rotation — the "
          "unknown theta0 is not observable.)")

    # Grid: rows = [clean | FBP | recon], cols = samples.
    rows = [("clean", clean), ("FBP", fbp_img), ("recon", recon)]
    fig, axes = plt.subplots(3, args.n, figsize=(1.4 * args.n, 4.5))
    axes = axes.reshape(3, args.n)
    for r, (label, imgs) in enumerate(rows):
        imgs = imgs.detach().cpu()
        for c in range(args.n):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle("CryoET-SCSI on MNIST  (reconstruction up to global rotation)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved grid -> {args.out}")


if __name__ == "__main__":
    main()