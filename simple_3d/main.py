import torch
import argparse
from pathlib import Path
import wandb

from data import load_modelnet10
from corruption import forward_channel
from em import train_estep, update_prior, log_em_step_wandb
from model import build_model, VOL_SIZE

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCSI: CryoEM 3D volume recovery from 2D projections (ModelNet10)"
    )
    parser.add_argument("--model", type=str, default="dit3d",
                        choices=["dit3d", "unet3d"],
                        help="Architecture: dit3d (Transformer3DModel) or unet3d (UNet3DConditionModel)")
    parser.add_argument("--n_em_steps", type=int, default=200)
    parser.add_argument("--epochs_per_em", type=int, default=2)
    parser.add_argument("--epochs_first_pass", type=int, default=10)
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"])
    parser.add_argument("--coupled_fraction", type=float, default=0.0)
    parser.add_argument("--small_model", action="store_true",
                        help="Use smaller model for debugging")
    parser.add_argument("--debug", action="store_true",
                        help="Tiny run: 2 EM steps, 1 epoch each")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.n_em_steps = 2
        args.epochs_per_em = 1
        args.epochs_first_pass = 1

    noise_std         = 0.3
    n_em_steps        = args.n_em_steps
    epochs_per_em     = args.epochs_per_em
    epochs_first_pass = args.epochs_first_pass
    sample_method     = "euler"
    sample_steps      = 50
    interpolant_style = args.interpolant_style
    coupled_fraction  = args.coupled_fraction
    fresh_model_every_em_step = False

    batch_size = 16   # 3D vols are 32x larger than 2D; keep activations manageable
    lr         = 3e-4

    print(f"Device: {device}")
    print(f"Architecture: {args.model}{'  [small]' if args.small_model else ''}")
    print(f"EM steps: {n_em_steps},  epochs/step: {epochs_per_em}")

    model = build_model(args.model, small=args.small_model, vol_size=VOL_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    wandb.init(
        project="scsi-cryoem-modelnet10",
        config=dict(
            arch=args.model,
            small_model=args.small_model,
            noise_std=noise_std,
            n_em_steps=n_em_steps,
            epochs_per_em=epochs_per_em,
            epochs_first_pass=epochs_first_pass,
            sample_method=sample_method,
            sample_steps=sample_steps,
            batch_size=batch_size,
            lr=lr,
            interpolant_style=interpolant_style,
            coupled_fraction=coupled_fraction,
            vol_size=VOL_SIZE,
            n_params=n_params,
        ),
    )
    global_step = [0]

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading ModelNet10 (downloads on first run) ...")
    x_gt_all = load_modelnet10(data_root="./data", split="train")
    N = x_gt_all.size(0)
    print(f"Loaded {N} volumes, shape {tuple(x_gt_all.shape)}")
    print(f"GT  range=[{x_gt_all.min():.2f}, {x_gt_all.max():.2f}]\n")

    # ── Fixed observations for M-step and logging ─────────────────────────────
    # y_obs: (N, 1, H, W) — 2D projections from ground-truth volumes
    # Note: a fresh random rotation is drawn per observation for diversity
    y_obs = forward_channel(x_gt_all, noise_std=noise_std)
    print(f"y_obs shape: {tuple(y_obs.shape)}")
    print(f"Obs range=[{y_obs.min():.2f}, {y_obs.max():.2f}]\n")

    # ── Bootstrap: π⁰ = Gaussian noise (y_obs is 2D; can't copy to 3D pool) ──
    x_pool = torch.randn(N, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE)
    z_pool = None

    run_tag = f"{args.model}{'_small' if args.small_model else ''}"
    ckpt_dir  = Path(f"checkpoints_cryoem_{run_tag}")
    prior_dir = Path(f"priors_cryoem_{run_tag}")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    # ── EM loop ───────────────────────────────────────────────────────────────
    for k in range(n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k}")
        print("=" * 60)

        torch.save(x_pool, prior_dir / f"prior_em{k:03d}.pt")

        if k == 0 or fresh_model_every_em_step:
            model = build_model(args.model, small=args.small_model, vol_size=VOL_SIZE).to(device)

        epochs = epochs_first_pass if k == 0 else epochs_per_em
        train_estep(
            model, x_pool,
            noise_std=noise_std,
            style=interpolant_style,
            epochs=epochs, batch_size=batch_size, lr=lr,
            global_step=global_step, device=device,
            z_pool=z_pool, coupled_fraction=coupled_fraction,
        )
        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:03d}.pt")
        print(f"  ✓ saved checkpoint")

        print(f"\n  M-step: sampling π({k+1}) from y_obs ...")
        x_pool, z_pool = update_prior(
            model, y_obs,
            n_steps=sample_steps,
            batch_size=batch_size * 2,
            method=sample_method,
            device=device,
        )
        log_em_step_wandb(x_gt_all, y_obs, x_pool, em_step=k)

    wandb.finish()
    print("Done.")
