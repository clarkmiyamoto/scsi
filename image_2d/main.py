import torch
import argparse
from pathlib import Path
import wandb

from data import load_mnist
from corruption import forward_channel
from em import train_estep, update_prior, log_em_step_wandb
from model import ConditionalDiT

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="SCSI algorithm on 2D image datasets")
    parser.add_argument(
        "--corruption",
        type=str,
        default="awgn",
        choices=["awgn", "mra"],
        help="Forward channel: awgn (Y=X+noise) or mra (random 2-D shift + noise)",
    )
    parser.add_argument(
        "--n_em_steps",
        type=int,
        default=200,
        help="Number of EM steps",
    )
    parser.add_argument(
        "--epochs_per_em",
        type=int,
        default=2,
        help="Number of epochs for each EM step",
    )
    parser.add_argument(
        "--epochs_first_pass",
        type=int,
        default=10,
        help="Number of epochs for the first pass of the E-step",
    )
    parser.add_argument(
        "--interpolant_style",
        type=str,
        default="linear",
        choices=["linear", "gvp"],
        help="Interpolant style: linear or gvp",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.n_em_steps = 1
        args.epochs_per_em = 1
        args.epochs_first_pass = 1

    # Problem parameter
    corruption   = args.corruption # "awgn", "mra", "drop", "drop_mra"
    noise_std    = 0.3 # Noise standard deviation
    p_drop       = 0.1 # Probability of removing a pixel from the image (0.0 to 1.0)
    n_obs        = 10_000 # Number of observations, instead of full dataset

    # SCSI parameters
    n_em_steps   = args.n_em_steps
    epochs_per_em = args.epochs_per_em
    epochs_first_pass = args.epochs_first_pass
    sample_method = "euler" # "euler" or "midpoint"
    sample_steps = 50
    interpolant_style = args.interpolant_style        # interpolant style: "linear" or "gvp"
    fresh_model_every_em_step = False

    # Training parameters
    batch_size   = 256
    lr           = 3e-4

    print(f"Device: {device}")
    print(f"Channel: {corruption},  noise_std={noise_std}")
    print(f"EM steps: {n_em_steps},  epochs/step: {epochs_per_em}\n")

    wandb.init(
        project="scsi-mnist",
        config=dict(
            corruption=corruption,
            noise_std=noise_std,
            n_obs=n_obs,
            n_em_steps=n_em_steps,
            epochs_per_em=epochs_per_em,
            epochs_first_pass=epochs_first_pass,
            sample_method=sample_method,
            sample_steps=sample_steps,
            batch_size=batch_size,
            lr=lr,
            interpolant_style=interpolant_style,
            fresh_model_every_em_step=fresh_model_every_em_step,
        ),
    )
    global_step = [0]

    # ── Load dataset ──────────────────────────────────────────────────
    x_gt_all = load_mnist(n_obs)
    y_obs = forward_channel(x_gt_all, noise_std=noise_std, p_drop=p_drop, corruption=corruption)
    print(f"GT  range=[{x_gt_all.min():.2f}, {x_gt_all.max():.2f}]")
    print(f"Obs range=[{y_obs.min():.2f}, {y_obs.max():.2f}]\n")

    # ── Bootstrap: pi^(0) = Y_obs ────────────────────────────────────
    x_pool = y_obs.clone()
    prior_history = [x_pool[:8].clone()]

    ckpt_dir = Path(f"checkpoints_{corruption}_mnist_transformer")
    prior_dir = Path(f"priors_{corruption}_mnist_transformer")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    # ── EM loop ───────────────────────────────────────────────────────
    for k in range(n_em_steps):
        print("=" * 60)
        print(f"EM iteration {k}")
        print("=" * 60)

        # Save prior for diagnostics
        torch.save(x_pool, prior_dir / f"prior_em{k:02d}.pt")

        # Fresh model each EM step (following SCSI paper)
        if k==0 or fresh_model_every_em_step:
            model = ConditionalDiT().to(device)
            print(f"Using fresh model")
        else:
            model = model
        
        if k == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}\n")

        # E-step
        epochs = epochs_first_pass if k == 0 else epochs_per_em
        train_estep(model, x_pool, noise_std=noise_std, p_drop=p_drop, corruption=corruption,
                    style=interpolant_style, epochs=epochs, batch_size=batch_size, lr=lr,
                    global_step=global_step, device=device)
        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:02d}.pt")
        print(f"  ✓ saved checkpoint")

        # M-step
        print(f"\n  M-step: sampling π({k+1}) ...")
        x_pool, initial_state = update_prior(model, y_obs, n_steps=sample_steps,
                              batch_size=batch_size*3, method=sample_method,
                              device=device)
        prior_history.append(x_pool[:8].clone())
        log_em_step_wandb(x_gt_all, y_obs, x_pool, em_step=k)

    wandb.finish()
    print("Done.")