import torch
import argparse
from pathlib import Path
import wandb

from data import load_mnist
from corruption import forward_channel
from em import train_estep_ell, update_prior_ell, log_em_step_wandb, log_curriculum_step_wandb
from model import ConditionalDiTWithEll

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        description="SCSI with curriculum warmup: progressively harder F_ell from identity to F"
    )
    parser.add_argument("--corruption", type=str, default="mra",
                        choices=["awgn", "mra", "drop_mra"])
    parser.add_argument("--n_em_steps", type=int, default=200)
    parser.add_argument("--epochs_per_em", type=int, default=2)
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"])
    parser.add_argument("--coupled_fraction", type=float, default=0.0)
    # Curriculum args
    parser.add_argument("--n_ell_levels", type=int, default=5,
                        help="Number of ell levels in the warmup (including ell_max)")
    parser.add_argument("--epochs_per_level", type=int, default=10,
                        help="E-step epochs per curriculum level")
    parser.add_argument("--em_steps_per_level", type=int, default=1,
                        help="Number of EM iterations (E-step + M-step) per curriculum level")
    parser.add_argument("--ell_min", type=float, default=0.0,
                        help="Starting ell value (0 = identity, easy problem)")
    parser.add_argument("--ell_max", type=float, default=1.0,
                        help="Ending ell value (1 = full F, hardest problem)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.n_em_steps = 1
        args.epochs_per_em = 1
        args.n_ell_levels = 3
        args.epochs_per_level = 1
        args.em_steps_per_level = 1

    # Problem parameters
    corruption   = args.corruption
    noise_std    = 0.3
    p_drop       = 0.3
    n_obs        = 10_000

    # SCSI parameters
    n_em_steps        = args.n_em_steps
    epochs_per_em     = args.epochs_per_em
    sample_method     = "euler"
    sample_steps      = 50
    interpolant_style = args.interpolant_style
    coupled_fraction  = args.coupled_fraction

    # Curriculum parameters
    n_ell_levels     = args.n_ell_levels
    epochs_per_level = args.epochs_per_level
    em_steps_per_level = args.em_steps_per_level
    ell_min          = args.ell_min
    ell_max          = args.ell_max
    ell_levels       = torch.linspace(ell_min, ell_max, n_ell_levels).tolist()

    # Training parameters
    batch_size = 256
    lr         = 3e-4

    print(f"Device: {device}")
    print(f"Channel: {corruption},  noise_std={noise_std}")
    print(f"Curriculum: {n_ell_levels} levels from ell={ell_min} to ell={ell_max}, "
          f"{em_steps_per_level} EM steps x {epochs_per_level} epochs/level")
    print(f"EM steps: {n_em_steps},  epochs/step: {epochs_per_em}\n")

    wandb.init(
        project="scsi-mnist-curriculum",
        config=dict(
            corruption=corruption,
            noise_std=noise_std,
            n_obs=n_obs,
            n_em_steps=n_em_steps,
            epochs_per_em=epochs_per_em,
            sample_method=sample_method,
            sample_steps=sample_steps,
            batch_size=batch_size,
            lr=lr,
            interpolant_style=interpolant_style,
            coupled_fraction=coupled_fraction,
            n_ell_levels=n_ell_levels,
            epochs_per_level=epochs_per_level,
            em_steps_per_level=em_steps_per_level,
            ell_min=ell_min,
            ell_max=ell_max,
            curriculum=True,
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
    z_pool = None

    ckpt_dir = Path(f"checkpoints_{corruption}_mnist_curriculum")
    prior_dir = Path(f"priors_{corruption}_mnist_curriculum")
    ckpt_dir.mkdir(exist_ok=True)
    prior_dir.mkdir(exist_ok=True)

    # ── Initialize model (2-channel + ell via class_labels) ───────────
    model = ConditionalDiTWithEll().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    # ── Curriculum warmup ─────────────────────────────────────────────
    print("=" * 60)
    print("CURRICULUM WARMUP")
    print("=" * 60)

    for ell_idx, ell in enumerate(ell_levels):
        print(f"\n{'=' * 60}")
        print(f"Curriculum level {ell_idx}/{n_ell_levels - 1}  (ell={ell:.3f})")
        print(f"{'=' * 60}")

        for em_k in range(em_steps_per_level):
            print(f"\n  EM step {em_k}/{em_steps_per_level - 1}  (ell={ell:.3f})")
            torch.save(x_pool, prior_dir / f"prior_curriculum_ell{ell_idx:02d}_em{em_k:02d}.pt")

            train_estep_ell(model, x_pool, ell=ell,
                            noise_std=noise_std, p_drop=p_drop, corruption=corruption,
                            style=interpolant_style, epochs=epochs_per_level,
                            batch_size=batch_size, lr=lr, global_step=global_step,
                            device=device, z_pool=z_pool, coupled_fraction=coupled_fraction)
            torch.save(model.state_dict(),
                       ckpt_dir / f"model_curriculum_ell{ell_idx:02d}_em{em_k:02d}.pt")

            x_pool, z_pool = update_prior_ell(model, y_obs, ell=ell,
                                              n_steps=sample_steps, batch_size=batch_size * 3,
                                              method=sample_method, device=device)

        log_curriculum_step_wandb(x_gt_all, y_obs, x_pool, ell=ell, ell_idx=ell_idx)

    # ── Main EM loop (ell fixed at 1.0) ───────────────────────────────
    print("\n" + "=" * 60)
    print("MAIN EM LOOP  (ell = 1.0)")
    print("=" * 60)

    for k in range(n_em_steps):
        print(f"\n--- EM iteration {k} ---")
        torch.save(x_pool, prior_dir / f"prior_em{k:02d}.pt")

        epochs = epochs_per_em
        train_estep_ell(model, x_pool, ell=1.0,
                        noise_std=noise_std, p_drop=p_drop, corruption=corruption,
                        style=interpolant_style, epochs=epochs,
                        batch_size=batch_size, lr=lr, global_step=global_step,
                        device=device, z_pool=z_pool, coupled_fraction=coupled_fraction)
        torch.save(model.state_dict(), ckpt_dir / f"model_em{k:02d}.pt")
        print(f"  checkpointed EM step {k}")

        x_pool, z_pool = update_prior_ell(model, y_obs, ell=1.0,
                                          n_steps=sample_steps, batch_size=batch_size * 3,
                                          method=sample_method, device=device)
        log_em_step_wandb(x_gt_all, y_obs, x_pool, em_step=k)

    wandb.finish()
    print("Done.")
