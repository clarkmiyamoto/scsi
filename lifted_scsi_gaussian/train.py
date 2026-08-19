"""Lifted SCSI on a 1D Gaussian clean prior under an AWGN channel.

Sweeps the coupling probability alpha_z and measures how fast the learned
conditional flow converges to the true (conjugate, closed-form) Gaussian
posterior p(x|y).

Usage:
    uv run python lifted_scsi_gaussian/train.py
    uv run python lifted_scsi_gaussian/train.py --alpha_z 0,1 --n_seeds 1 --K 10
"""
import argparse
import copy
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


########################################################
# Problem setup: 1D Gaussian prior, AWGN channel
########################################################

def sample_clean(n: int, mu0: float, sigma0: float) -> torch.Tensor:
    return mu0 + sigma0 * torch.randn(n, 1, device=device)


def forward_channel(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    return x + noise_std * torch.randn_like(x)


def sample_observations(n: int, mu0: float, sigma0: float, noise_std: float) -> torch.Tensor:
    """y ~ mu, obtained by pushing the (unobserved) clean prior through F."""
    x = sample_clean(n, mu0, sigma0)
    return forward_channel(x, noise_std)


def analytic_posterior(y: torch.Tensor, mu0: float, sigma0: float, noise_std: float):
    """Conjugate Gaussian posterior p(x|y) for a Gaussian prior + AWGN likelihood."""
    var0, varn = sigma0 ** 2, noise_std ** 2
    shrink = var0 / (var0 + varn)
    mu_post = mu0 + shrink * (y - mu0)
    sigma_post = math.sqrt(var0 * varn / (var0 + varn))
    return mu_post, sigma_post


########################################################
# Model
########################################################

class VelocityMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t, y], dim=-1))


########################################################
# Interpolant (linear schedule, matches image_2d/si.py convention)
########################################################

def interpolant(z: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
    I_t = (1.0 - t) * z + t * x
    I_dot_t = x - z
    return I_t, I_dot_t


@torch.no_grad()
def euler_sample(model: nn.Module, init: torch.Tensor, cond: torch.Tensor, n_steps: int) -> torch.Tensor:
    model.eval()
    B = init.size(0)
    x = init
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B, 1), i * dt, device=init.device)
        x = x + model(x, t, cond) * dt
    return x


########################################################
# Evaluation: analytic posterior vs. model's conditional flow
########################################################

@torch.no_grad()
def evaluate(model: nn.Module, y_eval: torch.Tensor, ode_steps: int,
             mu0: float, sigma0: float, noise_std: float):
    """One fresh z' per y_eval sample (no repeats) -> y_eval.size(0) flow samples.

    sigma_post doesn't depend on y (Gaussian prior + AWGN), so subtracting the
    y-dependent posterior mean from every x_hat pools all samples into a single
    residual distribution that should be N(0, sigma_post) if the model is exact.
    """
    n = y_eval.size(0)
    z_init = torch.randn(n, 1, device=device)
    x_hat = euler_sample(model, z_init, y_eval, ode_steps)

    mu_post, sigma_post = analytic_posterior(y_eval.squeeze(-1), mu0, sigma0, noise_std)
    residual = x_hat.squeeze(-1) - mu_post

    mean_err = residual.mean().item()
    std_err = (residual.std() - sigma_post).item()
    w2_sq = mean_err ** 2 + std_err ** 2

    metrics = dict(w2_sq=w2_sq, mean_err=abs(mean_err), std_err=abs(std_err))
    return metrics, x_hat.detach()


def plot_x1_distribution(x_hat: torch.Tensor, mu0: float, sigma0: float) -> plt.Figure:
    """Histogram of the flow's terminal output X_{t=1}, vs. the true marginal prior.

    Marginalizing the true posterior p(x|y) over y ~ mu recovers exactly the
    clean prior N(mu0, sigma0^2), so that's the ground-truth curve to compare
    the pooled x_hat samples against.
    """
    x = x_hat.squeeze(-1).cpu()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x.numpy(), bins=50, density=True, color="steelblue", alpha=0.7,
            label="model X_{t=1}")

    grid = torch.linspace(x.min() - 1, x.max() + 1, 200)
    density = torch.exp(-0.5 * ((grid - mu0) / sigma0) ** 2) / (sigma0 * math.sqrt(2 * math.pi))
    ax.plot(grid.numpy(), density.numpy(), color="crimson", lw=2,
            label="true prior N(mu0, sigma0^2)")

    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    return fig


########################################################
# Lifted SCSI training loop for a single (alpha_z, seed) model
########################################################

def train_one_model(alpha_z: float, seed: int, args, y_eval: torch.Tensor,
                     use_wandb: bool) -> list:
    torch.manual_seed(seed)

    model = VelocityMLP(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    phi_prev = copy.deepcopy(model)
    for p in phi_prev.parameters():
        p.requires_grad_(False)

    history = []

    if use_wandb:
        config = vars(args).copy()
        config["alpha_z"] = alpha_z  # overrides the full sweep-grid string with this model's value
        config["seed"] = seed
        run = wandb.init(
            project="lifted-scsi-gaussian-awgn",
            group="alpha_z_sweep",
            name=f"alpha{alpha_z}_seed{seed}",
            config=config,
            reinit="finish_previous",
        )

    for k in range(1, args.K + 1):
        model.train()
        running_loss = 0.0
        for _ in range(args.T_tr):
            y = sample_observations(args.batch_size, args.mu0, args.sigma0, args.sigma_n)

            z_prime = torch.randn(args.batch_size, 1, device=device)
            t = torch.rand(args.batch_size, 1, device=device)
            
            # # Option 1: Bernoulli coupling
            # # Coupling decided up front, mirrors the paper's ordering exactly.
            # # The coin flip never touches how x_hat is generated below (always
            # # integrated from z'); it only decides what enters the interpolant.
            # coupled_mask = torch.rand(args.batch_size, 1, device=device) < alpha_z
            # z = torch.where(coupled_mask, z_prime, torch.randn(args.batch_size, 1, device=device))
            
            # Option 2: Gaussian coupling 
            z = alpha_z * z_prime + math.sqrt(1 - alpha_z**2) * torch.randn(args.batch_size, 1, device=device) 

            with torch.no_grad():
                x_hat = euler_sample(phi_prev, z_prime, y, args.ode_steps)
                y_hat = forward_channel(x_hat, args.sigma_n)

            I_t, I_dot_t = interpolant(z, x_hat, t)

            pred = model(I_t, t, y_hat)  # conditioned on y_hat, NOT the real y
            loss = F.mse_loss(pred, I_dot_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        phi_prev = copy.deepcopy(model)
        for p in phi_prev.parameters():
            p.requires_grad_(False)

        if k % args.eval_every == 0 or k == args.K:
            metrics, x_hat_eval = evaluate(model, y_eval, args.ode_steps,
                                            args.mu0, args.sigma0, args.sigma_n)
            metrics["train_loss"] = running_loss / args.T_tr
            history.append((k, metrics))

            print(f"  alpha_z={alpha_z:.2f} seed={seed} k={k:4d}  "
                  f"loss={metrics['train_loss']:.4f}  w2_sq={metrics['w2_sq']:.4f}")

            if use_wandb:
                fig = plot_x1_distribution(x_hat_eval, args.mu0, args.sigma0)
                wandb.log({
                    "train/loss": metrics["train_loss"],
                    "eval/w2_sq": metrics["w2_sq"],
                    "eval/mean_err": metrics["mean_err"],
                    "eval/std_err": metrics["std_err"],
                    "eval/x1_distribution": wandb.Image(fig),
                }, step=k)
                plt.close(fig)

    if use_wandb:
        wandb.finish()

    return history


########################################################
# Sweep + comparison plot
########################################################

def run_sweep(args):
    y_eval = sample_observations(args.n_eval_samples, args.mu0, args.sigma0, args.sigma_n)

    alpha_grid = [float(a) for a in args.alpha_z.split(",")]
    results = {}  # alpha_z -> list of per-seed histories

    for alpha_z in alpha_grid:
        results[alpha_z] = []
        for seed in range(args.n_seeds):
            history = train_one_model(alpha_z, seed, args, y_eval, use_wandb=args.wandb)
            results[alpha_z].append(history)

    plot_comparison(results, args)


def plot_comparison(results: dict, args):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(torch.linspace(0, 1, len(results)).numpy())

    for color, (alpha_z, histories) in zip(colors, results.items()):
        ks = [k for k, _ in histories[0]]
        w2_per_seed = torch.tensor([[m["w2_sq"] for _, m in h] for h in histories])
        mean_w2 = w2_per_seed.mean(dim=0)
        std_w2 = w2_per_seed.std(dim=0)

        ax.plot(ks, mean_w2.numpy(), label=f"alpha_z={alpha_z}", color=color)
        ax.fill_between(ks, (mean_w2 - std_w2).clamp(min=1e-8).numpy(),
                         (mean_w2 + std_w2).numpy(), color=color, alpha=0.2)

    ax.set_yscale("log")
    ax.set_xlabel("outer iteration k")
    ax.set_ylabel("W2^2(model posterior, true posterior)")
    ax.set_title("Lifted SCSI convergence vs. alpha_z (1D Gaussian / AWGN)")
    ax.legend()
    fig.tight_layout()

    out_path = "lifted_scsi_gaussian/convergence_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot to {out_path}")
    plt.close(fig)


########################################################
# CLI
########################################################

def parse_args():
    p = argparse.ArgumentParser(description="Lifted SCSI on 1D Gaussian / AWGN")
    p.add_argument("--mu0", type=float, default=0.0, help="clean prior mean")
    p.add_argument("--sigma0", type=float, default=2.0, help="clean prior std")
    p.add_argument("--sigma_n", type=float, default=2.0, help="AWGN noise std")

    p.add_argument("--alpha_z", type=str, default="0.0,0.001,0.01,0.05,0.1",
                    help="comma-separated coupling probabilities to sweep")
    p.add_argument("--n_seeds", type=int, default=3, help="seeds per alpha_z")

    p.add_argument("--K", type=int, default=200, help="outer EM-style iterations")
    p.add_argument("--T_tr", type=int, default=50, help="inner SGD steps per outer iteration")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64, help="MLP hidden width")

    p.add_argument("--ode_steps", type=int, default=30, help="Euler steps for the flow")
    p.add_argument("--eval_every", type=int, default=5, help="outer iterations between evals")
    p.add_argument("--n_eval_samples", type=int, default=4096,
                    help="held-out (y, z) pairs for eval -- one fresh z per y, "
                         "so this is also the total number of flow samples per eval")

    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {device}")
    print(f"Prior: N({args.mu0}, {args.sigma0}^2)  AWGN std: {args.sigma_n}")
    print(f"alpha_z sweep: {args.alpha_z}  seeds: {args.n_seeds}\n")
    run_sweep(args)
