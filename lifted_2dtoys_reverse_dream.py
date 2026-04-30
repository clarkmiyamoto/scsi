"""
Supervised flow matching on two-moons with AWGN corruption channel.

Joint interpolant:
    I_t^X = (1-t) z_x + t X
    I_t^Y = (1-t) z_y + t F(X),     F(X) = X + sigma * eps

Network learns the joint velocity b_t(x, y) -> (v_x, v_y) targeting
    v_x* = X - z_x
    v_y* = F(X) - z_y

Inference: given y, sample z_y, build I_t^Y = (1-t) z_y + t y analytically.
Initialize x at z_x ~ N(0, I), integrate dx/dt = b_t^X(x, I_t^Y) only.
The y output of the net is discarded; y trajectory is pinned to the analytic line.
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


# -------- model --------

class MLP(nn.Module):
    """b_t(x, y) -> (v_x, v_y). Inputs: x (B,2), y (B,2), t (B,1). Output: (B,4)."""
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(2 + 2 + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 4)]  # 2 for v_x, 2 for v_y
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        # t shape (B,) or (B,1)
        if t.dim() == 1:
            t = t[:, None]
        h = torch.cat([x, y, t], dim=-1)
        out = self.net(h)
        return out[:, :2], out[:, 2:]  # v_x, v_y


# -------- data --------

def sample_two_moons(n, noise=0.05, device="cpu"):
    x, _ = make_moons(n_samples=n, noise=noise)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    # center / scale a bit
    x = x - x.mean(0, keepdim=True)
    return x


def corrupt(x, sigma):
    return x + sigma * torch.randn_like(x)


# -------- training --------

def train_step(model, opt, batch_size, sigma, device):
    X = sample_two_moons(batch_size, noise=0.05, device=device)
    Y = corrupt(X, sigma)

    z_x = torch.randn_like(X)
    z_y = torch.randn_like(Y)
    t = torch.rand(batch_size, device=device)
    tt = t[:, None]

    Ix = (1 - tt) * z_x + tt * X
    Iy = (1 - tt) * z_y + tt * Y

    target_vx = X - z_x
    target_vy = Y - z_y

    pred_vx, pred_vy = model(Ix, Iy, t)
    loss = ((pred_vx - target_vx) ** 2).mean() + ((pred_vy - target_vy) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


# -------- inference --------

@torch.no_grad()
def sample_conditional(model, y, n_steps=100, device="cpu"):
    """
    Given y (B,2), produce X samples by:
      - sample z_y once, build I_t^Y analytically
      - sample z_x ~ N(0,I), integrate only the x-component of the drift
    Uses Euler integration on a uniform grid t in [0,1].
    """
    B = y.shape[0]
    z_x = torch.randn(B, 2, device=device)
    z_y = torch.randn(B, 2, device=device)

    x = z_x.clone()
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        Iy = (1 - t) * z_y + t * y
        t_batch = t.expand(B)
        v_x, _ = model(x, Iy, t_batch)
        x = x + dt * v_x
    return x


@torch.no_grad()
def sample_unconditional(model, n=2000, n_steps=100, device="cpu"):
    """For comparison: integrate both channels jointly from pure noise.
    Here y at t=0 is just noise and there's no observed y to condition on.
    This is the 'joint generation' mode."""
    x = torch.randn(n, 2, device=device)
    y = torch.randn(n, 2, device=device)
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = t.expand(n)
        v_x, v_y = model(x, y, t_batch)
        x = x + dt * v_x
        y = y + dt * v_y
    return x, y


# -------- plotting --------

def make_diagnostic_plot(model, sigma, step, out_dir, device):
    """
    Three panels:
      (a) clean two-moons + corrupted y points (the conditioning data)
      (b) reconstructions x_hat from the conditional sampler, colored by which y they came from
      (c) unconditional joint sampling for sanity (X marginal)
    """
    n_eval = 1500
    X_clean = sample_two_moons(n_eval, noise=0.05, device=device)
    Y_obs = corrupt(X_clean, sigma)
    X_hat = sample_conditional(model, Y_obs, n_steps=100, device=device)
    X_uncond, _ = sample_unconditional(model, n=n_eval, n_steps=100, device=device)

    X_clean_np = X_clean.cpu().numpy()
    Y_obs_np = Y_obs.cpu().numpy()
    X_hat_np = X_hat.cpu().numpy()
    X_uncond_np = X_uncond.cpu().numpy()

    # High-contrast palette to make "real vs generated" immediately legible.
    real_color = "#222222"       # real clean distribution
    observed_color = "#9AA0A6"   # noisy observations (de-emphasized)
    generated_color = "#D94801"  # NN-generated samples
    uncond_color = "#7B2CBF"     # unconditional joint sample

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(X_clean_np[:, 0], X_clean_np[:, 1], s=5, alpha=0.85, label="real clean X", color=real_color)
    ax.scatter(Y_obs_np[:, 0], Y_obs_np[:, 1], s=4, alpha=0.30, label=f"observed y = X + N(0,{sigma}²)", color=observed_color)
    ax.set_title("data + corruption")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(Y_obs_np[:, 0], Y_obs_np[:, 1], s=4, alpha=0.20, label="observed y", color=observed_color)
    ax.scatter(X_hat_np[:, 0], X_hat_np[:, 1], s=6, alpha=0.85, label="NN generated x-hat ~ p(X|y)", color=generated_color)
    ax.set_title(f"conditional reconstruction  (step {step})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    ax = axes[2]
    ax.scatter(X_clean_np[:, 0], X_clean_np[:, 1], s=5, alpha=0.35, label="real clean X", color=real_color)
    ax.scatter(X_uncond_np[:, 0], X_uncond_np[:, 1], s=5, alpha=0.70, label="NN generated X (joint sample)", color=uncond_color)
    ax.set_title("unconditional joint X")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    # match limits across the three panels
    all_pts = np.concatenate([X_clean_np, Y_obs_np, X_hat_np, X_uncond_np], axis=0)
    pad = 0.3
    xmin, ymin = all_pts.min(0) - pad
    xmax, ymax = all_pts.max(0) + pad
    for a in axes:
        a.set_xlim(xmin, xmax)
        a.set_ylim(ymin, ymax)

    fig.suptitle(f"step {step}   |   σ = {sigma}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# -------- main --------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.1, help="AWGN noise std for corruption")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--plot_every", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./outputs/supervised_flow_runs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model = MLP(hidden=args.hidden, depth=args.depth).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # plot at init for reference
    make_diagnostic_plot(model, args.sigma, step=0, out_dir=args.out_dir, device=args.device)

    losses = []
    saved = []
    for step in range(1, args.steps + 1):
        loss = train_step(model, opt, args.batch_size, args.sigma, args.device)
        losses.append(loss)
        if step % 200 == 0:
            recent = sum(losses[-200:]) / 200
            print(f"step {step:6d}   loss {loss:.4f}   ema200 {recent:.4f}")
        if step % args.plot_every == 0 or step == args.steps:
            path = make_diagnostic_plot(model, args.sigma, step, args.out_dir, args.device)
            saved.append(path)
            print(f"  saved {path}")

    # final loss curve
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    fig.tight_layout()
    loss_path = os.path.join(args.out_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=110)
    plt.close(fig)

    print("done.")
    print("plots:", args.out_dir)


if __name__ == "__main__":
    main()