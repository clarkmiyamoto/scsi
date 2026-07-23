import torch
import torch.nn as nn
import torch.nn.functional as F

from si import interpolant, pose_interpolant
from ode import sample_joint
from corruption import forward_channel, sample_uniform_angle
from model import IMAGE_SIZE
from wandb_logging import log_train_step

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

########################################################
# The E-step and M-step of the SCSI EM algorithm (see algorithm block in the paper /
# top-level docs). em.py's outer loop calls these once per k; everything else in this
# file is in service of one or the other.
########################################################


########################################################
# E-step: propose clean data using the teacher Phi^(k-1)
########################################################

def sample_pool_indices(N_obs: int, pool_size: int) -> torch.Tensor:
    """Fresh random subset of observation indices — the `y ~ mu` draw for one outer iteration."""
    pool_size = min(pool_size, N_obs)
    perm = torch.randperm(N_obs)
    return perm[:pool_size]


@torch.no_grad()
def propose_estep(
    model: nn.Module,
    y_obs_subset: torch.Tensor,   # (N, 1, W) — only the resampled subset for this iteration
    n_steps: int = 50,
    batch_size: int = 256,
    method: str = "euler",
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    E-step / Phi^(k-1): run the joint ODE (ode.sample_joint) conditioned on y_obs_subset to
    produce (x_hat, R_hat) for every observation in the subset. `y_obs_subset` may be the whole
    dataset or a small slice, depending on how the caller sized `pool_size` (see main.py's
    literal-vs-amortized parameterization).
    """
    model.eval()
    N = y_obs_subset.size(0)
    x_chunks, theta_chunks = [], []
    y_gpu = y_obs_subset.to(device)

    for start in tqdm(range(0, N, batch_size), desc="  E-step", leave=False):
        end = min(start + batch_size, N)
        B = end - start
        z_image = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        x_batch, theta_batch = sample_joint(model, z_image, y_gpu[start:end],
                                            n_steps=n_steps, method=method)
        x_chunks.append(x_batch.cpu())
        theta_chunks.append(theta_batch.cpu())

    x_pool = torch.cat(x_chunks, dim=0)
    theta_pool = torch.cat(theta_chunks, dim=0)
    print(f"    pool  x range=[{x_pool.min():.3f}, {x_pool.max():.3f}]"
          f"  mean={x_pool.mean():.4f}  std={x_pool.std():.4f}")

    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    return x_pool, theta_pool


########################################################
# M-step: update the student against the teacher-proposed pool
########################################################

def loss_func_joint(
    model: nn.Module,
    x_hat: torch.Tensor,      # (B, 1, H, W) — image branch's x1 target
    theta_hat: torch.Tensor,  # (B,)          — pose branch's x1 target
    y: torch.Tensor,          # (B, 1, W)     — observation to condition on
    style: str = "linear",
    pose_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, image_loss.detach(), pose_loss.detach()) for separate logging."""
    B = x_hat.size(0)
    t = torch.rand(B, device=x_hat.device)

    z_image = torch.randn_like(x_hat)
    theta_z = sample_uniform_angle(B, x_hat.device)

    t4 = t[:, None, None, None]
    I_t, I_dot_t = interpolant(z_image, x_hat, t4, style)
    theta_t, theta_dot_t = pose_interpolant(theta_z, theta_hat, t)

    v_x, v_theta = model(I_t, theta_t, t, y)

    loss_image = F.mse_loss(v_x, I_dot_t)
    loss_pose = F.mse_loss(v_theta, theta_dot_t)
    loss = loss_image + pose_loss_weight * loss_pose
    return loss, loss_image.detach(), loss_pose.detach()


def train_mstep(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    x_pool: torch.Tensor,      # (N, 1, H, W) — this iteration's Phi^(k-1)-generated x_hat's
    theta_pool: torch.Tensor,  # (N,)          — this iteration's Phi^(k-1)-generated R_hat's
    noise_std: float,
    style: str,
    pose_loss_weight: float,
    steps_per_em: int,
    batch_size: int,
    global_step: list,
    device: torch.device,
    use_wandb: bool,
) -> None:
    """
    M-step: runs exactly `steps_per_em` gradient steps (T_tr in the pseudocode) against the
    FIXED (x_pool, theta_pool) generation — not "epochs" over it, so the caller can dial
    steps_per_em down to 1 (literal pseudocode: one fresh Phi^(k-1) draw per SGD step) or up
    (amortized: many gradient steps reuse one Phi^(k-1) draw, cheaper per step).

    The optimizer is passed in and persists across EM outer iterations (unlike this repo's
    other experiment directories, which recreate AdamW every SGD-training call). That reset is
    harmless when each call takes many steps, but would cripple Adam's moment estimates if
    steps_per_em is small — since our whole point is to support steps_per_em=1, the optimizer
    has to survive across calls.

    ŷ = F(x_pool; theta_pool) is recomputed fresh each batch: pose fixed to that sample's
    Phi^(k-1)-recovered R_hat, noise redrawn every time — the literal ŷ = F(x̂; R̂) term.
    """
    N = x_pool.size(0)
    model.train()

    perm = torch.randperm(N)
    ptr = 0
    running_img, running_pose = 0.0, 0.0

    for step in tqdm(range(steps_per_em), desc="  M-step", leave=False):
        if ptr + batch_size > N:
            perm = torch.randperm(N)
            ptr = 0
        idx = perm[ptr:ptr + min(batch_size, N)]
        ptr += batch_size

        x_batch = x_pool[idx].to(device)
        theta_batch = theta_pool[idx].to(device)

        y_batch, _ = forward_channel(x_batch, noise_std=noise_std, theta=theta_batch)

        loss, loss_img, loss_pose = loss_func_joint(
            model, x_batch, theta_batch, y_batch,
            style=style, pose_loss_weight=pose_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_img += loss_img.item()
        running_pose += loss_pose.item()

        log_train_step(loss_img, loss_pose, grad_norm, global_step[0], use_wandb)
        if global_step is not None:
            global_step[0] += 1

    print(f"    steps={steps_per_em}  loss_image={running_img / steps_per_em:.5f}"
          f"  loss_pose={running_pose / steps_per_em:.5f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
