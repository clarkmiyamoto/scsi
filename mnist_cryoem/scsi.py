import torch
import torch.nn as nn
import torch.nn.functional as F

from si import interpolant, pose_interpolant
from ode import sample_joint, sample_joint_3d
from corruption import (
    forward_channel, sample_uniform_angle, forward_channel_3d, forward_channel_mra,
)
from model import IMAGE_SIZE, VOL_SIZE
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    E-step / Phi^(k-1): run the joint ODE (ode.sample_joint) conditioned on y_obs_subset to
    produce (x_hat, R_hat) for every observation in the subset. `y_obs_subset` may be the whole
    dataset or a small slice, depending on how the caller sized `pool_size` (see main.py's
    literal-vs-amortized parameterization).

    R_hat is OPTIONAL: if `model` has no pose/latent branch (model.pose_branch is None, see
    model.ConditionalVelocityMRA's use_pose_head), ode.sample_joint returns theta_batch=None
    every chunk, and theta_pool is returned as None rather than a concatenated tensor. Callers
    (scsi.train_mstep_mra, wandb_logging.log_em_pool_diagnostics) already treat it as Optional.
    """
    model.eval()
    N = y_obs_subset.size(0)
    has_pose = getattr(model, "pose_branch", None) is not None
    x_chunks, theta_chunks = [], []
    y_gpu = y_obs_subset.to(device)

    for start in tqdm(range(0, N, batch_size), desc="  E-step", leave=False):
        end = min(start + batch_size, N)
        B = end - start
        z_image = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        x_batch, theta_batch = sample_joint(model, z_image, y_gpu[start:end],
                                            n_steps=n_steps, method=method)
        x_chunks.append(x_batch.cpu())
        if has_pose:
            theta_chunks.append(theta_batch.cpu())

    x_pool = torch.cat(x_chunks, dim=0)
    theta_pool = torch.cat(theta_chunks, dim=0) if has_pose else None
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
    x_hat: torch.Tensor,              # (B, 1, H, W) — image branch's x1 target
    theta_hat: torch.Tensor | None,   # (B,)          — pose branch's x1 target, or None
    y: torch.Tensor,                  # (B, 1, W)     — observation to condition on
    style: str = "linear",
    pose_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, image_loss.detach(), pose_loss.detach()) for separate logging.

    The pose/latent branch is OPTIONAL, gated ONCE on `model.pose_branch is not None` (the
    same single source of truth ode.sample_joint and scsi.propose_estep use) -- NOT on whether
    `theta_hat` happens to be non-None. This matters: a caller with a pose-free model may still
    legitimately pass a real theta_hat (e.g. main_mra_rotation.py's --overfit path, which needs
    SOME angle to render y regardless of whether the model estimates one) -- gating on
    `theta_hat is not None` there would run pose_interpolant for a value the model was always
    going to ignore. When `model.pose_branch is None`: theta_z/pose_interpolant are skipped
    entirely, `model(..., None, ...)` is called, and loss_pose is a constant zero tensor
    (nothing to backprop, nothing informative to log). `theta_hat` must be non-None whenever
    `model.pose_branch is not None` -- the converse (a real pose branch called with
    theta_t=None) crashes inside PoseHead.forward, which is the correct loud failure.

    This is the general F: X x R -> Y pattern: R is an optional per-sample latent the loss's
    interpolant/regression targets happen to include; dropping it drops one branch of this
    function, not a parallel loss function.
    """
    B = x_hat.size(0)
    t = torch.rand(B, device=x_hat.device)

    z_image = torch.randn_like(x_hat)
    t4 = t[:, None, None, None]
    I_t, I_dot_t = interpolant(z_image, x_hat, t4, style)

    has_pose = getattr(model, "pose_branch", None) is not None
    if has_pose:
        theta_z = sample_uniform_angle(B, x_hat.device)
        theta_t, theta_dot_t = pose_interpolant(theta_z, theta_hat, t)
    else:
        theta_t, theta_dot_t = None, None

    v_x, v_theta = model(I_t, theta_t, t, y)

    loss_image = F.mse_loss(v_x, I_dot_t)
    if has_pose:
        loss_pose = F.mse_loss(v_theta, theta_dot_t)
    else:
        loss_pose = torch.zeros((), device=x_hat.device)
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


########################################################
# Rotational-MRA M-step. No propose_estep_mra / loss_func_joint_mra needed -- propose_estep and
# loss_func_joint are both already channel-agnostic (neither ever calls forward_channel or
# inspects y's shape) and are reused UNMODIFIED. train_mstep_mra is the ONE function that needs
# an MRA-specific copy, since train_mstep hardcodes a call to forward_channel.
########################################################

def train_mstep_mra(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    x_pool: torch.Tensor,              # (N, 1, H, W) — this iteration's Phi^(k-1) x_hat's
    theta_pool: torch.Tensor | None,   # (N,) this iteration's Phi^(k-1) R_hat's, or None
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
    MRA analogue of train_mstep — identical fixed-pool/persistent-optimizer contract (see
    train_mstep's docstring; the canonical-target invariant also applies unchanged: x_pool is
    always the canonical, un-rotated image, and forward_channel_mra RE-APPLIES theta_pool to
    build y_batch, never the other way around). The ONLY difference from train_mstep is which
    corruption channel builds y_batch — forward_channel_mra (rotate + full-image AWGN, no
    projection) instead of forward_channel.

    ŷ = F(x_pool; theta_pool) is recomputed fresh each batch: pose fixed to that sample's
    Phi^(k-1)-recovered R_hat, noise redrawn every time — the literal ŷ = F(x̂; R̂) term.

    theta_pool is OPTIONAL (None when model.pose_branch is None — see
    model.ConditionalVelocityMRA's use_pose_head docstring). `has_pose` is re-derived from
    `model` here too (not inferred from `theta_pool is not None`) so this function has the same
    single source of truth as propose_estep/loss_func_joint/sample_joint, rather than trusting
    that whoever produced theta_pool got it right.
    """
    N = x_pool.size(0)
    model.train()
    has_pose = getattr(model, "pose_branch", None) is not None

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
        theta_batch = theta_pool[idx].to(device) if has_pose else None

        # theta_batch=None (no pose head) falls into forward_channel_mra's OWN theta=None
        # branch, drawing a FRESH Haar-uniform rotation here instead of re-applying a
        # recovered R_hat — i.e. the M-step MARGINALIZES over the unobserved rotation rather
        # than holding a believed one fixed. Same call, same line, different meaning — see
        # model.ConditionalVelocityMRA's use_pose_head docstring.
        y_batch, _ = forward_channel_mra(x_batch, noise_std=noise_std, theta=theta_batch)

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


########################################################
# 3D / CryoET E-step and M-step. Mirror propose_estep/loss_func_joint/train_mstep exactly,
# with the volume (B,1,D,H,W) in place of the image and the flat 6D pose representation
# (si.py's module docstring) in place of the SO(2) angle — including reusing si.interpolant
# for BOTH branches in loss_func_joint_3d, since pose is no longer manifold-valued.
########################################################

@torch.no_grad()
def propose_estep_3d(
    model: nn.Module,
    y_obs_subset: torch.Tensor,   # (N, 1, H, W) — only the resampled subset for this iteration
    n_steps: int = 50,
    batch_size: int = 256,
    method: str = "euler",
    device: torch.device = None,
    vol_size: int = VOL_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D analogue of propose_estep — runs ode.sample_joint_3d conditioned on y_obs_subset."""
    model.eval()
    N = y_obs_subset.size(0)
    x_chunks, pose_chunks = [], []
    y_gpu = y_obs_subset.to(device)

    for start in tqdm(range(0, N, batch_size), desc="  E-step", leave=False):
        end = min(start + batch_size, N)
        B = end - start
        z_vol = torch.randn(B, 1, vol_size, vol_size, vol_size, device=device)
        x_batch, pose_batch = sample_joint_3d(model, z_vol, y_gpu[start:end],
                                              n_steps=n_steps, method=method)
        x_chunks.append(x_batch.cpu())
        pose_chunks.append(pose_batch.cpu())

    x_pool = torch.cat(x_chunks, dim=0)
    pose_pool = torch.cat(pose_chunks, dim=0)
    print(f"    pool  x range=[{x_pool.min():.3f}, {x_pool.max():.3f}]"
          f"  mean={x_pool.mean():.4f}  std={x_pool.std():.4f}")

    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    return x_pool, pose_pool


def loss_func_joint_3d(
    model: nn.Module,
    x_hat: torch.Tensor,      # (B, 1, D, H, W) — image branch's x1 target
    pose_hat: torch.Tensor,   # (B, 6)           — pose branch's x1 target
    y: torch.Tensor,          # (B, 1, H, W)     — observation to condition on
    style: str = "linear",
    pose_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    3D analogue of loss_func_joint. Reuses si.interpolant for BOTH branches — the pose branch
    needs no geodesic interpolant since it's the flat 6D representation (si.py's module
    docstring), so this is the same function, same `style`, called twice with different
    broadcast shapes.

    Returns (total_loss, image_loss.detach(), pose_loss.detach()) for separate logging.
    """
    B = x_hat.size(0)
    t = torch.rand(B, device=x_hat.device)

    z_vol = torch.randn_like(x_hat)
    pose_z = torch.randn(B, 6, device=x_hat.device)  # plain Gaussian noise, not a rotation draw

    t5 = t[:, None, None, None, None]
    t2 = t[:, None]
    I_t, I_dot_t = interpolant(z_vol, x_hat, t5, style)
    pose_t, pose_dot_t = interpolant(pose_z, pose_hat, t2, style)

    v_x, v_pose = model(I_t, pose_t, t, y)

    loss_image = F.mse_loss(v_x, I_dot_t)
    loss_pose = F.mse_loss(v_pose, pose_dot_t)
    loss = loss_image + pose_loss_weight * loss_pose
    return loss, loss_image.detach(), loss_pose.detach()


def train_mstep_3d(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    x_pool: torch.Tensor,     # (N, 1, D, H, W) — this iteration's Phi^(k-1)-generated x_hat's
    pose_pool: torch.Tensor,  # (N, 6)           — this iteration's Phi^(k-1)-generated poses
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
    3D analogue of train_mstep — same fixed-pool/persistent-optimizer contract (see
    train_mstep's docstring). NOTE: --steps_per_em 1 (literal pseudocode, a fresh Phi^(k-1) pool
    generation every SGD step) is computationally IMPRACTICAL at 3D scale — the E-step's cost is
    pool_size * sample_steps full 3D UNet passes per pool refresh; amortized mode (steps_per_em
    in the tens-to-hundreds) is the only practical regime here, even though the code path itself
    supports steps_per_em=1 just like the 2D version.
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
        pose_batch = pose_pool[idx].to(device)

        y_batch, _ = forward_channel_3d(x_batch, noise_std=noise_std, pose6=pose_batch)

        loss, loss_img, loss_pose = loss_func_joint_3d(
            model, x_batch, pose_batch, y_batch,
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
