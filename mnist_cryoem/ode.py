import torch
import torch.nn as nn

from corruption import sample_uniform_angle
from si import wrap_to_pi

########################################################
# Phi: the joint ODE integrator that the E-step runs to turn noise into a proposed
# (x_hat, R_hat) pair, conditioned on an observation y. Owning this as its own module
# (rather than inlining it in scsi.py) is what makes the integrator tunable: adding a
# new `method` only touches this file.
#
# Currently Euler-only, matching every other experiment directory in this repo. The
# pose branch integrates on SO(2) (wrap_to_pi after each step), so a future Heun/RK4
# step can't just reuse a generic Euclidean implementation for both branches — it needs
# its own geodesic-aware version for the pose half of the state.
########################################################

@torch.no_grad()
def sample_joint(
    model: nn.Module,
    z_image: torch.Tensor,   # (B, 1, H, W) initial noise image
    y: torch.Tensor,         # (B, 1, W) observation to condition on
    n_steps: int = 50,
    method: str = "euler",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Jointly integrates the image and pose/latent branches from t=0 (noise) to t=1 (data).

    The pose/latent state R is OPTIONAL: `model` is the single source of truth for whether one
    exists (`model.pose_branch is not None` -- see model.ConditionalVelocityMRA's use_pose_head
    docstring). If absent, theta is never initialized or updated -- only the image branch
    integrates -- and the returned theta is None. Every caller of this function
    (scsi.propose_estep, wandb_logging.log_reconstruction_grid_mra) must treat the second
    return value as Optional. This is the general F: X x R -> Y pattern applied to the
    integrator: R can be any per-sample nuisance/latent the forward channel depends on (here
    an SO(2) angle), and dropping it should only ever require this function to skip touching
    it, not a parallel copy of the integration loop.
    """
    model.eval()
    if method != "euler":
        raise ValueError(f"Unknown method: {method!r} (only 'euler' is implemented)")

    has_pose = getattr(model, "pose_branch", None) is not None
    B = z_image.size(0)
    x = z_image
    theta = sample_uniform_angle(B, z_image.device) if has_pose else None
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t_frac = torch.full((B,), t_val, device=z_image.device)
        v_x, v_theta = model(x, theta, t_frac, y)
        x = x + v_x * dt
        if has_pose:
            theta = wrap_to_pi(theta + v_theta * dt)

    return x, theta


########################################################
# 3D joint integrator. Unlike the SO(2) pose branch above, the 3D pose state is the flat
# 6D continuous rotation representation (si.gram_schmidt_to_matrix/matrix_to_gram_schmidt —
# see si.py's module docstring for the full rationale), so BOTH branches here integrate via
# plain Euler with no wrap/exponential-map step — the pose branch is structurally identical
# to the image branch, not a special case.
########################################################

@torch.no_grad()
def sample_joint_3d(
    model: nn.Module,
    z_vol: torch.Tensor,     # (B, 1, D, H, W) initial noise volume
    y: torch.Tensor,         # (B, 1, H, W) observation to condition on
    n_steps: int = 50,
    method: str = "euler",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly integrates the volume and 6D-pose branches from t=0 (noise) to t=1 (data)."""
    model.eval()
    if method != "euler":
        raise ValueError(f"Unknown method: {method!r} (only 'euler' is implemented)")

    B = z_vol.size(0)
    x = z_vol
    pose = torch.randn(B, 6, device=z_vol.device)  # plain Gaussian noise, not a rotation draw
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t_frac = torch.full((B,), t_val, device=z_vol.device)
        v_x, v_pose = model(x, pose, t_frac, y)
        x = x + v_x * dt
        pose = pose + v_pose * dt  # plain Euler — no wrap, no exponential map

    return x, pose
