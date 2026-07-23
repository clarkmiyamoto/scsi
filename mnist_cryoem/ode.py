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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly integrates the image and pose branches from t=0 (noise) to t=1 (data)."""
    model.eval()
    if method != "euler":
        raise ValueError(f"Unknown method: {method!r} (only 'euler' is implemented)")

    B = z_image.size(0)
    x = z_image
    theta = sample_uniform_angle(B, z_image.device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t_frac = torch.full((B,), t_val, device=z_image.device)
        v_x, v_theta = model(x, theta, t_frac, y)
        x = x + v_x * dt
        theta = wrap_to_pi(theta + v_theta * dt)

    return x, theta
