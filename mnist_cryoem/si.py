import torch

########################################################
# Image-branch interpolant (linear / gvp), ported from image_2d/si.py
########################################################

def alpha_linear(t):     return 1.0 - t
def beta_linear(t):      return t
def alpha_dot_linear(t): return -1.0
def beta_dot_linear(t):  return 1.0

def alpha_gvp(t):     return torch.cos(t * torch.pi / 2.0)
def beta_gvp(t):      return torch.sin(t * torch.pi / 2.0)
def alpha_dot_gvp(t): return -torch.pi / 2.0 * torch.sin(t * torch.pi / 2.0)
def beta_dot_gvp(t):  return  torch.pi / 2.0 * torch.cos(t * torch.pi / 2.0)


def interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                style: str = "linear") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x0: (B, 1, H, W) noise
        x1: (B, 1, H, W) data (x_hat, the previous network's generated sample)
        t:  (B, 1, 1, 1) broadcastable time
    Returns:
        I_t, dI_t/dt
    """
    if style == "linear":
        I_t     = alpha_linear(t) * x0     + beta_linear(t) * x1
        I_dot_t = alpha_dot_linear(t) * x0 + beta_dot_linear(t) * x1
    elif style == "gvp":
        I_t     = alpha_gvp(t) * x0     + beta_gvp(t) * x1
        I_dot_t = alpha_dot_gvp(t) * x0 + beta_dot_gvp(t) * x1
    else:
        raise ValueError(f"Unknown interpolant style: {style!r}")
    return I_t, I_dot_t


########################################################
# Pose-branch interpolant: geodesic (shortest-arc) interpolation on SO(2).
# Always uses a constant-angular-velocity schedule regardless of --interpolant_style
# (that flag applies to the image branch only) — geodesics are the natural constant-speed
# paths on a circle, so there's no "gvp" analogue here.
########################################################

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle (or angle difference) into (-pi, pi]."""
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def pose_interpolant(theta_z: torch.Tensor, theta_hat: torch.Tensor,
                     t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        theta_z:   (B,) noise angle (pose "x0")
        theta_hat: (B,) data angle (pose "x1", from the previous network's Phi)
        t:         (B,) time in [0, 1]
    Returns:
        theta_t:     (B,) raw angle (not re-wrapped; cos/sin downstream are periodic)
        theta_dot_t: (B,) constant angular velocity along the shortest arc
    """
    delta = wrap_to_pi(theta_hat - theta_z)  # shortest-arc difference in (-pi, pi]
    theta_t = theta_z + t * delta
    theta_dot_t = delta
    return theta_t, theta_dot_t
