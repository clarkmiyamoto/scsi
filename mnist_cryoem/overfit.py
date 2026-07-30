import torch
import torch.nn as nn

from scsi import loss_func_joint
from wandb_logging import log_overfit_step

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

########################################################
# --overfit sanity check (dispatched from main.py, before the EM loop starts): can the joint
# image+pose velocity network overfit a single FIXED (x_hat, theta_hat, y) batch? The batch is
# drawn once by the caller and never resampled here -- only the interpolant's own randomness
# (t, z_image, theta_z inside scsi.py::loss_func_joint) varies step to step, so this is NOT a
# plain regression test: the MSE target is E[I_dot | I_t, t, y], whose achievable floor is the
# irreducible conditional variance, not zero. Expect a substantial monotone decrease followed
# by a plateau at a small positive value (heaviest near t->1, where I_t stops carrying much
# information about the endpoint) -- a loss that refuses to move off its initial value at all,
# or diverges, is the actual bug signal (broken gradients, dead units, wrong target), independent
# of dataset size, EM dynamics, or corruption difficulty.
########################################################

def overfit_single_batch(
    model: nn.Module,
    x_batch: torch.Tensor,      # (B, 1, H, W) -- fixed image target, drawn once by the caller
    theta_batch: torch.Tensor,  # (B,)          -- fixed pose target, drawn once by the caller
    y_batch: torch.Tensor,      # (B, 1, W)     -- fixed observation, drawn once by the caller
    n_steps: int,
    lr: float,
    style: str,
    pose_loss_weight: float,
    use_wandb: bool,
    loss_fn=loss_func_joint,
) -> float:
    """Runs n_steps of SGD against the exact same (x_batch, theta_batch, y_batch) every step,
    using a fresh optimizer scoped to this check. Returns the final total loss.

    `loss_fn` defaults to the 2D `loss_func_joint` (unchanged behavior for every existing
    caller). main_3d.py passes `loss_fn=scsi.loss_func_joint_3d` to run this same sanity check
    against the 3D/CryoET pipeline without duplicating this file."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    loss = torch.tensor(float("nan"))
    for step in tqdm(range(n_steps), desc="overfit"):
        loss, loss_img, loss_pose = loss_fn(
            model, x_batch, theta_batch, y_batch,
            style=style, pose_loss_weight=pose_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        log_overfit_step(loss, loss_img, loss_pose, grad_norm, step, use_wandb)

        if step % max(1, n_steps // 20) == 0 or step == n_steps - 1:
            print(f"    step={step:5d}  loss={loss.item():.5f}  "
                  f"loss_image={loss_img.item():.5f}  loss_pose={loss_pose.item():.5f}  "
                  f"grad_norm={grad_norm.item():.4f}")

    return loss.item()
