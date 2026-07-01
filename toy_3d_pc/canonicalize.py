"""Canonicalization operator C: PCA/moment-axis alignment for point clouds.

Removes the *identifiable* part of a cloud's SO(3) pose by aligning its principal axes
to a fixed reference frame. Axis order comes from eigenvalue magnitude (descending);
sign per axis is fixed via the third moment (skew) along that axis, and overall
handedness (det = +1) is enforced by flipping the least-informative (smallest-eigenvalue)
axis if needed. Translation is left untouched -- the CryoET forward channel ``F`` only
ever rotates clouds, never translates them.

Continuously-symmetric shapes (e.g. a solid torus about its own axis) have no
identifiable in-plane angle: ``C`` pins the axis but the residual rotation about it
stays arbitrary from call to call -- an intrinsic limitation of pose-only
canonicalization, not a bug. Validate on an asymmetric shape (``trefoil``, ``l_shape``,
``t_shape``) where ``R_hat`` is fully determined.
"""
from __future__ import annotations

import torch


def pca_canonicalize(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align each cloud's principal axes to a fixed frame (rotation only, no translation).

    points: (B, N, 3) -> (canonical (B, N, 3), R_hat (B, 3, 3)) where
    ``canonical - mean == (points - mean) @ R_hat``.
    """
    mean = points.mean(dim=1, keepdim=True)                              # (B, 1, 3)
    centered = points - mean
    cov = torch.einsum("bni,bnj->bij", centered, centered) / centered.shape[1]  # (B, 3, 3)

    # torch.linalg.eigh has no MPS kernel; the matrices are tiny (B, 3, 3) so a CPU
    # round-trip is negligible regardless of the training device.
    eigvals, eigvecs = torch.linalg.eigh(cov.detach().cpu())             # ascending eigvals
    eigvecs = eigvecs.flip(-1).to(points.device, points.dtype)           # -> descending order

    # Sign disambiguation: flip each axis so its third moment (skew) is non-negative.
    proj = torch.einsum("bni,bij->bnj", centered, eigvecs)               # (B, N, 3)
    skew = (proj ** 3).sum(dim=1)                                        # (B, 3)
    signs = torch.where(skew >= 0, 1.0, -1.0).to(eigvecs.dtype)
    eigvecs = eigvecs * signs.unsqueeze(1)

    # Enforce a proper rotation (det = +1) by flipping the smallest-eigenvalue axis.
    flip = torch.linalg.det(eigvecs) < 0
    if flip.any():
        eigvecs = eigvecs.clone()
        eigvecs[flip, :, -1] *= -1

    canonical = torch.einsum("bni,bij->bnj", centered, eigvecs) + mean
    return canonical, eigvecs
