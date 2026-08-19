"""Blind rigid-alignment primitives, for evaluating a reconstruction against a known
ground truth under an unknown global SO(3) rotation.

Copied from ``toy_3d_pc/canonicalize.py`` (``kabsch``, ``chamfer``, ``icp_align``
only -- the EM-specific ``seed_reference``/``update_reference``, which assume a
self-bootstrapped reference rather than a known ground truth, are not needed here).

Point clouds are **unordered sets**, so there is no cross-cloud correspondence:
alignment needs multi-restart ICP (nearest-neighbor correspondences + Kabsch,
repeated from several seed rotations to escape local minima), not a one-shot
Procrustes solve. Runs non-differentiably (this is an eval-time metric, not part of
the training loss). ``torch.linalg.svd`` has poor/absent MPS kernels, so the tiny
``(..., 3, 3)`` linear algebra is routed through CPU regardless of device.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Rotation ``R`` (…, 3, 3) minimizing ``||P @ R^T - Q||`` for *ordered* rows.

    Row ``i`` of ``P`` corresponds to row ``i`` of ``Q``. Reflections are forbidden
    via the standard determinant sign fix, so chiral shapes are never mirrored.
    """
    Pc = P - P.mean(dim=-2, keepdim=True)
    Qc = Q - Q.mean(dim=-2, keepdim=True)
    H = Pc.transpose(-1, -2) @ Qc                                        # (…, 3, 3)
    on_mps = H.device.type == "mps"
    Hc = H.detach().cpu() if on_mps else H.detach()
    U, _, Vh = torch.linalg.svd(Hc)
    V = Vh.transpose(-1, -2)
    Ut = U.transpose(-1, -2)
    d = torch.linalg.det(V @ Ut)                                         # (…,)  +/-1
    D = torch.eye(3, device=Hc.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = torch.sign(d)
    R = V @ D @ Ut
    return R.to(P.device, P.dtype)


def chamfer(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Symmetric mean nearest-neighbor distance. A (…, N, 3), B (…, M, 3) -> (…,)."""
    d = torch.cdist(A, B)
    return 0.5 * (d.amin(dim=-1).mean(dim=-1) + d.amin(dim=-2).mean(dim=-1))


_seed_cache: dict[tuple, torch.Tensor] = {}


def _seed_rotations(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``n`` fixed rotations spread over SO(3), identity first. Cached, deterministic."""
    key = (n, str(device), dtype)
    if key not in _seed_cache:
        mats = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        if n > 1:
            mats[1:] = Rotation.random(n - 1, random_state=0).as_matrix().astype(np.float32)
        _seed_cache[key] = torch.from_numpy(mats).to(device, dtype)
    return _seed_cache[key]


def icp_align(
    source: torch.Tensor,        # (B, N, 3) centered
    reference: torch.Tensor,     # (M, 3) centered
    n_iters: int = 6,
    n_restarts: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-start ICP aligning each ``source`` cloud onto the shared ``reference``.

    Returns ``(R (B, 3, 3), aligned (B, N, 3))`` with ``aligned = source @ R^T``
    (centered), where ``R`` is the best rotation (lowest Chamfer to ``reference``)
    over ``n_restarts`` seed rotations, each refined by ``n_iters`` NN-correspondence
    + Kabsch iterations.
    """
    B, N, _ = source.shape
    ref = reference.unsqueeze(0).expand(B, -1, -1)
    seeds = _seed_rotations(n_restarts, source.device, source.dtype)

    best_err = source.new_full((B,), float("inf"))
    best_R = torch.eye(3, device=source.device, dtype=source.dtype).expand(B, 3, 3).clone()
    best_aligned = source.clone()

    for s in range(n_restarts):
        R = seeds[s].expand(B, 3, 3).contiguous()
        cur = source @ R.transpose(-1, -2)
        for _ in range(n_iters):
            nn = torch.cdist(cur, ref).argmin(dim=-1)
            Q = torch.gather(ref, 1, nn.unsqueeze(-1).expand(-1, -1, 3))
            dR = kabsch(cur, Q)
            R = dR @ R
            cur = source @ R.transpose(-1, -2)
        err = chamfer(cur, ref)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_R = torch.where(take[:, None, None], R, best_R)
        best_aligned = torch.where(take[:, None, None], cur, best_aligned)

    return best_R, best_aligned
