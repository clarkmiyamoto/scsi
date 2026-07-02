"""Canonicalization operator C for point clouds (reference-ICP only).

Removes the *identifiable* part of a cloud's SO(2) pose so the velocity field only ever
has to emit clouds in one frame: :func:`reference_canonicalize` (used by ``--canonicalize``)
rigidly aligns each cloud to a shared *reference* cloud by ICP (nearest-neighbor
correspondences + Kabsch, multi-start to escape local minima), so every sample lands in the
*same* frame **by construction**. This is online "class-averaging": the reference is seeded
from one representative cloud and EMA-updated through the ICP correspondences.

Unlike ``toy_3d_pc/canonicalize.py`` there is **no** ``pca_canonicalize`` here: that package's
own lesson (documented in its module docstring) is that the PCA/moment-axis variant is
reference-free, so every degree of freedom a shape's symmetry (or a near-degenerate eigenvalue)
leaves unidentifiable gets decided by per-sample *sampling noise* -- a different arbitrary
frame each call, which injects rotation noise into the regression target and hurt results
there. This package deliberately does not reintroduce that path.

Point clouds are **unordered sets**, so there are no cross-cloud correspondences: alignment
needs ICP, not a one-shot Procrustes solve, and the reference cannot be built by point-wise
averaging.

Everything here runs pose-only (translation is never touched -- the CryoEM channel ``F`` only
rotates) and non-differentiably (``x_hat`` arrives from a ``no_grad`` transport and enters the
loss only as a data target). ``torch.linalg.svd`` has poor/absent MPS kernels, so the tiny
``(B, 2, 2)`` linear algebra is routed through CPU regardless of device.
"""
from __future__ import annotations

import math

import torch


# ── Rigid-alignment primitives ────────────────────────────────────────────────


def kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Rotation ``R`` (…, 2, 2) minimizing ``||P @ R^T - Q||`` for *ordered* rows.

    Row ``i`` of ``P`` corresponds to row ``i`` of ``Q``. Reflections are forbidden via the
    standard determinant sign fix. ``torch.linalg.svd`` has no MPS kernel (auto-falls-back to
    CPU with a warning), so on MPS the tiny ``(…, 2, 2)`` SVD is routed through CPU explicitly;
    on CUDA/CPU it stays on-device (native, no sync).
    """
    Pc = P - P.mean(dim=-2, keepdim=True)
    Qc = Q - Q.mean(dim=-2, keepdim=True)
    H = Pc.transpose(-1, -2) @ Qc                                        # (…, 2, 2) = sum p q^T
    on_mps = H.device.type == "mps"
    Hc = H.detach().cpu() if on_mps else H.detach()
    U, _, Vh = torch.linalg.svd(Hc)
    V = Vh.transpose(-1, -2)
    Ut = U.transpose(-1, -2)
    d = torch.linalg.det(V @ Ut)                                         # (…,)  +/-1
    D = torch.eye(2, device=Hc.device).expand(*d.shape, 2, 2).clone()
    D[..., 1, 1] = torch.sign(d)
    R = V @ D @ Ut                                                       # q_i ~= R p_i
    return R.to(P.device, P.dtype)


def chamfer(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Symmetric mean nearest-neighbor distance between clouds. A (…, N, 2), B (…, M, 2) -> (…)."""
    d = torch.cdist(A, B)                                               # (…, N, M)
    return 0.5 * (d.amin(dim=-1).mean(dim=-1) + d.amin(dim=-2).mean(dim=-1))


_seed_cache: dict[tuple, torch.Tensor] = {}


def _seed_rotations(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``n`` fixed rotations spread over SO(2), identity first. Cached, deterministic.

    ICP only converges locally; ``x_hat`` can sit at any global pose, so each alignment is
    restarted from this fixed spread and the lowest-Chamfer result is kept.
    """
    key = (n, str(device), dtype)
    if key not in _seed_cache:
        angles = torch.zeros(n)
        if n > 1:
            angles[1:] = torch.linspace(0.0, 2.0 * math.pi, n, dtype=torch.float32)[:-1] + (
                math.pi / n
            )  # evenly spread, offset from 0 so none coincide with the identity seed
        c, s = torch.cos(angles), torch.sin(angles)
        mats = torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ], dim=-2)
        _seed_cache[key] = mats.to(device, dtype)
    return _seed_cache[key]


def icp_align(
    source: torch.Tensor,        # (B, N, 2) centered
    reference: torch.Tensor,     # (M, 2) centered
    n_iters: int = 6,
    n_restarts: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-start ICP aligning each ``source`` cloud onto the shared ``reference``.

    Returns ``(R (B, 2, 2), aligned (B, N, 2))`` with ``aligned = source @ R^T`` (centered),
    where ``R`` is the best rotation (lowest Chamfer to ``reference``) over ``n_restarts``
    seed rotations, each refined by ``n_iters`` NN-correspondence + Kabsch iterations.
    """
    B, N, _ = source.shape
    ref = reference.unsqueeze(0).expand(B, -1, -1)                      # (B, M, 2)
    seeds = _seed_rotations(n_restarts, source.device, source.dtype)   # (R, 2, 2)

    best_err = source.new_full((B,), float("inf"))
    best_R = torch.eye(2, device=source.device, dtype=source.dtype).expand(B, 2, 2).clone()
    best_aligned = source.clone()

    for s in range(n_restarts):
        R = seeds[s].expand(B, 2, 2).contiguous()
        cur = source @ R.transpose(-1, -2)
        for _ in range(n_iters):
            nn = torch.cdist(cur, ref).argmin(dim=-1)                  # (B, N) -> ref index
            Q = torch.gather(ref, 1, nn.unsqueeze(-1).expand(-1, -1, 2))
            dR = kabsch(cur, Q)                                        # cur -> Q
            R = dR @ R
            cur = source @ R.transpose(-1, -2)
        err = chamfer(cur, ref)                                        # (B,)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_R = torch.where(take[:, None, None], R, best_R)
        best_aligned = torch.where(take[:, None, None], cur, best_aligned)

    return best_R, best_aligned


# ── Reference-based canonicalization operator C ────────────────────────────────


def seed_reference(x_hat: torch.Tensor) -> torch.Tensor:
    """Seed the canonical-frame reference from one representative cloud (centered).

    Uses ``x_hat[0]`` -- an arbitrary but fixed choice, which is all that is needed since the
    canonical frame is only defined *up to* a global rotation; consistency across samples,
    not absolute orientation, is what matters. The EMA updates in :func:`update_reference`
    then denoise it over training. A point-wise mean over the batch would be invalid here
    (clouds are unordered, no cross-cloud correspondence).
    """
    return (x_hat[0] - x_hat[0].mean(dim=0, keepdim=True)).detach().clone()


def reference_canonicalize(
    x_hat: torch.Tensor,         # (B, N, 2)
    reference: torch.Tensor,     # (M, 2) centered
    n_iters: int = 6,
    n_restarts: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C(x_hat): rigidly align each cloud onto the shared reference frame.

    Returns ``(x_hat_C (B, N, 2), aligned_centered (B, N, 2))``. ``x_hat_C`` keeps each
    cloud's own centroid (only rotation is removed); ``aligned_centered`` is the same clouds
    in the reference frame with the centroid removed, reused by :func:`update_reference`.
    """
    mu = x_hat.mean(dim=1, keepdim=True)
    _, aligned = icp_align(x_hat - mu, reference, n_iters=n_iters, n_restarts=n_restarts)
    return aligned + mu, aligned


def update_reference(
    reference: torch.Tensor,     # (M, 2) centered
    aligned: torch.Tensor,       # (B, N, 2) centered, in the reference frame
    decay: float,
) -> torch.Tensor:
    """EMA-update the reference through ICP correspondences (online class-averaging).

    Each aligned point is scattered onto its nearest reference index and averaged there, then
    blended into the old reference with ``decay``. Reference points that no batch point maps
    to keep their old value. Valid despite unordered clouds because the *reference index space*
    is fixed and the correspondence (not a raw point-index pairing) defines the average.
    """
    M = reference.shape[0]
    nn = torch.cdist(aligned, reference.unsqueeze(0).expand(aligned.size(0), -1, -1)).argmin(dim=-1)
    flat_pts = aligned.reshape(-1, 2)
    flat_idx = nn.reshape(-1)
    sums = torch.zeros_like(reference).index_add_(0, flat_idx, flat_pts)
    counts = torch.zeros(M, device=reference.device, dtype=reference.dtype).index_add_(
        0, flat_idx, torch.ones_like(flat_idx, dtype=reference.dtype)
    )
    has = counts > 0
    new_ref = reference.clone()
    means = sums[has] / counts[has].unsqueeze(-1)
    new_ref[has] = decay * reference[has] + (1.0 - decay) * means
    return new_ref
