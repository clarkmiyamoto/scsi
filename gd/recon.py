"""Pluggable comparison/loss registry for direct gradient-descent reconstruction.

Both methods answer the same question -- "how far is the current guess X from the
observed dataset D?" -- but differ in whether they solve or marginalize each real
particle's unknown pose:

* `DistributionalRecon`: no per-particle pose needed. Each step, render a fresh
  batch of synthetic tilt series from the current X (fresh random pose *and* fresh
  noise, matching the true corruption process) and minimize a distributional
  distance against a minibatch of real observations.
* `JointPoseRecon`: treats each real particle's pose as an extra free parameter,
  jointly optimized with X to minimize direct per-particle reprojection error
  (bundle-adjustment / classical ab-initio SPA style).

New methods register by subclassing `ReconMethod` and adding one line to
`RECON_METHODS`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corruption import render_projection


class ReconMethod:
    """Base interface. `init_state` builds any extra learnable state (e.g.
    per-particle poses); `parameters` exposes that state to the optimizer; `loss`
    computes the scalar training loss for one minibatch."""

    def init_state(self, n_objects: int, device: torch.device) -> dict:
        return {}

    def parameters(self, state: dict) -> list[nn.Parameter]:
        return []

    def loss(
        self,
        render_args: tuple[torch.Tensor, torch.Tensor],  # (positions, weights) of current X
        render_kwargs: dict,
        y_batch: torch.Tensor,     # (B, T, P, P)
        idx_batch: torch.Tensor,   # (B,) indices into the full dataset
        state: dict,
    ) -> torch.Tensor:
        raise NotImplementedError


# ── distributional ────────────────────────────────────────────────────────────


def _sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """a, b: (B, D) flattened samples -> scalar SWD (mean squared gap over
    `n_proj` random 1D projections, each direction's samples sorted then compared)."""
    d = a.shape[-1]
    dirs = torch.randn(n_proj, d, device=a.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    pa = torch.sort(a @ dirs.T, dim=0).values   # (B, n_proj)
    pb = torch.sort(b @ dirs.T, dim=0).values
    return (pa - pb).pow(2).mean()


def _mmd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (B, D) -> scalar MMD^2 with an RBF kernel (median-heuristic bandwidth)."""
    xy = torch.cat([a, b], dim=0)
    d2 = torch.cdist(xy, xy).pow(2)
    sigma2 = d2.detach().median().clamp_min(1e-6)
    k = torch.exp(-d2 / (2 * sigma2))
    n = a.shape[0]
    k_aa, k_bb, k_ab = k[:n, :n], k[n:, n:], k[:n, n:]
    return k_aa.mean() + k_bb.mean() - 2 * k_ab.mean()


DIST_METRICS = {"sliced_wasserstein": _sliced_wasserstein, "mmd": _mmd}


class DistributionalRecon(ReconMethod):
    def __init__(self, dist_metric: str = "sliced_wasserstein"):
        self.metric = DIST_METRICS[dist_metric]

    def loss(self, render_args, render_kwargs, y_batch, idx_batch, state):
        positions, weights = render_args
        B = y_batch.shape[0]
        y_sim = render_projection(positions, weights, batch=B, **render_kwargs)  # (B,T,P,P)
        return self.metric(y_sim.reshape(B, -1), y_batch.reshape(B, -1))


# ── joint pose ─────────────────────────────────────────────────────────────────


def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) proper rotation via Gram-Schmidt (Zhou et al. 2019,
    "On the Continuity of Rotation Representations in Neural Networks") -- smooth,
    unconstrained, SGD-friendly (no gimbal lock / quaternion double-cover)."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class JointPoseRecon(ReconMethod):
    def init_state(self, n_objects: int, device: torch.device) -> dict:
        init = torch.zeros(n_objects, 6, device=device)
        init[:, 0] = 1.0  # identity-biased: first two rotation-matrix columns
        init[:, 4] = 1.0
        return {"pose6d": nn.Parameter(init)}

    def parameters(self, state: dict) -> list[nn.Parameter]:
        return [state["pose6d"]]

    def loss(self, render_args, render_kwargs, y_batch, idx_batch, state):
        positions, weights = render_args
        R = _rotation_6d_to_matrix(state["pose6d"][idx_batch])  # (B, 3, 3)
        # No extra noise on the simulated side: this is a direct least-squares fit
        # against noisy targets, so the optimal prediction is the noise-free mean.
        kwargs = dict(render_kwargs, noise_std=0.0)
        y_sim = render_projection(positions, weights, R=R, **kwargs)
        return (y_sim - y_batch).pow(2).mean()


RECON_METHODS = {"distributional": DistributionalRecon, "joint_pose": JointPoseRecon}


def get_recon_method(name: str, **kwargs) -> ReconMethod:
    return RECON_METHODS[name](**kwargs)
