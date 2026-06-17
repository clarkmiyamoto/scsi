"""Vector Neuron (VN) primitives for SO(3)-equivariant point-cloud networks.

Vector Neurons (Deng et al., 2021, "Vector Neurons: A General Framework for
SO(3)-Equivariant Networks") lift each scalar neuron to a *vector* neuron: a
feature is a stack of C three-vectors rather than C scalars.

    V : (B, N, C, 3)        C channels, each a 3-vector, per point

Rotation acts on the trailing 3-dim by right-multiplication with R^T, matching
the convention used by the corruption channel (``points @ R.T`` in
``corruption.forward_channel``):

    rotate(V) = V @ R^T          (R a 3x3 rotation matrix)

Every layer below is *equivariant*: ``layer(V @ R^T) == layer(V) @ R^T``. The
two ways to read information out without breaking equivariance are
  * invariant scalars via ``vn_norm`` (channel-wise norms, rotation-invariant),
  * equivariant vectors via ``VNLinear`` (linear mixing over the channel axis).

Norm / division ops upcast to fp32 and add an ``eps`` so the layers stay stable
under the ``torch.amp.autocast`` paths used in ``flow.py`` / ``scsi.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def _hi(dtype: torch.dtype) -> torch.dtype:
    """Stable dtype for norms/divisions: upcast low precision, otherwise keep.

    Half/bfloat16 -> float32 (AMP stability); float32/float64 are left untouched
    so the float64 equivariance checks stay at full precision.
    """
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """L2 norm computed in a numerically stable dtype."""
    return torch.linalg.vector_norm(x.to(_hi(x.dtype)), dim=dim, keepdim=keepdim)


def vn_norm(V: torch.Tensor) -> torch.Tensor:
    """Channel-wise L2 norms -> invariant scalar features.

    Args:
        V: (B, N, C, 3) vector feature.
    Returns:
        (B, N, C) per-channel norms (stable dtype), rotation-invariant.
    """
    return safe_norm(V, dim=-1)


class VNLinear(nn.Module):
    """Equivariant linear map over the channel axis (no bias).

    ``out[..., o, :] = sum_i W[o, i] * V[..., i, :]``. A bias would add a constant
    3-vector and break equivariance, so there is none.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, V: torch.Tensor) -> torch.Tensor:  # V: (B, N, c_in, 3)
        return torch.einsum("bncd,oc->bnod", V, self.weight.to(V.dtype))


class VNLeakyReLU(nn.Module):
    """Equivariant leaky-ReLU.

    Learns a per-channel direction ``K`` (a ``VNLinear`` of the input). Where the
    feature's component along ``K`` is negative it is (leaky-)removed; where it is
    positive the feature passes through. Inner products and projections onto an
    equivariant direction are themselves equivariant.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.dir = VNLinear(channels, channels)
        self.negative_slope = negative_slope

    def forward(self, V: torch.Tensor) -> torch.Tensor:  # (B, N, C, 3)
        d = self.dir(V)
        hi = _hi(V.dtype)
        Vf, df = V.to(hi), d.to(hi)
        dot = (Vf * df).sum(dim=-1, keepdim=True)          # (B, N, C, 1) invariant
        d_sq = (df * df).sum(dim=-1, keepdim=True)
        # Branchless: subtract the (leaky) negative component along d. Continuous at
        # dot == 0, so equivariance is exact (no threshold-flip near the boundary).
        out = Vf - (1.0 - self.negative_slope) * dot.clamp(max=0.0) / (d_sq + EPS) * df
        return out.to(V.dtype)


class VNLayerNorm(nn.Module):
    """Equivariant layer norm: normalize per-channel *magnitudes*, keep directions.

    LayerNorm is applied to the (invariant) channel norms; a ``softplus`` maps the
    normalized values to non-negative magnitudes so directions are never flipped.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, V: torch.Tensor) -> torch.Tensor:  # (B, N, C, 3)
        norms = vn_norm(V)                                  # (B, N, C) stable dtype
        new_mag = F.softplus(self.ln(norms))               # >= 0, invariant
        scale = (new_mag / (norms + EPS)).unsqueeze(-1)    # (B, N, C, 1)
        return (V.to(norms.dtype) * scale).to(V.dtype)


def scalar_gate(V: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Scale vector channels by invariant per-channel scalars (equivariant).

    Args:
        V: (B, N, C, 3) vector feature.
        g: (B, N, C) invariant gate.
    """
    return V * g.unsqueeze(-1).to(V.dtype)
