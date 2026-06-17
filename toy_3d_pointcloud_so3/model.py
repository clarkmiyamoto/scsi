"""SO(3)-equivariant velocity fields for point-cloud flow matching (Vector Neurons).

A point cloud is X in R^{N x 3} (a *set* of N points). The networks here are built
on a **two-stream** backbone:

  * an invariant **scalar** stream  ``h : (B, N, S)``  (carries rotation-invariant
    features + time, mixes points with ordinary attention),
  * an equivariant **vector** stream ``V : (B, N, C, 3)`` (carries direction, built
    from Vector Neuron layers in ``vn.py``).

Both streams are **permutation-equivariant** over the N points (attention with no
positional encoding). The vector stream is additionally **SO(3)-equivariant**: the
scalar stream depends only on rotation-invariants of ``V`` (and on ``t``), every
vector op commutes with ``V @ R^T``, and the output is read out *from the vector
stream only*. Hence the unconditional field satisfies, exactly,

    f(X @ R^T, t) == f(X, t) @ R^T            (rotation convention of corruption.py)

so the generated distribution is rotation-invariant.

The **conditional** field ``f(X, t | y)`` additionally injects a 2D image
observation ``y`` into the scalar stream via cross-attention. The image fixes a
pose, so this field is *intentionally not* SO(3)-equivariant -- it must orient its
reconstruction to match the observed projection. The Vector Neuron backbone still
provides the rotation-aware geometric inductive bias for the per-point geometry.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .vn import VNLayerNorm, VNLeakyReLU, VNLinear, safe_norm, scalar_gate, vn_norm


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of scalar time.

    Args:
        t:   (B,) tensor of times in [0, 1].
        dim: embedding width.
    Returns:
        (B, dim) tensor.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None] * 1000.0  # scale so [0,1] spans the frequency range
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad if dim is odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _vec_channels(dim: int, heads: int) -> int:
    """Number of vector channels C: ~dim/4, rounded up to a multiple of `heads`.

    Vector-stream attention splits C across `heads`, so C must be divisible by it.
    """
    c = max(dim // 4, heads)
    return ((c + heads - 1) // heads) * heads


def encode_inputs(
    x: torch.Tensor, vec_in: VNLinear, scalar_in: nn.Linear
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift raw coordinates into (scalar, vector) features.

    Vector stream: ``[x, x - centroid]`` (2 equivariant channels) -> ``vec_in``.
    Scalar stream: ``[vn_norm(V0), ||x||, ||x - centroid||]`` (all invariant) ->
    ``scalar_in``. Both transform correctly: ``x`` and ``x - centroid`` rotate as
    ``@ R^T``, and the scalar inputs are rotation-invariant.

    Returns:
        h: (B, N, dim) invariant scalar features.
        V: (B, N, C, 3) equivariant vector features.
    """
    c = x.mean(dim=1, keepdim=True)                              # (B, 1, 3) centroid
    xc = x - c                                                  # (B, N, 3)
    V = vec_in(torch.stack([x, xc], dim=-2))                    # (B, N, C, 3)
    nx = safe_norm(x, dim=-1, keepdim=True).to(x.dtype)        # (B, N, 1) invariant
    nxc = safe_norm(xc, dim=-1, keepdim=True).to(x.dtype)
    s = torch.cat([vn_norm(V).to(x.dtype), nx, nxc], dim=-1)    # (B, N, C + 2)
    return scalar_in(s), V


class VNBlockBase(nn.Module):
    """Shared machinery for the (un)conditional two-stream blocks.

    Three equivariance-preserving stages, used by the forwards of the subclasses:
      * ``_self_attn``    : ordinary attention over the scalar stream (perm-equiv).
      * ``_vector_mix``   : aggregate the vector stream with *invariant* attention
                            weights derived from the scalar stream.
      * ``_cross_stream`` : scalar reads vector norms (invariant); vector gets a
                            VN-MLP update gated by invariant scalars from h.
    """

    def __init__(self, dim: int, channels: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.dh = dim // heads

        # scalar self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # invariant attention weights for the vector stream
        self.normqk = nn.LayerNorm(dim)
        self.to_qk = nn.Linear(dim, 2 * dim, bias=False)
        self.vec_val = VNLinear(channels, channels)
        self.vec_proj = VNLinear(channels, channels)

        # cross-stream + per-point MLP / VN-MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim + channels, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.vn_ln = VNLayerNorm(channels)
        self.vec_lin = VNLinear(channels, channels)
        self.vec_act = VNLeakyReLU(channels)
        self.gate = nn.Linear(dim, channels)

    def _self_attn(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h)
        a, _ = self.self_attn(x, x, x, need_weights=False)
        return h + a

    def _vector_mix(self, h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        q, k = self.to_qk(self.normqk(h)).chunk(2, dim=-1)
        q = q.view(B, N, self.heads, self.dh).transpose(1, 2)       # (B, h, N, dh)
        k = k.view(B, N, self.heads, self.dh).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)  # (B,h,N,N)

        cph = self.channels // self.heads
        Vv = self.vec_val(V).view(B, N, self.heads, cph, 3).permute(0, 2, 1, 3, 4)  # (B,h,N,cph,3)
        # weighted sum of vectors with invariant weights -> equivariant aggregation
        agg = torch.einsum("bhij,bhjcd->bhicd", attn.to(Vv.dtype), Vv)
        agg = agg.permute(0, 2, 1, 3, 4).reshape(B, N, self.channels, 3)
        return V + self.vec_proj(agg)

    def _cross_stream(
        self, h: torch.Tensor, V: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = h + self.mlp(torch.cat([self.norm2(h), vn_norm(V).to(h.dtype)], dim=-1))
        g = torch.sigmoid(self.gate(h))                            # (B, N, C) invariant gate
        V = V + scalar_gate(self.vec_act(self.vec_lin(self.vn_ln(V))), g)
        return h, V


class VNSetBlock(VNBlockBase):
    """Unconditional two-stream block."""

    def forward(
        self, h: torch.Tensor, V: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._self_attn(h)
        V = self._vector_mix(h, V)
        return self._cross_stream(h, V)


class PointCloudVelocity(nn.Module):
    """v_theta(X, t): (B, N, 3) x (B,) -> (B, N, 3), SO(3)- & permutation-equivariant."""

    def __init__(self, dim: int = 128, depth: int = 6, heads: int = 4):
        super().__init__()
        self.dim = dim
        c = _vec_channels(dim, heads)
        self.vec_in = VNLinear(2, c)                              # [x, x-centroid] -> C
        self.scalar_in = nn.Linear(c + 2, dim)                   # invariants -> scalar
        self.t_proj = nn.Sequential(                             # time -> feature
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([VNSetBlock(dim, c, heads) for _ in range(depth)])
        self.out_ln = VNLayerNorm(c)
        self.out_proj = VNLinear(c, 1)                           # vector readout -> velocity
        # Start as the zero field -> stabler early training (matches the baseline).
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), t: (B,)
        h, V = encode_inputs(x, self.vec_in, self.scalar_in)
        h = h + self.t_proj(timestep_embedding(t, self.dim))[:, None, :]
        for blk in self.blocks:
            h, V = blk(h, V)
        return self.out_proj(self.out_ln(V)).squeeze(-2)         # (B, N, 3)


# ── Conditional variant: condition the set on a 2D image observation y ──────────


class ImagePatchEncoder(nn.Module):
    """Encode a 2D image y -> a set of patch tokens for cross-attention.

    Unlike the point set (permutation-invariant, no positional encoding), image
    patches carry spatial meaning, so we add a learned positional embedding.
    """

    def __init__(self, image_size: int, patch_size: int, dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch_size {patch_size}")
        self.proj = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # y: (B, 1, P, P)
        h = self.proj(y)                       # (B, dim, P/ps, P/ps)
        h = h.flatten(2).transpose(1, 2)       # (B, n_patches, dim)
        return h + self.pos


class VNCrossAttnSetBlock(VNBlockBase):
    """Conditional two-stream block: scalar stream also cross-attends to image tokens.

    The (pose-dependent) image information enters via the scalar stream, which then
    gates / updates the vector stream in ``_cross_stream`` -- the deliberate symmetry
    break that aligns the reconstruction to the observed projection.
    """

    def __init__(self, dim: int, channels: int, heads: int, mlp_ratio: int = 4):
        super().__init__(dim, channels, heads, mlp_ratio)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(
        self, h: torch.Tensor, V: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._self_attn(h)
        q, kv = self.norm_q(h), self.norm_kv(ctx)
        c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h = h + c
        V = self._vector_mix(h, V)
        return self._cross_stream(h, V)


class ConditionalPointCloudVelocity(nn.Module):
    """v_theta(X, t | y): (B, N, 3) x (B,) x (B, 1, P, P) -> (B, N, 3).

    Vector Neuron backbone over the N points (permutation-equivariant), conditioned
    on a 2D image y via cross-attention to its patch tokens. The conditioning fixes
    a pose, so this field is intentionally *not* SO(3)-equivariant (see module docs).
    """

    def __init__(
        self,
        dim: int = 128,
        depth: int = 6,
        heads: int = 4,
        image_size: int = 32,
        patch_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        c = _vec_channels(dim, heads)
        self.vec_in = VNLinear(2, c)
        self.scalar_in = nn.Linear(c + 2, dim)
        self.t_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.img_encoder = ImagePatchEncoder(image_size, patch_size, dim)
        self.blocks = nn.ModuleList(
            [VNCrossAttnSetBlock(dim, c, heads) for _ in range(depth)]
        )
        self.out_ln = VNLayerNorm(c)
        self.out_proj = VNLinear(c, 1)
        # Zero-init output head -> (near) zero velocity field early -> stabler training.
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), t: (B,), y: (B, 1, P, P)
        h, V = encode_inputs(x, self.vec_in, self.scalar_in)
        h = h + self.t_proj(timestep_embedding(t, self.dim))[:, None, :]
        ctx = self.img_encoder(y)                                # (B, n_patches, dim)
        for blk in self.blocks:
            h, V = blk(h, V, ctx)
        return self.out_proj(self.out_ln(V)).squeeze(-2)         # (B, N, 3)
