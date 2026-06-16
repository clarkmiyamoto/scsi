"""Permutation-equivariant velocity field v_theta(X, t) for point-cloud flow matching.

A point cloud is X in R^{N x 3} (a *set* of N points). The network must be
permutation-equivariant: permuting the input rows permutes the output rows the
same way. We get that from per-point MLPs + self-attention with NO positional
encoding (attention over a set is permutation-equivariant by construction).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


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


class SetBlock(nn.Module):
    """Pre-norm transformer block over a SET of tokens (no positional encoding).

    Self-attention mixes information across the N points; the per-point MLP
    transforms each point's feature. Both are permutation-equivariant.
    """

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, N, dim)
        x = self.norm1(h)
        a, _ = self.attn(x, x, x, need_weights=False)  # uses fused SDPA / flash on CUDA
        h = h + a
        h = h + self.mlp(self.norm2(h))
        return h


class PointCloudVelocity(nn.Module):
    """v_theta(X, t): (B, N, 3) x (B,) -> (B, N, 3), permutation-equivariant."""

    def __init__(self, dim: int = 128, depth: int = 6, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(3, dim)                       # per-point embed
        self.t_proj = nn.Sequential(                           # time -> feature
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([SetBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 3)                      # per-point velocity
        # Start as (near) the identity-ish zero field -> stabler early training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), t: (B,)
        h = self.in_proj(x)                                    # (B, N, dim)
        temb = self.t_proj(timestep_embedding(t, self.dim))    # (B, dim)
        h = h + temb[:, None, :]                               # add time to every point
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(self.out_norm(h))                 # (B, N, 3)


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


class CrossAttnSetBlock(nn.Module):
    """Pre-norm block: self-attention over points + cross-attention to image tokens + MLP.

    Self-attention mixes points (permutation-equivariant); cross-attention lets
    each point read from the conditioning image; the per-point MLP transforms.
    """

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # h: (B, N, dim) points, ctx: (B, M, dim) image tokens
        x = self.norm1(h)
        a, _ = self.self_attn(x, x, x, need_weights=False)
        h = h + a
        q, kv = self.norm_q(h), self.norm_kv(ctx)
        c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h = h + c
        h = h + self.mlp(self.norm2(h))
        return h


class ConditionalPointCloudVelocity(nn.Module):
    """v_theta(X, t | y): (B, N, 3) x (B,) x (B, 1, P, P) -> (B, N, 3).

    Permutation-equivariant over the N points, conditioned on a 2D image y via
    cross-attention to its patch tokens.
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
        self.in_proj = nn.Linear(3, dim)
        self.t_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.img_encoder = ImagePatchEncoder(image_size, patch_size, dim)
        self.blocks = nn.ModuleList([CrossAttnSetBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 3)
        # Zero-init output head -> (near) zero velocity field early -> stabler training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), t: (B,), y: (B, 1, P, P)
        h = self.in_proj(x)
        temb = self.t_proj(timestep_embedding(t, self.dim))
        h = h + temb[:, None, :]
        ctx = self.img_encoder(y)                              # (B, n_patches, dim)
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.out_proj(self.out_norm(h))                 # (B, N, 3)
