"""Permutation-equivariant conditional velocity field b_t(X, t | y) for point clouds.

A point cloud is ``X in R^{N x 3}`` (a *set* of N points), so the network must be
permutation-equivariant: permuting the input rows permutes the output rows the same
way. We get that from per-point MLPs + self-attention with NO positional encoding.
The cloud is conditioned on the CryoET observation ``y`` (a K-channel tilt series)
via cross-attention to its image patch tokens.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of scalar time ``t`` (B,) in [0,1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None] * 1000.0  # scale so [0,1] spans the frequency range
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ImagePatchEncoder(nn.Module):
    """Encode an observation ``y`` (B, C, P, P) -> patch tokens for cross-attention.

    Image patches carry spatial meaning, so a learned positional embedding is added.
    ``in_channels`` = K stacks a CryoET tilt series as conv input channels: the K
    projections fold into the patch embedding (token count -- and thus cross-attention
    cost -- is unchanged vs a single image).
    """

    def __init__(self, image_size: int, patch_size: int, dim: int, in_channels: int = 1):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch_size {patch_size}")
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # y: (B, C, P, P)
        h = self.proj(y)                       # (B, dim, P/ps, P/ps)
        h = h.flatten(2).transpose(1, 2)       # (B, n_patches, dim)
        return h + self.pos


class CrossAttnSetBlock(nn.Module):
    """Pre-norm: self-attn over points + cross-attn to image tokens + per-point MLP."""

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
        # h: (B, N, dim) points; ctx: (B, M, dim) image tokens
        x = self.norm1(h)
        a, _ = self.self_attn(x, x, x, need_weights=False)
        h = h + a
        q, kv = self.norm_q(h), self.norm_kv(ctx)
        c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h = h + c
        h = h + self.mlp(self.norm2(h))
        return h


class ConditionalPointCloudVelocity(nn.Module):
    """b_t(X, t | y): (B, N, 3) x (B,) x (B, C, P, P) -> (B, N, 3), permutation-equivariant."""

    def __init__(
        self,
        dim: int = 128,
        depth: int = 6,
        heads: int = 4,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(3, dim)
        self.t_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.img_encoder = ImagePatchEncoder(image_size, patch_size, dim, in_channels)
        self.blocks = nn.ModuleList([CrossAttnSetBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 3)
        # Zero-init output head -> (near) zero velocity early -> stabler training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), t: (B,), y: (B, C, P, P)
        h = self.in_proj(x)
        temb = self.t_proj(timestep_embedding(t, self.dim))
        h = h + temb[:, None, :]                               # add time to every point
        ctx = self.img_encoder(y)                              # (B, n_patches, dim)
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.out_proj(self.out_norm(h))                 # (B, N, 3)


# ── Config / factory ──────────────────────────────────────────────────────────


@dataclass
class ConditionalModelConfig:
    dim: int = 128
    depth: int = 6
    heads: int = 4
    n_points: int = 512
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 11  # = n_tilts (K) for CryoET


def build_conditional_model(
    cfg: ConditionalModelConfig, device: torch.device
) -> ConditionalPointCloudVelocity:
    return ConditionalPointCloudVelocity(
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
    ).to(device)


# ── EMA over the outer EM loop ─────────────────────────────────────────────────


def clone_ema(model: nn.Module) -> nn.Module:
    """Frozen EMA copy: deep-copied, eval mode, no grad. Initializes Theta_EMA <- Theta."""
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update_outer(model_ema: nn.Module, model: nn.Module, gamma: float) -> None:
    """Theta_EMA^(k) <- gamma * Theta_EMA^(k-1) + (1 - gamma) * Theta^(k).

    Called once per outer EM iteration (the EMA is frozen during the inner loop).
    """
    for p_ema, p in zip(model_ema.parameters(), model.parameters()):
        p_ema.lerp_(p.detach(), 1.0 - gamma)
