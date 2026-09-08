"""
3D counterpart of cryoet_mnist/model.py. The velocity field b_t(x | y) for the 3D->2D CryoET
channel, backed by the diffusers UNet3DConditionModel.

UNet3DConditionModel is the ModelScope text-to-video UNet: a FACTORISED (2+1)D net, not a
volumetric one. `conv_in` is an nn.Conv2d -- internally it reshapes (B, C, D, H, W) ->
(B*D, C, H, W), applies per-slice 2D spatial convs with weights SHARED across depth, and mixes
information across the depth axis only through TransformerTemporalModel / temporal-conv blocks.
Depth is not spatially downsampled.

Because the 2D spatial convs share weights across every depth slice and only see their own
slice's channels, the T tilt-series projections are fed in as T EXTRA INPUT CHANNELS (each
projection tiled unchanged across all D slices), giving in_channels = 1 + num_tilts. Laying the
T projections out ALONG the depth axis instead -- the direct analogue of the 2D
broadcast_tilt_series row-tiling -- would hide all but ~D/T of them from any given slice's 2D
receptive field, since the only cross-slice path is the temporal block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet3DConditionModel

VOL_SIZE: int = 32
INTEGRATION_SCALE: float = 999


def stack_tilt_series(y: torch.Tensor, D: int) -> torch.Tensor:
    """
    Turn a raw tilt-series observation (B, T, 1, H, W) into a (B, T, D, H, W) conditioning
    stack: each of the T 2D projections is tiled unchanged across all D depth slices. The
    caller channel-concats this with the (B, 1, D, H, W) volume state, so every depth slice
    sees the whole tilt series through the shared per-slice 2D convs.

    Args:
        y: (B, T, 1, H, W)
        D: depth of the volume state (targets vol_size).

    Returns:
        (B, T, D, H, W)
    """
    y = y.squeeze(2)                                  # (B, T, H, W)
    return y.unsqueeze(2).expand(-1, -1, D, -1, -1)   # (B, T, D, H, W)


class ConditionalVelocityCryoET3D(nn.Module):
    """
    b_t(x_t | y) for the 3D->2D CryoET channel. in_channels = 1 (volume state) + num_tilts
    (one channel per tilt projection); out_channels = 1 (the velocity volume).
    """

    def __init__(self, vol_size: int = VOL_SIZE, num_tilts: int = 16,
                 block_out_channels: tuple[int, ...] = (64, 128, 256, 256),
                 layers_per_block: int = 2, norm_num_groups: int = 8):
        super().__init__()
        self.vol_size = vol_size
        self.num_tilts = num_tilts
        self.unet = UNet3DConditionModel(
            sample_size=vol_size,
            in_channels=1 + num_tilts,
            out_channels=1,
            down_block_types=tuple("DownBlock3D" for _ in block_out_channels),
            up_block_types=tuple("UpBlock3D" for _ in block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=block_out_channels[0],
            attention_head_dim=8,
            norm_num_groups=norm_num_groups,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x_t: (B, 1, D, H, W)   t: (B,) or (B,1,1,1,1) in [0, 1]   y: (B, T, 1, H, W)
        B, _, D, H, W = x_t.shape
        assert y.size(1) == self.num_tilts, (
            f"model built for num_tilts={self.num_tilts}, got a tilt series of length {y.size(1)}"
        )
        t_int = (t.reshape(-1) * INTEGRATION_SCALE).long()
        y_stack = stack_tilt_series(y, D=D)                       # (B, T, D, H, W)
        inp = torch.cat([x_t, y_stack], dim=1)                    # (B, 1 + T, D, H, W)
        # The mid block always carries a cross-attention layer; feed it zeros so it reduces to
        # plain self-attention (same trick as simple_3d/model.py).
        dummy = torch.zeros(B, 1, self.unet.config.cross_attention_dim,
                            device=x_t.device, dtype=x_t.dtype)
        return self.unet(inp, timestep=t_int, encoder_hidden_states=dummy).sample


# ---------------------------------------------------------------------------
# DiT backbone -- volumetric alternative to the diffusers video-UNet above.
# ---------------------------------------------------------------------------
#
# ConditionalVelocityCryoET3D wraps UNet3DConditionModel, which is FACTORISED (2+1)D: per-slice
# 2D convs sharing weights across depth, cross-depth mixing only in its temporal blocks (see the
# module docstring). ConditionalDiTCryoET3D below is the honestly-volumetric counterpart: it
# 3D-patchifies the (1 + num_tilts)-channel input into (vol_size / patch_size)**3 tokens and runs
# FULL 3D self-attention over them -- every token attends to every other, so depth is treated
# exactly like the in-plane axes. Timestep conditioning is adaLN-Zero (Peebles & Xie, "Scalable
# Diffusion Models with Transformers"): each block and the final projection start at the identity
# / zero, so an untrained net predicts a ~0 velocity field.
#
# This is a self-contained implementation on purpose. diffusers' `Transformer3DModel` is *also*
# the (2+1)D ModelScope block (spatial self-attention + a temporal block), i.e. the same
# factorisation as the UNet -- simple_3d/model.py's `ConditionalDiT3D` tried it and is now
# dead/commented-out code. Nothing in `diffusers` 0.37 provides a true 3D-patch DiT.
#
# Interface parity with ConditionalVelocityCryoET3D: identical forward(x_t, t, y) signature,
# identical (B, 1, D, H, W) output, identical tilt conditioning (stack_tilt_series -> channel
# concat), and t is quantised to INTEGRATION_SCALE integer bins exactly as the UNet's `.long()`
# does -- so an ODE-step / interpolant sweep measures the same thing on both backbones.


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B, N, C);  shift, scale: (B, C)  ->  broadcast over the token axis.
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the (already INTEGRATION_SCALE-scaled, integer-binned) timestep,
    then a 2-layer MLP up to the transformer width."""

    def __init__(self, hidden: int, freq_dim: int = 256, max_period: int = 10_000):
        super().__init__()
        assert freq_dim % 2 == 0
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, INTEGRATION_SCALE]
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, freq_dim)
        return self.mlp(emb)


class _Attention(nn.Module):
    """Plain multi-head self-attention over the full token set (no mask, no bias term on the
    scaled-dot-product) via F.scaled_dot_product_attention."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        assert hidden % heads == 0, "dit_hidden must be divisible by dit_heads"
        self.heads = heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B, heads, N, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)     # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning: the timestep embedding produces per-block
    (shift, scale, gate) for both the attention and the MLP sub-layer; `gate` is zero-init so the
    block is the identity at start of training."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _FinalLayer(nn.Module):
    """adaLN-modulated LayerNorm + a zero-init linear projecting each token back to its
    out_channels * patch_size**3 voxel block."""

    def __init__(self, hidden: int, patch_numel: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch_numel)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class _DiT3D(nn.Module):
    """3D-patchify -> N-token adaLN-Zero transformer -> unpatchify. Operates on a plain
    (B, in_channels, D, H, W) tensor; the CryoET-specific channel packing lives in the
    ConditionalDiTCryoET3D wrapper."""

    def __init__(self, vol_size: int, in_channels: int, out_channels: int = 1,
                 patch_size: int = 4, hidden: int = 384, depth: int = 12, heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        assert vol_size % patch_size == 0, "vol_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid = vol_size // patch_size                 # tokens per axis
        num_patches = self.grid ** 3
        self.patch_numel = out_channels * patch_size ** 3

        self.x_embed = nn.Conv3d(in_channels, hidden, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden))
        self.t_embed = _TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([_DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = _FinalLayer(hidden, self.patch_numel)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        nn.init.zeros_(self.x_embed.bias)
        for lin in (self.t_embed.mlp[0], self.t_embed.mlp[2]):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        # adaLN-Zero: every conditioning projection and the output projection start at zero.
        for blk in self.blocks:
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, grid**3, out_channels * patch_size**3)  ->  (B, out_channels, D, H, W).
        # Token axis unrolls (gd, gh, gw) row-major, matching Conv3d(...).flatten(2); the per-token
        # vector unrolls (id, ih, iw, c) row-major, matching nn.Linear's output layout.
        B = x.shape[0]
        G, p, oc = self.grid, self.patch_size, self.out_channels
        x = x.reshape(B, G, G, G, p, p, p, oc)
        x = torch.einsum("b d h w i j k c -> b c d i h j w k", x)
        return x.reshape(B, oc, G * p, G * p, G * p)

    def forward(self, inp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # inp: (B, in_channels, D, H, W)   t: (B,) in [0, INTEGRATION_SCALE]
        tok = self.x_embed(inp).flatten(2).transpose(1, 2)   # (B, grid**3, hidden)
        tok = tok + self.pos_embed
        c = self.t_embed(t)
        for blk in self.blocks:
            tok = blk(tok, c)
        return self.unpatchify(self.final(tok, c))


class ConditionalDiTCryoET3D(nn.Module):
    """b_t(x_t | y) for the 3D->2D CryoET channel, DiT backbone. Drop-in alternative to
    ConditionalVelocityCryoET3D: identical forward(x_t, t, y) signature and (B, 1, D, H, W)
    output. in_channels = 1 (volume state) + num_tilts (one channel per tilt projection, tiled
    across depth by stack_tilt_series -- same conditioning the UNet path uses)."""

    def __init__(self, vol_size: int = VOL_SIZE, num_tilts: int = 16, patch_size: int = 4,
                 hidden: int = 384, depth: int = 12, heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.vol_size = vol_size
        self.num_tilts = num_tilts
        self.dit = _DiT3D(
            vol_size=vol_size, in_channels=1 + num_tilts, out_channels=1,
            patch_size=patch_size, hidden=hidden, depth=depth, heads=heads, mlp_ratio=mlp_ratio,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x_t: (B, 1, D, H, W)   t: (B,) or (B,1,1,1,1) in [0, 1]   y: (B, T, 1, H, W)
        D = x_t.size(2)
        assert y.size(1) == self.num_tilts, (
            f"model built for num_tilts={self.num_tilts}, got a tilt series of length {y.size(1)}"
        )
        # Match ConditionalVelocityCryoET3D's timestep binning exactly (`.long()` on t * 999).
        t_int = (t.reshape(-1) * INTEGRATION_SCALE).long()
        y_stack = stack_tilt_series(y, D=D)                       # (B, T, D, H, W)
        inp = torch.cat([x_t, y_stack], dim=1)                    # (B, 1 + T, D, H, W)
        return self.dit(inp, t_int)


def build_velocity_model(arch: str, *, vol_size: int = VOL_SIZE, num_tilts: int = 16,
                         block_out_channels: tuple[int, ...] = (64, 128, 256, 256),
                         layers_per_block: int = 2, patch_size: int = 4, dit_hidden: int = 384,
                         dit_depth: int = 12, dit_heads: int = 6,
                         dit_mlp_ratio: float = 4.0) -> nn.Module:
    """Factory for the conditional velocity net b_t(.|y).

    arch="unet" -> ConditionalVelocityCryoET3D  (diffusers video-UNet; the original backbone,
                   and the default so main.py's construction is unaffected).
    arch="dit"  -> ConditionalDiTCryoET3D       (volumetric DiT, defined above).

    block_out_channels / layers_per_block are read only for "unet"; patch_size / dit_* only for
    "dit". Both branches return a module with the same forward(x_t, t, y) contract.
    """
    if arch == "unet":
        return ConditionalVelocityCryoET3D(
            vol_size=vol_size, num_tilts=num_tilts,
            block_out_channels=tuple(block_out_channels), layers_per_block=layers_per_block,
        )
    if arch == "dit":
        return ConditionalDiTCryoET3D(
            vol_size=vol_size, num_tilts=num_tilts, patch_size=patch_size,
            hidden=dit_hidden, depth=dit_depth, heads=dit_heads, mlp_ratio=dit_mlp_ratio,
        )
    raise ValueError(f"Unknown arch: {arch!r}. Choose 'unet' or 'dit'.")
