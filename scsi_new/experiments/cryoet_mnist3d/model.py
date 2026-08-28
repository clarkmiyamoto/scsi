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

import torch
import torch.nn as nn
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
