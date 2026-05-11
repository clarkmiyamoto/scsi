import torch
import torch.nn as nn
# from diffusers.models import Transformer3DModel
from diffusers import UNet3DConditionModel

VOL_SIZE: int = 32
INTEGRATION_SCALE: float = 999


# class ConditionalDiT3D(nn.Module):
#     """
#     3D velocity-field network using Transformer3DModel from diffusers.

#     Applies per-slice spatial self-attention + TransformerTemporalModel
#     temporal attention across the depth axis for inter-slice reasoning.

#     Conditioning: 2D projection y is tiled along depth and channel-concatenated
#     with the interpolated volume I_t, mirroring the 2D implementation.

#     Input:
#         x_t:  (B, 1, D, H, W)   interpolated volume I_t
#         t:    (B,)               integer in [0, INTEGRATION_SCALE]
#         cond: (B, 1, H, W)       2D projection observation y
#     Output:
#         (B, 1, D, H, W)          predicted velocity dI_t/dt
#     """

#     def __init__(
#         self,
#         vol_size: int = VOL_SIZE,
#         hidden: int = 384,
#         depth: int = 6,
#         heads: int = 6,
#         norm_num_groups: int = 8,
#     ):
#         super().__init__()
#         assert hidden % heads == 0
#         self.vol_size = vol_size
#         self.transformer = Transformer3DModel(
#             num_attention_heads=heads,
#             attention_head_dim=hidden // heads,
#             in_channels=2,
#             out_channels=1,
#             num_layers=depth,
#             norm_num_groups=norm_num_groups,
#             num_embeds_ada_norm=1000,
#         )

#     def forward(
#         self,
#         x_t: torch.Tensor,     # (B, 1, D, H, W)
#         t: torch.Tensor,        # (B,) long in [0, 999]
#         cond: torch.Tensor,     # (B, 1, H, W)
#     ) -> torch.Tensor:
#         B, _, D, H, W = x_t.shape

#         y_3d = cond.unsqueeze(2).expand(-1, -1, D, -1, -1)    # (B, 1, D, H, W)
#         inp = torch.cat([x_t, y_3d], dim=1)                    # (B, 2, D, H, W)

#         # Transformer3DModel expects (B*D, C, H, W); temporal blocks use num_frames=D
#         inp_flat = inp.permute(0, 2, 1, 3, 4).reshape(B * D, 2, H, W)
#         t_flat = t.repeat_interleave(D)                        # (B*D,)

#         out = self.transformer(inp_flat, timestep=t_flat, num_frames=D).sample
#         # (B*D, 1, H, W) -> (B, 1, D, H, W)
#         return out.reshape(B, D, 1, H, W).permute(0, 2, 1, 3, 4).contiguous()


class ConditionalUNet3D(nn.Module):
    """
    3D velocity-field network using UNet3DConditionModel from diffusers.

    Input/output shapes are identical to ConditionalDiT3D.
    """

    def __init__(
        self,
        vol_size: int = VOL_SIZE,
        block_out_channels: tuple = (64, 128, 256, 256),
        layers_per_block: int = 2,
        norm_num_groups: int = 8,
    ):
        super().__init__()
        self.unet = UNet3DConditionModel(
            sample_size=vol_size,
            in_channels=2,
            out_channels=1,
            down_block_types=tuple("DownBlock3D" for _ in block_out_channels),
            up_block_types=tuple("UpBlock3D" for _ in block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=block_out_channels[0],
            attention_head_dim=8,
            norm_num_groups=norm_num_groups,
        )

    def forward(
        self,
        x_t: torch.Tensor,     # (B, 1, D, H, W)
        t: torch.Tensor,        # (B,) long in [0, 999]
        cond: torch.Tensor,     # (B, 1, H, W)
    ) -> torch.Tensor:
        B, _, D, H, W = x_t.shape

        y_3d = cond.unsqueeze(2).expand(-1, -1, D, -1, -1)
        inp = torch.cat([x_t, y_3d], dim=1)                    # (B, 2, D, H, W)

        # Mid block requires encoder_hidden_states; zeros -> effective self-attention
        dummy = torch.zeros(
            B, 1, self.unet.config.cross_attention_dim,
            device=x_t.device, dtype=x_t.dtype,
        )
        return self.unet(inp, timestep=t, encoder_hidden_states=dummy).sample


def build_model(arch: str, small: bool = False, vol_size: int = VOL_SIZE) -> nn.Module:
    """
    Factory function.

    Args:
        arch:  "dit3d" or "unet3d"
        small: use smaller channel counts for debugging
    """
    if arch == "dit3d":
        if small:
            return ConditionalDiT3D(vol_size=vol_size, hidden=192, depth=4, heads=4)
        return ConditionalDiT3D(vol_size=vol_size)
    elif arch == "unet3d":
        if small:
            return ConditionalUNet3D(vol_size=vol_size,
                                     block_out_channels=(32, 64, 128),
                                     layers_per_block=1)
        return ConditionalUNet3D(vol_size=vol_size)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'dit3d' or 'unet3d'.")
