import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel, UNet2DModel

IMAGE_SIZE: int = 32
VOL_SIZE: int = 32
INTEGRATION_SCALE: float = 999

class ConditionalDiT(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, patch_size=4,
                 hidden=192, depth=6, heads=6):
        super().__init__()
        self.dit = DiTTransformer2DModel(
            sample_size=image_size,
            patch_size=patch_size,
            in_channels=2,
            out_channels=1,
            num_layers=depth,
            num_attention_heads=heads,
            attention_head_dim=hidden // heads,
            num_embeds_ada_norm=1000,
        )

    def forward(self, x_t: torch.Tensor, t_int: torch.Tensor,
                y_broadcast: torch.Tensor) -> torch.Tensor:
        # x_t:         (B, 1, H, W)  interpolated image state I_t^x
        # t_int:       (B,)          integer in [0, INTEGRATION_SCALE]
        # y_broadcast: (B, 1, H, W)  1D observation tiled across rows
        inp = torch.cat([x_t, y_broadcast], dim=1)
        dummy = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t_int, class_labels=dummy).sample


class ConditionalUNet2D(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE,
                 block_out_channels=(64, 128, 128, 256),
                 layers_per_block=2, norm_num_groups=8):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=2,
            out_channels=1,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=tuple("DownBlock2D" for _ in block_out_channels),
            up_block_types=tuple("UpBlock2D" for _ in block_out_channels),
            norm_num_groups=norm_num_groups,
        )

    def forward(self, x_t: torch.Tensor, t_int: torch.Tensor,
                y_broadcast: torch.Tensor) -> torch.Tensor:
        # x_t:         (B, 1, H, W)  interpolated image state I_t^x
        # t_int:       (B,)          integer in [0, INTEGRATION_SCALE]
        # y_broadcast: (B, 1, H, W)  1D observation tiled across rows
        inp = torch.cat([x_t, y_broadcast], dim=1)
        return self.unet(inp, timestep=t_int).sample


def broadcast_tilt_series(y: torch.Tensor, H: int) -> torch.Tensor:
    """
    Reduce a raw tilt-series observation (B, T, 1, W) to a single (B, 1, H, W) conditioning
    channel that ConditionalDiT / ConditionalUNet2D can consume unmodified (in_channels=2, same
    as the single-projection case).

    Args:
        y: (B, T, 1, W)
        H: target row count (image_size)

    Returns:
        (B, 1, H, W)
    """
    B, T, _, W = y.shape
    y = y.squeeze(2)  # (B, T, W)
    if H % T == 0:
        y_broadcast = y.repeat_interleave(H // T, dim=1)  # (B, H, W), exact tiling
    else:
        # General fallback (H not a multiple of T): nearest-neighbor resample of the T-row
        # "image" up to H rows.
        y_broadcast = F.interpolate(y.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
    return y_broadcast.unsqueeze(1)  # (B, 1, H, W)


class ConditionalVelocityCryoET(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, arch: str = "dit", patch_size: int = 4):
        super().__init__()
        self.image_size = image_size
        if arch == "dit":
            self.image_branch = ConditionalDiT(image_size=image_size, patch_size=patch_size)
        elif arch == "unet":
            self.image_branch = ConditionalUNet2D(image_size=image_size)
        else:
            raise ValueError(f"Unknown arch: {arch!r}. Choose 'dit' or 'unet'.")

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x_t: (B, 1, H, W)   t: (B,) or (B,1,1,1) in [0, 1]   y: (B, T, 1, W)
        t_frac = t.reshape(-1)
        y_broadcast = broadcast_tilt_series(y, H=x_t.size(-2))
        t_int = (t_frac * INTEGRATION_SCALE).long()
        return self.image_branch(x_t, t_int, y_broadcast)