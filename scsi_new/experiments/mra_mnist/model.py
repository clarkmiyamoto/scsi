import torch
import torch.nn as nn
from diffusers import DiTTransformer2DModel

IMAGE_SIZE: int = 28
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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        # x_t: (B, 1, H, W)      interpolated image state I_t^x
        # t:   (B,) or (B,1,1,1) fractional time in [0, 1] -- scaled to DiT's integer
        #      timestep embedding range here, per the repo-wide INTEGRATION_SCALE convention.
        # y:   (B, 1, H, W)      MRA observation (a rotated copy of x, same shape as x_t)
        t_frac = t.reshape(-1)
        t_int = (t_frac * INTEGRATION_SCALE).long()
        inp = torch.cat([x_t, y], dim=1)
        dummy = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t_int, class_labels=dummy).sample