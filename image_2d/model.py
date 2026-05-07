from diffusers import DiTTransformer2DModel
import torch
import torch.nn as nn

IMAGE_SIZE: int = 32
INTEGRATION_SCALE: float = 999
ELL_SCALE: int = 999

class ConditionalDiT(nn.Module):
    """
    Input:  cat([I_t, Y], dim=1)  ->  2 channels
    Output: velocity prediction   ->  1 channel
    t is continuous in [0,1], scaled to [0,999] for DiT's ada-norm.
    """
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
                cond: torch.Tensor) -> torch.Tensor:
        # x_t:  (B, 1, H, W)  interpolated sample I_t
        # t:    (B,)           integer in [0, 999]
        # cond: (B, 1, H, W)  observation Y
        inp = torch.cat([x_t, cond], dim=1)
        dummy = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t, class_labels=dummy).sample


class ConditionalDiTWithEll(nn.Module):
    """
    Like ConditionalDiT but also conditioned on curriculum level ell.
    Input:  cat([I_t, Y], dim=1)  ->  2 channels (unchanged)
    ell is discretized to [0, ELL_SCALE] and passed via class_labels AdaLN slot,
    mirroring how t is discretized to [0, INTEGRATION_SCALE] for the timestep.
    """
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
                cond: torch.Tensor, ell: float) -> torch.Tensor:
        # x_t:  (B, 1, H, W)  interpolated sample I_t
        # t:    (B,)           integer in [0, 999]
        # cond: (B, 1, H, W)  observation Y
        # ell:  float in [0, 1], discretized to [0, ELL_SCALE] via class_labels
        inp = torch.cat([x_t, cond], dim=1)
        ell_label = torch.full(
            (x_t.size(0),), int(round(float(ell) * ELL_SCALE)),
            dtype=torch.long, device=x_t.device,
        )
        return self.dit(inp, timestep=t, class_labels=ell_label).sample