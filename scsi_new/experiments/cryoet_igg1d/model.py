"""
The velocity field b_t(x | y) for IgG-1D: the supervised cryofm model's network
(~/cryofm/src/cryofm/projects/igg1d/image_encoder.py + configs/igg1d/igg1d_cond_64.py), vendored
so this experiment runs in the scsi venv. The UNet is ./unet3d (see its __init__ for the changes).

ImageEncoder2D and ImageCondUNet3D below are copied from cryofm unchanged, so a cryofm
IgG1DImageCond checkpoint's `model.*` weights load into ImageCondUNet3D with strict=True.
ConditionalVelocityIgG adapts it to the forward(x_t, t, y) interface scsi.mstep_lifted and
ode.euler_integration call: it standardises the image as cryofm does, never drops the condition
(no classifier-free guidance here), maps t in [0, 1] to the UNet's timestep scale, and runs the
network under bf16 autocast on CUDA -- cryofm trains in bf16-mixed; at 64^3 fp32 would roughly
double memory. The output is cast back to fp32, so the loss and the ODE stay in fp32.

Time convention differs from cryofm (there t = 0 is data): here t = 0 is noise and t = 1 data, as
everywhere in scsi_new. Weights are not shared between the two, so only consistency matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet3d import UNet3DModel

INTEGRATION_SCALE: float = 999  # t in [0, 1] -> UNet timestep, as cryoet_mnist3d/model.py

# `model` dict of cryofm's configs/igg1d/igg1d_cond_64.py; sample_size is set from the resolution.
UNET_CFG = dict(
    in_channels=3,
    out_channels=1,
    time_embedding_type="positional",
    time_embedding_dim=None,
    freq_shift=0,
    flip_sin_to_cos=True,
    down_block_types=("DownBlock3D", "DownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D"),
    up_block_types=("AttnUpBlock3D", "AttnUpBlock3D", "UpBlock3D", "UpBlock3D"),
    block_out_channels=(64, 128, 256, 512),
    layers_per_block=2,
    mid_block_scale_factor=1,
    downsample_padding=1,
    downsample_type="conv",
    upsample_type="conv",
    dropout=0.0,
    act_fn="silu",
    attention_head_dim=8,
    norm_num_groups=32,
    attn_norm_num_groups=None,
    norm_eps=1e-5,
    resnet_time_scale_shift="scale_shift",
    class_embed_type="identity",
)

# `image_encoder` dict of the same config.
ENCODER_CFG = dict(channels=(64, 128, 256, 512), blocks_per_stage=2, norm_num_groups=32)


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class ImageEncoder2D(nn.Module):
    """Residual CNN mapping a (B, 1, D, D) particle image to a (B, emb_dim) embedding.

    Each stage halves the resolution; the final feature map is average-pooled, which makes the
    embedding insensitive to the particle's in-plane shift.
    """

    def __init__(self, emb_dim, channels=(64, 128, 256, 512), blocks_per_stage=2, norm_num_groups=32):
        super().__init__()
        self.conv_in = nn.Conv2d(1, channels[0], 3, padding=1)
        layers = []
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            for _ in range(blocks_per_stage):
                layers.append(ResBlock2D(in_ch, ch, norm_num_groups))
                in_ch = ch
            if i < len(channels) - 1:
                layers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        self.blocks = nn.Sequential(*layers)
        self.norm_out = nn.GroupNorm(norm_num_groups, in_ch)
        self.proj = nn.Sequential(nn.Linear(in_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, img):
        h = self.blocks(self.conv_in(img))
        h = F.silu(self.norm_out(h)).mean(dim=(-2, -1))
        return self.proj(h)


class ImageCondUNet3D(nn.Module):
    """CryoFM2's UNet3DModel conditioned on a particle image, for targets in the image's frame.

    Like CryoFM2Cond, the condition enters as extra input channels next to x_t: the image
    broadcast along z (the viewing axis, i.e. an unfiltered back-projection) and a flag channel
    that is 1 for conditional and 0 for unconditional inputs. A 2D CNN embedding of the image is
    also summed into the timestep embedding (the UNet must use `class_embed_type="identity"`),
    giving every block a global summary of the image. The unconditional branch used by
    classifier-free guidance sees zeros and a learned null embedding instead.
    """

    def __init__(self, unet_cfg, encoder_cfg):
        super().__init__()
        assert unet_cfg.get("class_embed_type") == "identity", "image conditioning needs class_embed_type='identity'"
        assert unet_cfg.get("in_channels") == 3, "inputs are [x_t, back-projected image, condition flag]"
        self.unet = UNet3DModel(**unet_cfg)
        emb_dim = self.unet.time_embedding.linear_2.out_features
        self.encoder = ImageEncoder2D(emb_dim, **encoder_cfg)
        self.null_emb = nn.Parameter(torch.zeros(emb_dim))

    def condition(self, img, drop=None):
        """Condition on (B, 1, D, D) images; rows where `drop` is True become unconditional.

        Returns (embedding (B, E), spatial conditioning (B, 2, D, D, D)).
        """
        n, D = len(img), img.shape[-1]
        emb = self.encoder(img)
        keep = torch.ones(n, dtype=img.dtype, device=img.device)
        if drop is not None:
            emb = torch.where(drop[:, None], self.null_emb.to(emb.dtype).expand_as(emb), emb)
            keep = (~drop).to(img.dtype)
        lifted = (img * keep[:, None, None, None])[:, :, None].expand(n, 1, D, D, D)
        flag = keep[:, None, None, None, None].expand(n, 1, D, D, D)
        return emb, torch.cat([lifted, flag], dim=1)

    def null_condition(self, n, D):
        emb = self.null_emb[None].expand(n, -1)
        return emb, torch.zeros(n, 2, D, D, D, dtype=emb.dtype, device=emb.device)

    def forward(self, zt, t, cond):
        emb, spatial = cond
        return self.unet(torch.cat([zt, spatial.to(zt.dtype)], dim=1), timestep=t, class_labels=emb).sample


def normalize_image(y: torch.Tensor) -> torch.Tensor:
    """cryofm IgG1DImageCond.normalize_image, for (B, 1, D, D) input: zero mean, unit variance each."""
    y = y - y.mean(dim=(-2, -1), keepdim=True)
    return y / y.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class ConditionalVelocityIgG(nn.Module):
    """b_t(x_t | y): x_t (B, 1, D, D, D), t (B,) or broadcastable in [0, 1], y (B, 1, D, D)."""

    def __init__(self, resolution: int = 64,
                 block_out_channels: tuple[int, ...] = UNET_CFG["block_out_channels"],
                 layers_per_block: int = UNET_CFG["layers_per_block"],
                 encoder_channels: tuple[int, ...] = ENCODER_CFG["channels"],
                 bf16: bool = True):
        super().__init__()
        unet_cfg = dict(UNET_CFG, sample_size=resolution, block_out_channels=tuple(block_out_channels),
                        layers_per_block=layers_per_block)
        self.net = ImageCondUNet3D(unet_cfg, dict(ENCODER_CFG, channels=tuple(encoder_channels)))
        self.bf16 = bf16

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        timestep = t.reshape(-1).float() * INTEGRATION_SCALE
        use_bf16 = self.bf16 and x_t.device.type == "cuda"
        with torch.autocast(x_t.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            cond = self.net.condition(normalize_image(y.float()))
            v = self.net(x_t, timestep, cond)
        return v.float()
