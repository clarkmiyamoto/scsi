from diffusers import DiTTransformer2DModel, UNet2DModel
import torch
import torch.nn as nn

IMAGE_SIZE: int = 32
INTEGRATION_SCALE: float = 999


class ConditionalDiT(nn.Module):
    """
    Image branch. Ported verbatim from image_2d/model.py: velocity field for the image
    component of the joint (image, pose) state, conditioned on the 1D observation broadcast
    to a 2D channel. Deliberately does NOT see the pose branch's state, to keep this proven
    module unmodified.

    Input:  cat([I_t^x, y_broadcast], dim=1)  ->  2 channels
    Output: velocity prediction v_x            ->  1 channel
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

    def forward(self, x_t: torch.Tensor, t_int: torch.Tensor,
                y_broadcast: torch.Tensor) -> torch.Tensor:
        # x_t:         (B, 1, H, W)  interpolated image state I_t^x
        # t_int:       (B,)          integer in [0, INTEGRATION_SCALE]
        # y_broadcast: (B, 1, H, W)  1D observation tiled across rows
        inp = torch.cat([x_t, y_broadcast], dim=1)
        dummy = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t_int, class_labels=dummy).sample


class ConditionalUNet2D(nn.Module):
    """
    Image branch alternative to ConditionalDiT, using diffusers' UNet2DModel. Same interface
    (forward signature, in/out channel counts) so it's a drop-in swap in
    ConditionalVelocityCryoEM — see the `arch` argument there.

    Input:  cat([I_t^x, y_broadcast], dim=1)  ->  2 channels
    Output: velocity prediction v_x            ->  1 channel
    """
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


class PoseHead(nn.Module):
    """
    Pose branch. New to this codebase: a small CNN + MLP regressing the angular velocity
    v_theta of the SO(2) pose state, conditioned on the same (image state, observation) pair
    plus the pose state itself (cos/sin of the current angle) and continuous time.

    Deliberately independent of ConditionalDiT's internals (separate small module, own
    downsampling stack) so the proven image branch stays untouched.
    """
    def __init__(self, image_size=IMAGE_SIZE, base_channels=16, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        feat_dim = base_channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden),  # + cos(theta_t), sin(theta_t), t
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_t: torch.Tensor, y_broadcast: torch.Tensor,
                theta_t: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        # x_t, y_broadcast: (B, 1, H, W)
        # theta_t: (B,) radians       t_frac: (B,) in [0, 1]
        inp = torch.cat([x_t, y_broadcast], dim=1)
        feat = self.conv(inp).flatten(1)
        h = torch.cat([
            feat,
            torch.cos(theta_t).unsqueeze(-1),
            torch.sin(theta_t).unsqueeze(-1),
            t_frac.unsqueeze(-1),
        ], dim=-1)
        return self.mlp(h).squeeze(-1)  # (B,)


class ConditionalVelocityCryoEM(nn.Module):
    """
    Joint velocity field b_hat over the (image, SO(2) pose) product state. Owns the
    y-broadcast and the t-fraction -> t_int conversion so si.py stays agnostic to how the two
    branches consume time/conditioning.
    """
    def __init__(self, image_size=IMAGE_SIZE, arch: str = "dit"):
        super().__init__()
        self.image_size = image_size
        if arch == "dit":
            self.image_branch = ConditionalDiT(image_size=image_size)
        elif arch == "unet":
            self.image_branch = ConditionalUNet2D(image_size=image_size)
        else:
            raise ValueError(f"Unknown arch: {arch!r}. Choose 'dit' or 'unet'.")
        self.pose_branch = PoseHead(image_size=image_size)

    def forward(self, x_t: torch.Tensor, theta_t: torch.Tensor,
                t_frac: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_t: (B, 1, H, W)   theta_t: (B,)   t_frac: (B,) in [0,1]   y: (B, 1, W)
        H = x_t.size(-2)
        y_broadcast = y.unsqueeze(-2).expand(-1, -1, H, -1)  # (B, 1, H, W)
        t_int = (t_frac * INTEGRATION_SCALE).long()

        v_x = self.image_branch(x_t, t_int, y_broadcast)
        v_theta = self.pose_branch(x_t, y_broadcast, theta_t, t_frac)
        return v_x, v_theta
