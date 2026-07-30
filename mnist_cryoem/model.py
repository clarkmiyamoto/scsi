from diffusers import DiTTransformer2DModel, UNet2DModel, UNet3DConditionModel
import torch
import torch.nn as nn

IMAGE_SIZE: int = 32
VOL_SIZE: int = 32
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


class ConditionalUNet3D(nn.Module):
    """
    3D image (volume) branch, for the CryoET generalization. Wraps diffusers'
    UNet3DConditionModel — the only working 3D backbone in this repo (a DiT3D analogue would
    wrap diffusers.Transformer3DModel, but the closest existing attempt, simple_3d/model.py's
    ConditionalDiT3D, is dead/commented-out code; not resurrected here).

    Input:  cat([I_t^x, y_broadcast], dim=1)  ->  2 channels, (B, 2, D, H, W)
    Output: velocity prediction v_x            ->  1 channel

    y_broadcast is expected ALREADY depth-broadcast to (B, 1, D, H, W) — broadcasting is owned
    by ConditionalVelocityCryoET3D below, not done internally here, matching this repo's
    ConditionalVelocityCryoEM convention (intentionally diverging from simple_3d/model.py's
    internal-broadcast style).
    """
    def __init__(self, vol_size=VOL_SIZE, block_out_channels=(64, 128, 256, 256),
                 layers_per_block=2, norm_num_groups=8):
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

    def forward(self, x_t: torch.Tensor, t_int: torch.Tensor,
                y_broadcast: torch.Tensor) -> torch.Tensor:
        # x_t:         (B, 1, D, H, W)  interpolated volume state I_t^x
        # t_int:       (B,)             integer in [0, INTEGRATION_SCALE]
        # y_broadcast: (B, 1, D, H, W)  2D observation tiled across depth
        inp = torch.cat([x_t, y_broadcast], dim=1)  # (B, 2, D, H, W)
        # UNet3DConditionModel expects (B, C, F, H, W) with a batch dim on `timestep`, and
        # requires `encoder_hidden_states` for its cross-attention API; fed an all-zeros dummy
        # since there's no real cross-attention conditioning here — same pattern as simple_3d.
        dummy_hidden = torch.zeros(x_t.size(0), 1, self.unet.config.cross_attention_dim,
                                   device=x_t.device, dtype=x_t.dtype)
        out = self.unet(inp, timestep=t_int, encoder_hidden_states=dummy_hidden).sample
        return out


class PoseHead3D(nn.Module):
    """
    SO(3) pose branch, via the 6D continuous rotation representation (see si.py's module
    docstring). Treats pose as a flat R^6 state exactly like the image branch treats pixels —
    deliberately NOT fed as a converted rotation matrix: a given (B,6) vector is already an
    unambiguous representative (no quaternion-style double-cover sign issue to disambiguate),
    so raw pose_t is the direct structural analogue of PoseHead's cos(theta)/sin(theta) inputs.

    Conv3d trunk on cat([x_t, y_broadcast], dim=1) -> global pool -> MLP taking
    [feat, pose_t (6 numbers), t_frac] -> (B, 6) unconstrained velocity output.
    """
    def __init__(self, vol_size=VOL_SIZE, base_channels=16, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        feat_dim = base_channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 6 + 1, hidden),  # + pose_t (6) + t (1)
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, x_t: torch.Tensor, y_broadcast: torch.Tensor,
                pose_t: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        # x_t, y_broadcast: (B, 1, D, H, W)
        # pose_t: (B, 6)       t_frac: (B,) in [0, 1]
        inp = torch.cat([x_t, y_broadcast], dim=1)
        feat = self.conv(inp).flatten(1)
        h = torch.cat([feat, pose_t, t_frac.unsqueeze(-1)], dim=-1)
        return self.mlp(h)  # (B, 6)


class ConditionalVelocityCryoET3D(nn.Module):
    """
    Joint velocity field b_hat over the (volume, 6D pose) product state — the 3D/CryoET
    analogue of ConditionalVelocityCryoEM. Owns the y-broadcast and the t-fraction -> t_int
    conversion so si.py stays agnostic to how the two branches consume time/conditioning.
    """
    def __init__(self, vol_size=VOL_SIZE, arch: str = "unet3d"):
        super().__init__()
        self.vol_size = vol_size
        if arch != "unet3d":
            raise ValueError(f"Unknown arch: {arch!r}. Only 'unet3d' is implemented.")
        self.image_branch = ConditionalUNet3D(vol_size=vol_size)
        self.pose_branch = PoseHead3D(vol_size=vol_size)

    def forward(self, x_t: torch.Tensor, pose_t: torch.Tensor,
                t_frac: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_t: (B, 1, D, H, W)   pose_t: (B, 6)   t_frac: (B,) in [0,1]   y: (B, 1, H, W)
        D = x_t.size(-3)
        y_broadcast = y.unsqueeze(-3).expand(-1, -1, D, -1, -1)  # (B, 1, D, H, W)
        t_int = (t_frac * INTEGRATION_SCALE).long()

        v_x = self.image_branch(x_t, t_int, y_broadcast)
        v_pose = self.pose_branch(x_t, y_broadcast, pose_t, t_frac)
        return v_x, v_pose
