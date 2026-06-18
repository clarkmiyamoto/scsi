"""DiT (Diffusion Transformer) network for SCSI, mirroring the collaborator's
clark_scsi/image_2d/model.py ConditionalDiT but adapted to our trainer's
(x, t, latents) call signature.

Conditioning pattern: image-shaped latent (e.g. the corrupted observation in
--embed mode) is concatenated to the interpolant input along the channel axis,
then a single DiTTransformer2DModel produces the velocity prediction. The
continuous interpolant time t in [0, 1] is bucketed to a long-integer
timestep in [0, INTEGRATION_SCALE] for DiT's adaptive layer norm, matching
the collaborator's `t_dit = (t * INTEGRATION_SCALE).long()` pattern at
clark_scsi/image_2d/si.py:82.
"""

import torch

INTEGRATION_SCALE = 999  # matches clark_scsi/image_2d/model.py:6


class ConditionalDiT(torch.nn.Module):
    """DiT wrapper that exposes the same forward(x, noise_labels, latents, augment_labels)
    contract as ConditionalDhariwalUNet, so it is drop-in replaceable in the
    interpolant call sites (`b(It, t, latent1)`).
    """

    def __init__(self,
                 img_resolution,
                 in_channels,
                 out_channels,
                 latent_dim=None,
                 patch_size=4,
                 hidden=192,
                 depth=6,
                 heads=6,
                 integration_scale=INTEGRATION_SCALE,
                 **_unused_kwargs):
        super().__init__()
        # Lazy import so non-DiT runs do not pay the diffusers import cost
        # (and runs in environments without diffusers do not break).
        from diffusers import DiTTransformer2DModel

        if latent_dim is None:
            latent_channels = 0
        elif len(latent_dim) == 3:
            latent_channels = int(latent_dim[0])
        elif len(latent_dim) == 2:
            latent_channels = 1
        else:
            raise ValueError(
                "ConditionalDiT only supports image-shaped latents (2D or 3D) "
                f"or no latents; got latent_dim={latent_dim}. For 1-D latents "
                "(e.g. shifts-as-vectors), keep --network unet."
            )

        self.latent_channels = latent_channels
        self.integration_scale = integration_scale

        self.dit = DiTTransformer2DModel(
            sample_size=img_resolution,
            patch_size=patch_size,
            in_channels=in_channels + latent_channels,
            out_channels=out_channels,
            num_layers=depth,
            num_attention_heads=heads,
            attention_head_dim=hidden // heads,
            num_embeds_ada_norm=integration_scale + 1,
        )

    def forward(self, x, noise_labels, latents=None, augment_labels=None):
        # Concatenate the corrupted observation onto the channel axis if
        # present (the embed-mode latent has the same H,W as x).
        if latents is not None and self.latent_channels > 0:
            inp = torch.cat([x, latents], dim=1)
        else:
            inp = x

        # Bucket continuous t -> long integer in [0, integration_scale - 1].
        # During training, t ~ U[0, 1) (half-open), so (t * 999).long() lands
        # in [0, 998] — bucket 999 is NEVER updated. Our transport, however,
        # starts at ti=1.0 exactly (interpolant_utils.py:104, i=1 → ti=1.0),
        # which would hit untrained bucket 999. Clamp to integration_scale-1
        # so the inference-time bucket distribution stays inside the
        # train-time support. (UNet uses continuous sinusoidal embeddings and
        # is unaffected; this is a DiT-only fix.)
        t_long = (noise_labels.clamp(0.0, 1.0) * self.integration_scale).long()
        t_long = t_long.clamp(max=self.integration_scale - 1)

        # class_labels are unused (class-unconditional); zero tensor satisfies
        # the DiT API.
        class_labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        return self.dit(inp, timestep=t_long, class_labels=class_labels).sample