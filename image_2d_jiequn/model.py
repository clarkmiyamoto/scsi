"""Drift network ``b(x, t, cond)`` for the stochastic interpolant.

Per the design decision, this reuses the repo's networks from ``src/`` rather
than rewriting them. ``src_clean/`` is therefore *not* fully standalone — it adds
``src/`` to ``sys.path`` and imports the network. The conditioning is the raw
K-projection stack ``[K, 32, 32]`` (``latent_dim``); both backbones consume it as
image-shaped channels.

Two backbones are selectable:
  * ``'unet'`` — ``ConditionalDhariwalUNet`` (vendored NVIDIA EDM code). The
    K-channel latent is routed through a small conv encoder then concatenated to
    the input. No extra dependency.
  * ``'dit'``  — ``ConditionalDiT`` (a ``DiTTransformer2DModel`` from the
    ``diffusers`` package). The K-channel latent is concatenated to the input and
    a single transformer predicts the velocity. This is the architecture behind
    the paper's best-FID tilt-series recipes (``private_docs/best_launch_recipes.md``).
    Requires ``pip install diffusers``.

Both expose ``forward(x, noise_labels, latents)`` matching the ``b(x, t, cond)``
call the interpolant makes, so no wrapper is needed.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from networks import ConditionalDhariwalUNet  # noqa: E402  (after sys.path tweak)


def build_model(D: int = 32, nc: int = 1, K: int = 16, model_channels: int = 32,
                network: str = "unet"):
    """Build the conditional drift network.

    Args:
        D:              image resolution (32 for padded MNIST).
        nc:             image channels (1 for MNIST).
        K:              number of tilt projections -> conditioning channels.
        model_channels: base channel multiplier for the UNet (ignored by DiT).
        network:        ``'unet'`` (default) or ``'dit'`` (best FID, needs diffusers).

    Returns a module whose ``forward(x, t, cond)`` predicts the velocity field on
    ``[N, nc, D, D]`` given time ``t`` of shape ``[N]`` and conditioning
    ``cond`` of shape ``[N, K, D, D]``.
    """
    latent_dim = [int(K), D, D]
    if network == "unet":
        return ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                                       model_channels=model_channels)
    if network == "dit":
        from networks_dit import ConditionalDiT
        try:
            # Production defaults (patch_size=4, hidden=192, depth=6, heads=6),
            # the "baseline DiT recipe" used for the published tilt-series FIDs.
            return ConditionalDiT(D, nc, nc, latent_dim=latent_dim)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "network='dit' requires the `diffusers` package: "
                "pip install diffusers"
            ) from e
    raise ValueError(f"unknown network '{network}'; use 'unet' or 'dit'")