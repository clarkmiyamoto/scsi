"""Drift network ``b(x, t, cond)`` for the stochastic interpolant.

Per the design decision, this reuses the repo's ``ConditionalDhariwalUNet``
(``src/networks.py``, vendored NVIDIA EDM code) rather than rewriting a compact
UNet. ``src_clean/`` is therefore *not* fully standalone — it adds ``src/`` to
``sys.path`` and imports the network. The conditioning is the raw K-projection
stack ``[K, 32, 32]`` (``latent_dim``), routed through the UNet's image-latent
path (a small conv encoder whose output is concatenated to the input).

``ConditionalDhariwalUNet.forward(x, noise_labels, latents)`` already matches the
``b(x, t, cond)`` call the interpolant makes, so no wrapper is needed.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from networks import ConditionalDhariwalUNet  # noqa: E402  (after sys.path tweak)


def build_model(D: int = 32, nc: int = 1, K: int = 16, model_channels: int = 32):
    """Build the conditional drift UNet.

    Args:
        D:              image resolution (32 for padded MNIST).
        nc:             image channels (1 for MNIST).
        K:              number of tilt projections -> conditioning channels.
        model_channels: base channel multiplier (32 keeps it light for smoke tests).

    Returns a module whose ``forward(x, t, cond)`` predicts the velocity field on
    ``[N, nc, D, D]`` given time ``t`` of shape ``[N]`` and conditioning
    ``cond`` of shape ``[N, K, D, D]``.
    """
    return ConditionalDhariwalUNet(
        D, nc, nc,
        latent_dim=[int(K), D, D],
        model_channels=model_channels,
    )
