"""
Vendored copy of CryoFM's 3D UNet (~/cryofm/src/cryofm/core/models/unet3d, cryofm @ 6448681), the
backbone of the supervised IgG-1D model in ~/cryofm/src/cryofm/projects/igg1d. Apache-2.0 headers
are kept in each file.

Changes from the source: controlnet.py and mid_blocks_no_attention.py are not copied, this
__init__ exports only UNet3DModel, and down_blocks.reshape_for_attention uses plain torch instead
of einops. Everything else is byte-identical, so cryofm checkpoints load with strict=True.
"""

from .unet import UNet3DModel, UNet3DOutput
