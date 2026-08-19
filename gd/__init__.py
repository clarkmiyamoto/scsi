"""Direct gradient-descent CryoET reconstruction.

Recover one known 3D shape from a finite CryoET tilt-series dataset via Adam
directly on the shape's own parameters (point cloud or voxel occupancy) -- no
neural network, no EM prior update. Run via `uv run python -m gd [flags]`.
"""
from .cli import main

__all__ = ["main"]
