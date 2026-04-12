"""
Synthetic Cryo-EM Dataset Generator (PyTorch)
====================================
Generates 2D projections of known 3D objects by integrating (ray-summing)
their density along a viewing direction. This gives you:

  1. The ground-truth 3D volume  (N x N x N voxel grid)
  2. A set of 2D projection images (N x N grayscale)
  3. The rotation matrices used for each projection

The forward model is a simple parallel-beam projection + additive noise:
    observation(x, y) = ∫ volume(R⁻¹ · [x, y, z]ᵀ) dz  +  ε

where R is a 3D rotation matrix and ε ~ N(0, σ²).
This is encapsulated in the `forward_model()` function, which you can
import directly and use / invert in your reconstruction code.

All core computation (volume building, projection, forward model) uses
PyTorch tensors, making the forward model differentiable via autograd.

Usage
-----
    python data_fake.py                       # defaults
    python data_fake.py --resolution 128      # higher res
    python data_fake.py --num_projections 200 # more views
    python data_fake.py --objects sphere torus # pick objects
    python data_fake.py --output_dir my_data  # custom dir
    python data_fake.py --noise_std 0.05      # add noise
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from torchvision import datasets
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# 3D object builders – each returns an (N, N, N) density volume in [0, 1]
# ---------------------------------------------------------------------------

def make_sphere(N: int) -> torch.Tensor:
    """Solid sphere centred in the volume, radius = 0.35 * N."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    vol = torch.zeros_like(r)
    vol[r <= 0.7] = 1.0
    return vol


def make_cube(N: int) -> torch.Tensor:
    """Solid cube centred in the volume, half-side = 0.3 * N."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vol = torch.zeros_like(X)
    mask = (torch.abs(X) <= 0.6) & (torch.abs(Y) <= 0.6) & (torch.abs(Z) <= 0.6)
    vol[mask] = 1.0
    return vol


def make_torus(N: int) -> torch.Tensor:
    """Torus centred in volume.  Major radius R=0.45, tube radius r=0.18."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    R_major, r_tube = 0.45, 0.18
    dist = (torch.sqrt(X**2 + Y**2) - R_major) ** 2 + Z**2
    vol = torch.zeros_like(X)
    vol[dist <= r_tube**2] = 1.0
    return vol


def make_cylinder(N: int) -> torch.Tensor:
    """Solid cylinder aligned along Z, radius=0.3, half-height=0.6."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)
    vol = torch.zeros_like(X)
    vol[(r <= 0.3) & (torch.abs(Z) <= 0.6)] = 1.0
    return vol


def make_double_helix(N: int) -> torch.Tensor:
    """Two intertwined helical tubes – a very simplified DNA-like shape."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vol = torch.zeros_like(X)
    tube_r = 0.08
    helix_R = 0.3
    pitch = 4.0 * math.pi

    for strand_offset in [0, math.pi]:
        cx = helix_R * torch.cos(pitch * Z / 2.0 + strand_offset)
        cy = helix_R * torch.sin(pitch * Z / 2.0 + strand_offset)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        vol[dist2 <= tube_r**2] = 1.0

    vol[torch.abs(Z) > 0.85] = 0.0
    return vol


def make_ellipsoid(N: int) -> torch.Tensor:
    """Ellipsoid with distinct semi-axes (0.7, 0.4, 0.3)."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    val = (X / 0.7) ** 2 + (Y / 0.4) ** 2 + (Z / 0.3) ** 2
    vol = torch.zeros_like(X)
    vol[val <= 1.0] = 1.0
    return vol


def make_hollow_sphere(N: int) -> torch.Tensor:
    """Hollow sphere (shell).  Outer radius 0.65, inner radius 0.50."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    vol = torch.zeros_like(r)
    vol[(r <= 0.65) & (r >= 0.50)] = 1.0
    return vol


def make_tetrahedral(N: int) -> torch.Tensor:
    """Four small spheres at tetrahedron vertices – a molecular-like object."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vol = torch.zeros_like(X)

    s = 0.45
    verts = s * torch.tensor([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=torch.float32) / math.sqrt(3)

    blob_r = 0.22
    for v in verts:
        d2 = (X - v[0]) ** 2 + (Y - v[1]) ** 2 + (Z - v[2]) ** 2
        vol[d2 <= blob_r**2] = 1.0
    return vol


def make_l_shape(N: int) -> torch.Tensor:
    """An L-shaped block – useful because it breaks symmetry."""
    coords = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vol = torch.zeros_like(X)
    mask1 = (X >= -0.6) & (X <= -0.1) & (Y >= -0.6) & (Y <= 0.6) & (torch.abs(Z) <= 0.3)
    mask2 = (X >= -0.6) & (X <= 0.6) & (Y >= -0.6) & (Y <= -0.1) & (torch.abs(Z) <= 0.3)
    vol[mask1 | mask2] = 1.0
    return vol


# Registry of available objects
OBJECT_REGISTRY = {
    "sphere": make_sphere,
    "cube": make_cube,
    "torus": make_torus,
    "cylinder": make_cylinder,
    "double_helix": make_double_helix,
    "ellipsoid": make_ellipsoid,
    "hollow_sphere": make_hollow_sphere,
    "tetrahedral": make_tetrahedral,
    "l_shape": make_l_shape,
}


# ---------------------------------------------------------------------------
# Forward model: parallel-beam projection via trilinear interpolation
# ---------------------------------------------------------------------------

def project_volume(volume: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Compute a 2D parallel-beam projection of a 3D volume.

    Parameters
    ----------
    volume : (N, N, N) tensor
        3D density map.
    rotation : (3, 3) tensor
        Rotation matrix.  The projection is taken along the rotated Z-axis.

    Returns
    -------
    image : (N, N) tensor
        2D projection (sum along the viewing ray).

    Forward model
    -------------
    For each pixel (u, v) in detector space we integrate along rays parallel
    to the rotated z-axis:

        image[u, v] = Σ_t  volume( R⁻¹ · [u, v, t]ᵀ )

    Coordinates are in voxel units centred on the volume.
    Uses torch.nn.functional.grid_sample (trilinear interpolation),
    making this operation differentiable w.r.t. the volume.
    """
    N = volume.shape[0]
    half = (N - 1) / 2.0

    # Detector pixel coordinates (centred)
    u = torch.arange(N, dtype=volume.dtype, device=volume.device) - half
    v = torch.arange(N, dtype=volume.dtype, device=volume.device) - half
    t = torch.arange(N, dtype=volume.dtype, device=volume.device) - half

    # Build the full (N, N, N, 3) array of query points in rotated frame
    UU, VV, TT = torch.meshgrid(u, v, t, indexing="ij")
    pts_rot = torch.stack([UU, VV, TT], dim=-1)  # (N, N, N, 3)

    # Rotate back into volume frame
    R_inv = rotation.T  # rotation matrices are orthogonal
    pts_vol = torch.einsum('ij,...j->...i', R_inv, pts_rot)  # (N, N, N, 3)

    # Normalize to [-1, 1] for grid_sample and reorder axes.
    # grid_sample expects (x, y, z) mapping to (W, H, D) of the input.
    # Volume dims after unsqueeze: D=dim0(X), H=dim1(Y), W=dim2(Z)
    # So: grid[...,0]->W(Z), grid[...,1]->H(Y), grid[...,2]->D(X)
    grid = pts_vol[..., [2, 1, 0]] / half  # (N, N, N, 3)
    grid = grid.unsqueeze(0)  # (1, N, N, N, 3)

    vol_input = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N, N)

    sampled = F.grid_sample(
        vol_input, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    sampled = sampled.squeeze(0).squeeze(0)  # (N, N, N)

    # Sum along the ray direction (axis=2) to get the projection
    image = sampled.sum(dim=2)
    return image


def forward_model(
    volume: torch.Tensor,
    rotation: torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Full forward model: project a 3D volume into a noisy 2D observation.

    This is the function you should use (and invert) in your reconstruction
    pipeline.  It chains two steps:

        1. Parallel-beam projection  (deterministic, linear, differentiable)
        2. Additive Gaussian noise   (stochastic)

    Parameters
    ----------
    volume : (N, N, N) tensor
        3D density map.
    rotation : (3, 3) tensor
        Rotation matrix defining the viewing direction.  The projection
        integrates along the rotated Z-axis.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise, expressed as a
        *fraction* of the clean image's maximum intensity.
        0 → noiseless (default).
    generator : torch.Generator or None
        Random number generator for reproducibility.  If None a fresh
        default generator is used when noise is requested.

    Returns
    -------
    observation : (N, N) tensor
        Noisy 2D projection image.

    Forward model equation
    ----------------------
        observation(u, v) = Σ_t volume(R⁻¹ · [u, v, t]ᵀ)  +  ε

    where  ε ~ N(0, σ²)  and  σ = noise_std × max(clean_image).
    """
    # Step 1 – deterministic projection
    clean = project_volume(volume, rotation)

    # Step 2 – additive Gaussian noise
    if noise_std > 0:
        sigma = noise_std * clean.max().item()
        noise = torch.normal(
            mean=0.0, std=sigma, size=clean.shape,
            dtype=clean.dtype, device=clean.device, generator=generator,
        )
        observation = clean + noise
    else:
        observation = clean

    return observation


# ---------------------------------------------------------------------------
# Rotation sampling
# ---------------------------------------------------------------------------

def sample_rotations(num: int, seed: int = 42) -> list[torch.Tensor]:
    """Return a list of uniformly random 3D rotation matrices as torch tensors."""
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(num, random_state=rng)
    return [torch.from_numpy(r.as_matrix()).float() for r in rotations]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    object_names: list[str],
    resolution: int = 64,
    num_projections: int = 50,
    noise_std: float = 0.0,
    seed: int = 42,
    output_dir: str = "synthetic_cryoem_dataset",
):
    """
    Generate the full dataset and save to disk.

    Outputs (per object)
    --------------------
    <output_dir>/<object_name>/volume.npy          – (N,N,N) ground truth
    <output_dir>/<object_name>/projections.npy      – (K,N,N) projection stack
    <output_dir>/<object_name>/rotations.npy        – (K,3,3) rotation matrices
    <output_dir>/<object_name>/projections/all/*.png  – individual images (ImageFolder-friendly)
    <output_dir>/<object_name>/metadata.json         – parameters used
    """
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rotations = sample_rotations(num_projections, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)

    for obj_name in object_names:
        if obj_name not in OBJECT_REGISTRY:
            print(f"WARNING: unknown object '{obj_name}', skipping. "
                  f"Available: {list(OBJECT_REGISTRY.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"  Generating: {obj_name}  (resolution={resolution}, "
              f"projections={num_projections})")
        print(f"{'='*60}")

        obj_dir = out / obj_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        img_dir = obj_dir / "projections" / "all"  # 'all' subfolder for torchvision ImageFolder
        img_dir.mkdir(parents=True, exist_ok=True)

        # Build the 3D volume
        volume = OBJECT_REGISTRY[obj_name](resolution)
        np.save(obj_dir / "volume.npy", volume.numpy().astype(np.float32))

        # Project from each viewing direction
        proj_stack = torch.zeros(num_projections, resolution, resolution)
        rot_stack = torch.zeros(num_projections, 3, 3)

        for i, R in enumerate(rotations):
            img = forward_model(volume, R, noise_std=noise_std, generator=generator)

            proj_stack[i] = img
            rot_stack[i] = R

            # Save individual PNG (normalised to 0-255 grayscale)
            img_np = img.detach().numpy()
            img_norm = img_np - img_np.min()
            mx = img_norm.max()
            if mx > 0:
                img_norm = img_norm / mx
            img_uint8 = (img_norm * 255).astype(np.uint8)
            Image.fromarray(img_uint8, mode="L").save(
                img_dir / f"proj_{i:04d}.png"
            )

            if (i + 1) % max(1, num_projections // 5) == 0 or i == 0:
                print(f"    [{i+1:>4d}/{num_projections}] projections done")

        np.save(obj_dir / "projections.npy", proj_stack.numpy())
        np.save(obj_dir / "rotations.npy", rot_stack.numpy())

        # Metadata
        meta = {
            "object": obj_name,
            "resolution": resolution,
            "num_projections": num_projections,
            "noise_std": noise_std,
            "seed": seed,
            "volume_file": "volume.npy",
            "projections_file": "projections.npy",
            "rotations_file": "rotations.npy",
        }
        with open(obj_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✓ Saved to {obj_dir}/")

    print(f"\nDone. Dataset root: {out}/")

def load_dataset(path: str, transform: transforms.Compose, seed: int = 42):
    '''
    Load the dataset and corresponding forward model from the given path

    Args:
        path: str
            The path to the dataset

    Returns:
        dataset: datasets.ImageFolder
            The dataset
        forward_model_func: callable
            The forward model
    '''
    path_projections = path + '/projections'
    dataset = datasets.ImageFolder(root=path_projections, transform=transform)

    metadata_path = path + '/metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    noise_std = metadata['noise_std']

    generator = torch.Generator().manual_seed(seed)
    forward_model_func = lambda x: forward_model(
        x, torch.eye(3), noise_std=noise_std, generator=generator,
    )

    return dataset, forward_model_func



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    all_objects = list(OBJECT_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Generate synthetic cryo-EM-like 2D projection datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available objects: {', '.join(all_objects)}",
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=64,
        help="Voxel grid size N (volume is NxNxN, images are NxN). Default: 64",
    )
    parser.add_argument(
        "--num_projections", "-n", type=int, default=50,
        help="Number of projection images per object. Default: 50",
    )
    parser.add_argument(
        "--objects", nargs="+", default=all_objects,
        help=f"Which objects to generate. Default: all. Choices: {all_objects}",
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.0,
        help="Gaussian noise level as a fraction of max intensity. Default: 0 (clean)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="data",
        help="Output directory. Default: data",
    )
    args = parser.parse_args()

    generate_dataset(
        object_names=args.objects,
        resolution=args.resolution,
        num_projections=args.num_projections,
        noise_std=args.noise_std,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
