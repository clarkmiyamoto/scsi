import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose

VOL_SIZE: int = 32
N_SURFACE_POINTS: int = 4096


def load_modelnet10(data_root: str = "./data",
                    split: str = "train",
                    vol_size: int = VOL_SIZE) -> torch.Tensor:
    """
    Download (if needed) and return ModelNet10 as voxel grids.

    Returns:
        (N, 1, D, H, W) float32 in [-1, 1]  (occupancy remapped {0,1} -> {-1,1})
    """
    pre_transform = Compose([
        SamplePoints(N_SURFACE_POINTS, remove_faces=True),
        NormalizeScale(),
    ])
    dataset = ModelNet(
        root=data_root,
        name="10",
        train=(split == "train"),
        pre_transform=pre_transform,
    )

    volumes = []
    for data in dataset:
        vol = points_to_occupancy(data.pos, grid_size=vol_size)
        volumes.append(vol)

    x = torch.stack(volumes, dim=0).unsqueeze(1).float()   # (N, 1, D, H, W)
    x = x * 2.0 - 1.0                                      # {0,1} -> {-1,1}
    return x


def points_to_occupancy(pos: torch.Tensor, grid_size: int = VOL_SIZE) -> torch.Tensor:
    """
    Convert (N, 3) point cloud in approximately [-1, 1]^3 to binary occupancy grid.

    Returns:
        (D, H, W) float32 with values in {0, 1}
    """
    idx = ((pos.clamp(-1.0, 1.0) + 1.0) / 2.0 * grid_size).long()
    idx = idx.clamp(0, grid_size - 1)                       # (N, 3) integer indices

    vol = torch.zeros(grid_size, grid_size, grid_size)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return vol
