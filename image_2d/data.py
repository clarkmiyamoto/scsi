import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset


from model import IMAGE_SIZE

def load_mnist(n_obs: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # -> [-1, 1]
    ])
    dataset = datasets.MNIST("./data", train=True, download=True,
                             transform=transform)
    loader = DataLoader(dataset, batch_size=n_obs, shuffle=True)
    x_gt_all, _ = next(iter(loader))          # (n_obs, 1, 32, 32)
    return x_gt_all