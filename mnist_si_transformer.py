"""
Conditional Stochastic-Interpolant Transformer for MNIST denoising.

Setup: noisy MNIST image  -->  clean MNIST image
Interpolant: I_t = (1-t)*Z + t*X,  t in [0,1]
  Z ~ N(0,I) (noise), X = clean image
Velocity target: dI_t/dt = X - Z  (constant along each path)
Model learns v_theta(I_t, t, cond) and inference integrates dx/dt = v_theta
from t=0 to t=1 with Euler steps.

Requirements:
    pip install torch torchvision diffusers accelerate
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DiTTransformer2DModel

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1. Data: pairs of (noisy, clean) MNIST ----------
class NoisyMNIST(torch.utils.data.Dataset):
    """Adds Gaussian noise to MNIST. Replace `corrupt` with whatever
    corruption you actually want (blur, masking, etc.)."""
    def __init__(self, train=True, noise_std=0.3, corruption="awgn"):
        self.base = datasets.MNIST(
            root="./data", train=train, download=True,
            transform=transforms.Compose([
                transforms.Resize(32),               # 32x32 is friendlier for patching
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
            ]),
        )
        self.noise_std = noise_std
        self.corruption = corruption

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.corruption == "awgn":
            return self.awgn(idx)
        if self.corruption == "mra":
            return self.mra(idx)
        raise ValueError(f"Invalid corruption: {self.corruption}")

    def awgn(self, idx):
        clean, _ = self.base[idx]
        noisy = clean + self.noise_std * torch.randn_like(clean)
        return noisy, clean

    def mra(self, idx):
        clean, _ = self.base[idx]
        # Uniform random 2D translation with periodic boundary conditions.
        h, w = clean.shape[-2], clean.shape[-1]
        shift_y = torch.randint(0, h, (1,)).item()
        shift_x = torch.randint(0, w, (1,)).item()
        translated = torch.roll(clean, shifts=(shift_y, shift_x), dims=(-2, -1))

        # Additive white Gaussian noise after the random translation.
        noisy = translated + self.noise_std * torch.randn_like(clean)
        return noisy, clean


# ---------- 2. Model: DiT with channel-concat conditioning ----------
# DiT expects `in_channels` input. We feed it 2 channels:
#   channel 0 = interpolated sample I_t
#   channel 1 = conditioning image (the noisy MNIST we want to clean)
# At inference, channel 1 stays fixed across all integration steps.
class ConditionalDiT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, hidden=192, depth=6, heads=6):
        super().__init__()
        self.dit = DiTTransformer2DModel(
            sample_size=image_size,
            patch_size=patch_size,
            in_channels=2,              # interpolated sample + condition
            out_channels=1,             # predicting velocity for the 1-channel target
            num_layers=depth,
            num_attention_heads=heads,
            attention_head_dim=hidden // heads,
            num_embeds_ada_norm=1000,   # t in [0,1] scaled to [0,999]
        )

    def forward(self, x_t, t, cond):
        # x_t:  (B, 1, H, W) -- interpolated sample I_t
        # cond: (B, 1, H, W) -- conditioning noisy MNIST
        # t:    (B,)         -- timestep in [0, 999] (long)
        inp = torch.cat([x_t, cond], dim=1)            # (B, 2, H, W)
        dummy_class = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t, class_labels=dummy_class).sample


# ---------- 3. Training loop ----------
def train(epochs=10, batch_size=128, lr=1e-4, device=device, noise_std=0.3, corruption="awgn"):
    ds = NoisyMNIST(train=True, noise_std=noise_std, corruption=corruption)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ConditionalDiT().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for step, (noisy_cond, clean) in enumerate(loader):
            noisy_cond = noisy_cond.to(device)   # conditioning input F(X)
            clean = clean.to(device)             # target X

            # Stochastic interpolant: I_t = (1-t)*Z + t*X
            noise = torch.randn_like(clean)                         # Z
            t = torch.rand(clean.size(0), device=device)            # t ~ U[0,1]
            t_view = t.view(-1, 1, 1, 1)
            x_t = (1 - t_view) * noise + t_view * clean            # I_t
            velocity = clean - noise                                 # dI_t/dt = X - Z

            # Scale t to DiT's ada-norm range
            t_dit = (t * 999).long()

            pred = model(x_t, t_dit, noisy_cond)
            loss = F.mse_loss(pred, velocity)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f}")

    return model


# ---------- 4. Sampling ----------
@torch.no_grad()
def sample(model, noisy_cond, device=device, num_steps=50, initial_state=None):
    """Integrate dx/dt = v_theta(x, t, cond) from t=0 to t=1 with Euler steps."""
    model.eval()
    if initial_state is not None:
        x = initial_state.to(device)
    else:
        x = torch.randn_like(noisy_cond).to(device)   # start from Z ~ N(0,I)
    noisy_cond = noisy_cond.to(device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((x.size(0),), t * 999, device=device).long()
        v = model(x, t_batch, noisy_cond)
        x = x + v * dt
    return x.clamp(-1, 1)

@torch.no_grad()
def visualize(model, train=True, noise_std=0.3, corruption="awgn"):
    model.eval()
    model_cpu = model.to('cpu')

    test_ds = NoisyMNIST(train=train, noise_std=noise_std, corruption=corruption)

    n_vis = 8
    noisy_batch = torch.stack([test_ds[i][0] for i in range(n_vis)]).to('cpu')  # (n_vis, 1, 32, 32)
    fixed_init = torch.randn(len(noisy_batch), 1, 32, 32, device='cpu')
    clean_pred = sample(model_cpu, noisy_batch, device='cpu')
    clean_pred_fixed = sample(model_cpu, noisy_batch, device='cpu', initial_state=fixed_init)

    all_images = torch.cat([noisy_batch, clean_pred, clean_pred_fixed], dim=0)
    vmin = all_images.min().item()
    vmax = all_images.max().item()

    fig, axes = plt.subplots(3, n_vis, figsize=(2 * n_vis, 6))

    row_data = [noisy_batch, clean_pred, clean_pred_fixed]
    row_titles = ['Noisy input', 'clean_pred', 'clean_pred_fixed']

    for row_idx, (data, title) in enumerate(zip(row_data, row_titles)):
        for col_idx in range(n_vis):
            ax = axes[row_idx, col_idx]
            img = data[col_idx].squeeze().detach().cpu().numpy()
            im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(title, loc='left', fontsize=11, x=-0.1, y=0.4)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"si_{corruption}_mnist_transformer_results_train_{train}_noise_std_{noise_std:.2f}.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    corruption = "awgn" # "awgn" or "mra"
    noise_std = 0.5

    print(f"Using device: {device}")
    print('Training model...')
    model = train(epochs=2, noise_std=noise_std, corruption=corruption)
    print('Training complete.')
    print('Generating samples from data in train set...')
    visualize(model, train=True, noise_std=noise_std, corruption=corruption)
    print('Generating samples complete. Generating samples from data in test set...')
    visualize(model, train=False, noise_std=noise_std, corruption=corruption)
    print('Generating samples complete. Done.')
    print('Done.')
