"""
Conditional Diffusion Transformer for MNIST denoising.

Setup: noisy MNIST image  -->  clean MNIST image
The conditioning image is concatenated channel-wise with the noisy latent
at each diffusion step.

Requirements:
    pip install torch torchvision diffusers accelerate
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DiTTransformer2DModel, DDPMScheduler

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
# Trick: DiT expects `in_channels` input. We feed it 2 channels:
#   channel 0 = current noisy latent x_t  (the thing being denoised)
#   channel 1 = conditioning image        (the noisy MNIST we want to clean)
# At inference, channel 1 stays fixed across all diffusion steps.
class ConditionalDiT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, hidden=192, depth=6, heads=6):
        super().__init__()
        self.dit = DiTTransformer2DModel(
            sample_size=image_size,
            patch_size=patch_size,
            in_channels=2,              # noisy latent + condition
            out_channels=1,             # predicting noise for the 1-channel target
            num_layers=depth,
            num_attention_heads=heads,
            attention_head_dim=hidden // heads,
            num_embeds_ada_norm=1000,   # diffusion timesteps
        )

    def forward(self, x_t, t, cond):
        # x_t:   (B, 1, H, W) -- current noisy sample in the diffusion chain
        # cond:  (B, 1, H, W) -- the noisy MNIST we are conditioning on
        # t:     (B,)         -- diffusion timestep
        inp = torch.cat([x_t, cond], dim=1)            # (B, 2, H, W)
        # DiT uses class_labels for adaLN conditioning; we pass dummy zeros
        # since our conditioning is via the concat'd channel, not class info.
        dummy_class = torch.zeros(x_t.size(0), dtype=torch.long, device=x_t.device)
        return self.dit(inp, timestep=t, class_labels=dummy_class).sample


# ---------- 3. Training loop ----------
def train(epochs=10, batch_size=128, lr=1e-4, device=device, noise_std=0.3, corruption="awgn"):
    ds = NoisyMNIST(train=True, noise_std=noise_std, corruption=corruption)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ConditionalDiT().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for step, (noisy_cond, clean) in enumerate(loader):
            noisy_cond = noisy_cond.to(device)   # the conditioning input
            clean = clean.to(device)             # the target we want to recover

            # Standard diffusion training: add noise to the *clean* image
            noise = torch.randn_like(clean)
            t = torch.randint(0, scheduler.config.num_train_timesteps,
                              (clean.size(0),), device=device).long()
            x_t = scheduler.add_noise(clean, noise, t)

            # Predict the noise, conditioned on the noisy MNIST
            pred = model(x_t, t, noisy_cond)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f}")

    return model, scheduler


# ---------- 4. Sampling ----------
@torch.no_grad()
def sample(model, scheduler, noisy_cond, device=device, num_steps=50, initial_state=None):
    """Given a noisy MNIST batch, generate clean versions via DDPM sampling."""
    model.eval()
    scheduler.set_timesteps(num_steps)
    if initial_state is not None:
        x = initial_state.to(device)
    else:
        x = torch.randn_like(noisy_cond).to(device)   # start from pure noise
    noisy_cond = noisy_cond.to(device)

    for t in scheduler.timesteps:
        t_batch = t.expand(x.size(0)).to(device)
        eps = model(x, t_batch, noisy_cond)
        x = scheduler.step(eps, t, x).prev_sample
    return x.clamp(-1, 1)

@torch.no_grad()
def visualize(model, scheduler, train=True, noise_std=0.3, corruption="awgn"):
    model.eval()
    model_cpu = model.to('cpu')  # move to CPU for visualization

    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to('cpu')
    if hasattr(scheduler, 'betas'):
        scheduler.betas = scheduler.betas.to('cpu')
    if hasattr(scheduler, 'alphas'):
        scheduler.alphas = scheduler.alphas.to('cpu')

    test_ds = NoisyMNIST(train=train, noise_std=noise_std, corruption=corruption)

    n_vis = 8
    noisy_batch = torch.stack([test_ds[i][0] for i in range(n_vis)]).to('cpu')  # (n_vis, 1, 32, 32)
    fixed_init = torch.randn(len(noisy_batch), 1, 32, 32, device='cpu')  # same initial noise for all samples
    clean_pred = sample(model_cpu, scheduler, noisy_batch, device='cpu')
    clean_pred_fixed = sample(model_cpu, scheduler, noisy_batch, device='cpu', initial_state=fixed_init)

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
    # Add a shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"{corruption}_mnist_transformer_results_train_{train}_noise_std_{noise_std:.2f}.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    corruption = "awgn"
    noise_std = 0.5

    print(f"Using device: {device}")
    print('Training model...')
    model, scheduler = train(epochs=2, noise_std=noise_std, corruption=corruption)
    print('Training complete.')
    print('Generating samples from data in train set...')
    visualize(model, scheduler, train=True, noise_std=noise_std, corruption=corruption)
    print('Generating samples complete. Generating samples from data in test set...')
    visualize(model, scheduler, train=False, noise_std=noise_std, corruption=corruption)
    print('Generating samples complete. Done.')
    print('Done.')