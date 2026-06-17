import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Define the Velocity Network
# ==========================================
class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Input: 2D coordinate + 1D time
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ==========================================
# 2. Data Generation & Forward Map
# ==========================================
def generate_spiral(n_points=2000, base_noise=0.05):
    """Generates a 2D Archimedean spiral point cloud."""
    # Square root of rand ensures uniform point distribution along the spiral
    theta = np.sqrt(np.random.rand(n_points)) * 4 * np.pi 
    r = theta + np.pi
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Add minor noise to give the "clean" manifold some volume
    x += np.random.randn(n_points) * base_noise
    y += np.random.randn(n_points) * base_noise
    
    data = np.stack([x, y], axis=1)
    
    # Normalize to mean 0, std 1 (crucial for MLP stability)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)

def forward_map(x, noise_level=0.75, flip_prob=0.5):
    """
    Forward Map: 50% chance to flip about the x-axis + AWGN.
    """
    if flip_prob > 0.0:
        x = forward_map_flip(x, noise_level, flip_prob)
    else:
        x = forward_map_awgn(x, noise_level)
    return x

def forward_map_awgn(x, noise_level=0.75):
    """AWGN Channel (Black-box corruption)."""
    return x + noise_level * torch.randn_like(x)

def forward_map_flip(x, noise_level=0.5, flip_prob=0.5):
    """
    Forward Map: 50% chance to flip about the x-axis + AWGN.
    """
    # Create a boolean mask where ~50% of the values are True
    flip_mask = torch.rand(x.shape[0], 1, device=x.device) < flip_prob
    
    # Create a multiplier tensor: [1, -1] for flipped, [1, 1] for normal
    multiplier = torch.ones_like(x)
    
    # Apply the flip to the y-coordinates based on the mask
    # We use squeeze() to ensure dimensions match
    multiplier[:, 1] = torch.where(flip_mask.squeeze(), -1.0, 1.0)
    
    # Apply the multiplier to flip, then add the Gaussian noise
    corrupted_x = x * multiplier + noise_level * torch.randn_like(x)
    
    return corrupted_x

# ==========================================
# 3. ODE Solver
# ==========================================
def solve_ode_backward(model, y, steps=32):
    x = y.clone()
    dt = 1.0 / steps
    
    for i in range(steps, 0, -1):
        t_val = i / steps
        t_tensor = torch.full((x.shape[0], 1), t_val, device=x.device)
        
        with torch.no_grad():
            velocity = model(x, t_tensor)
            
        x = x - velocity * dt
        
    return x

# ==========================================
# 4. Training Loop
# ==========================================
# ==========================================
# 4. Training Loop (with Pre-training Warmup)
# ==========================================
def train_scsi(n_points=2500, noise_level=0.5, flip_prob=0.5, warmup_epochs=30, main_epochs=120):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate clean spiral 
    clean_data = generate_spiral(n_points=n_points).to(device)
    
    # 2. Generate corrupted observations (50% flip + AWGN)
    observed_data = forward_map(clean_data, noise_level=noise_level, flip_prob=flip_prob)
    
    dataset = torch.utils.data.TensorDataset(observed_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model = VelocityMLP(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    warmup_noise_level = 1.0 # Extra noise added strictly for warmup
    
    model.train()
    
    print("--- Starting Warmup Phase ---")
    for epoch in range(warmup_epochs):
        epoch_loss = 0.0
        for (batch_y,) in dataloader:
            batch_y = batch_y.to(device)
            batch_size = batch_y.shape[0]
            
            # WARMUP: Create a noisier version of the observed data
            extra_noise = torch.randn_like(batch_y) * warmup_noise_level
            noisier_y = batch_y + extra_noise
            
            # WARMUP Interpolant: t=0 is batch_y, t=1 is noisier_y
            t = torch.rand(batch_size, 1, device=device)
            I_t = (1 - t) * batch_y + t * noisier_y
            
            # Target velocity is simply the extra noise
            target_velocity = noisier_y - batch_y 
            
            pred_velocity = model(I_t, t)
            loss = torch.mean((pred_velocity - target_velocity) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Warmup Epoch {epoch+1}/{warmup_epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    print("\n--- Starting Main SCSI Phase ---")
    for epoch in range(main_epochs):
        epoch_loss = 0.0
        for (batch_y,) in dataloader:
            batch_y = batch_y.to(device)
            batch_size = batch_y.shape[0]
            
            # Step 1: Backward ODE transport to get current clean estimates
            # Because of warmup, this will be much more stable right out of the gate!
            x_hat = solve_ode_backward(model, batch_y, steps=10) 
            
            # Step 2: Push through forward map to simulate observation
            y_tilde = forward_map(x_hat, noise_level=noise_level, flip_prob=flip_prob)
            
            # Step 3: Interpolant and target matching
            t = torch.rand(batch_size, 1, device=device)
            I_t = (1 - t) * x_hat + t * y_tilde
            target_velocity = y_tilde - x_hat
            
            # Step 4: Regress
            pred_velocity = model(I_t, t)
            loss = torch.mean((pred_velocity - target_velocity) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Main Epoch {epoch+1}/{main_epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    return model, clean_data, observed_data

# ==========================================
# 5. Visualization
# ==========================================
if __name__ == "__main__":
    n_points_to_generate = 2500
    noise_level = 0.2
    warmup_epochs = 1000
    main_epochs = 250
    flip_prob = 0.5

    print(f"Training SCSI Model on Spiral with {n_points_to_generate} points...")
    trained_model, true_clean, corrupted_obs = train_scsi(n_points=n_points_to_generate, warmup_epochs=warmup_epochs, main_epochs=main_epochs, noise_level=noise_level, flip_prob=flip_prob)
    trained_model.eval()
    
    print("Restoring corrupted samples...")
    # Map observations back to the clean space
    restored_samples = solve_ode_backward(trained_model, corrupted_obs, steps=64) 
    
    # Plotting
    true_clean = true_clean.cpu().numpy()
    corrupted_obs = corrupted_obs.cpu().numpy()
    restored_samples = restored_samples.cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(true_clean[:, 0], true_clean[:, 1], alpha=0.5, s=5)
    axs[0].set_title("True (Unobserved) Spiral")
    
    axs[1].scatter(corrupted_obs[:, 0], corrupted_obs[:, 1], alpha=0.5, s=5, color='orange')
    axs[1].set_title("Corrupted Observations (AWGN)")
    
    axs[2].scatter(restored_samples[:, 0], restored_samples[:, 1], alpha=0.5, s=5, color='green')
    axs[2].set_title("SCSI Restored Point Cloud")
    
    for ax in axs:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.show()