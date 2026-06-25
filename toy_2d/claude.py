import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Define the Velocity Network (Point Cloud / DeepSets)
# ==========================================
class VelocityMLP(nn.Module):
    """
    Permutation-equivariant velocity field over *point clouds*.

    Input:  x of shape (batch, n_points, 2), t of shape (batch, 1)
    Output: velocity of shape (batch, n_points, 2)

    Each point is processed by a shared per-point MLP, and a permutation-
    invariant mean-pooled global feature is broadcast back to every point so
    the network reasons about the cloud as a set (no dependence on point order
    or count).
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Per-point encoder: 2D coord + 1D time -> hidden feature
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Per-point decoder: [per-point feat | global feat] -> 2D velocity
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        # x: (B, N, 2), t: (B, 1)
        B, N, _ = x.shape
        # Broadcast the per-cloud time across every point -> (B, N, 1)
        t_exp = t.unsqueeze(1).expand(B, N, 1)
        h = self.encoder(torch.cat([x, t_exp], dim=-1))            # (B, N, H)
        # Global context via permutation-invariant mean pooling
        g = h.mean(dim=1, keepdim=True).expand(-1, N, -1)          # (B, N, H)
        return self.decoder(torch.cat([h, g], dim=-1))            # (B, N, 2)

# ==========================================
# 2. Data Generation & Forward Map
# ==========================================
def generate_square(n_clouds=512, n_points=256, base_noise=0.05, random_rotation=True):
    """
    Generates a *dataset of* 2D square-outline point clouds.

    Points are sampled uniformly along the perimeter of an axis-aligned square
    centered at the origin (corners at (+/-1, +/-1)). Returns a tensor of shape
    (n_clouds, n_points, 2). Each cloud is an independent draw (independent point
    sampling + noise, and optionally a random rotation) so the model learns a
    genuine distribution over clouds.
    """
    clouds = []
    for _ in range(n_clouds):
        # Uniform position along the perimeter: u in [0, 4) selects side + offset
        u = np.random.rand(n_points) * 4.0
        side = np.floor(u).astype(int)        # which edge: 0,1,2,3
        p = (u - side) * 2.0 - 1.0            # position along the edge in [-1, 1)

        x = np.empty(n_points)
        y = np.empty(n_points)
        s0, s1, s2, s3 = side == 0, side == 1, side == 2, side == 3
        x[s0], y[s0] = p[s0], -1.0            # bottom edge
        x[s1], y[s1] = 1.0, p[s1]             # right edge
        x[s2], y[s2] = p[s2], 1.0             # top edge
        x[s3], y[s3] = -1.0, p[s3]            # left edge

        # Add minor noise to give the "clean" manifold some volume
        x += np.random.randn(n_points) * base_noise
        y += np.random.randn(n_points) * base_noise

        cloud = np.stack([x, y], axis=1)

        if random_rotation:
            angle = np.random.rand() * 2 * np.pi
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            cloud = cloud @ rot.T

        clouds.append(cloud)

    data = np.stack(clouds, axis=0)  # (n_clouds, n_points, 2)

    # Normalize globally to mean 0, std 1 per coordinate (crucial for stability)
    data = (data - data.mean(axis=(0, 1))) / data.std(axis=(0, 1))
    return torch.tensor(data, dtype=torch.float32)

def forward_map(x, noise_level=0.75, rotate=True):
    """
    Forward Map: per-cloud random rotation about the origin + AWGN.
    Operates on point clouds of shape (batch, n_points, 2).
    """
    if rotate:
        x = forward_map_rotate(x, noise_level)
    else:
        x = forward_map_awgn(x, noise_level)
    return x

def forward_map_awgn(x, noise_level=0.75):
    """AWGN Channel (Black-box corruption)."""
    return x + noise_level * torch.randn_like(x)

def forward_map_rotate(x, noise_level=0.5, max_angle=2 * np.pi):
    """
    Forward Map: each *cloud* is rigidly rotated about the origin by an
    independent angle drawn uniformly from [0, max_angle), then AWGN is added.
    """
    B = x.shape[0]
    # One rotation angle per cloud -> (B,)
    angles = torch.rand(B, device=x.device) * max_angle
    c, s = torch.cos(angles), torch.sin(angles)
    # Per-cloud rotation matrices -> (B, 2, 2)
    rot = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)
    # Rotate every point in each cloud: (B, N, 2) @ (B, 2, 2)^T -> (B, N, 2)
    x_rot = torch.matmul(x, rot.transpose(-1, -2))

    return x_rot + noise_level * torch.randn_like(x)

# ==========================================
# 3. ODE Solver
# ==========================================
def solve_ode_backward(model, y, steps=32):
    x = y.clone()
    dt = 1.0 / steps

    for i in range(steps, 0, -1):
        t_val = i / steps
        # One scalar time per cloud -> (B, 1)
        t_tensor = torch.full((x.shape[0], 1), t_val, device=x.device)

        with torch.no_grad():
            velocity = model(x, t_tensor)

        x = x - velocity * dt

    return x

# ==========================================
# 4. Training Loop (with Pre-training Warmup)
# ==========================================
def train_scsi(n_clouds=512, n_points=256, noise_level=0.5, rotate=True,
               warmup_epochs=30, main_epochs=120, lr=1e-3, max_grad_norm=1.0,
               batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generate clean square clouds in a *canonical* orientation -> (n_clouds, n_points, 2)
    #    (the prior has no rotation so the forward map's rotation is identifiable)
    clean_data = generate_square(n_clouds=n_clouds, n_points=n_points,
                                 random_rotation=False).to(device)

    # 2. Generate corrupted observations (per-cloud random rotation + AWGN)
    observed_data = forward_map(clean_data, noise_level=noise_level, rotate=rotate)

    dataset = torch.utils.data.TensorDataset(observed_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VelocityMLP(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    warmup_noise_level = 1.0  # Extra noise added strictly for warmup

    model.train()

    print("--- Starting Warmup Phase ---")
    for epoch in range(warmup_epochs):
        epoch_loss = 0.0
        for (batch_y,) in dataloader:
            batch_y = batch_y.to(device)               # (B, N, 2)
            batch_size_cur = batch_y.shape[0]

            # WARMUP: Create a noisier version of the observed clouds
            extra_noise = torch.randn_like(batch_y) * warmup_noise_level
            noisier_y = batch_y + extra_noise

            # WARMUP Interpolant: t=0 is batch_y, t=1 is noisier_y
            t = torch.rand(batch_size_cur, 1, device=device)        # (B, 1)
            t_b = t.view(batch_size_cur, 1, 1)                      # (B, 1, 1) for broadcast
            I_t = (1 - t_b) * batch_y + t_b * noisier_y

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
            batch_y = batch_y.to(device)               # (B, N, 2)
            batch_size_cur = batch_y.shape[0]

            # Step 1: Backward ODE transport to get current clean cloud estimates
            # Because of warmup, this is much more stable right out of the gate.
            x_hat = solve_ode_backward(model, batch_y, steps=10)

            # Step 2: Push through forward map to simulate observation
            y_tilde = forward_map(x_hat, noise_level=noise_level, rotate=rotate)

            # Step 3: Interpolant and target matching
            t = torch.rand(batch_size_cur, 1, device=device)        # (B, 1)
            t_b = t.view(batch_size_cur, 1, 1)                      # (B, 1, 1)
            I_t = (1 - t_b) * x_hat + t_b * y_tilde
            target_velocity = y_tilde - x_hat

            # Step 4: Regress
            pred_velocity = model(I_t, t)
            loss = torch.mean((pred_velocity - target_velocity) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Main Epoch {epoch+1}/{main_epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    return model, clean_data, observed_data

# ==========================================
# 5. Visualization
# ==========================================
if __name__ == "__main__":
    n_clouds_to_generate = 512
    n_points_per_cloud = 256
    noise_level = 0.1
    warmup_epochs = 0
    main_epochs = 200
    rotate = True
    lr = 4e-4
    max_grad_norm = 0.2

    print(f"Training SCSI Model on {n_clouds_to_generate} square point clouds "
          f"({n_points_per_cloud} points each)...")
    trained_model, true_clean, corrupted_obs = train_scsi(
        n_clouds=n_clouds_to_generate,
        n_points=n_points_per_cloud,
        warmup_epochs=warmup_epochs,
        main_epochs=main_epochs,
        noise_level=noise_level,
        rotate=rotate,
        lr=lr,
        max_grad_norm=max_grad_norm,
    )
    trained_model.eval()

    print("Restoring corrupted clouds...")
    # Map observations back to the clean space
    restored = solve_ode_backward(trained_model, corrupted_obs, steps=64)

    # Plot a single representative cloud across the three stages
    idx = 0
    true_cloud = true_clean[idx].cpu().numpy()
    corr_cloud = corrupted_obs[idx].cpu().numpy()
    rest_cloud = restored[idx].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(true_cloud[:, 0], true_cloud[:, 1], alpha=0.5, s=5)
    axs[0].set_title("True (Unobserved) Square Cloud")

    axs[1].scatter(corr_cloud[:, 0], corr_cloud[:, 1], alpha=0.5, s=5, color='orange')
    axs[1].set_title("Corrupted Observation (rotation + AWGN)")

    axs[2].scatter(rest_cloud[:, 0], rest_cloud[:, 1], alpha=0.5, s=5, color='green')
    axs[2].set_title("SCSI Restored Point Cloud")

    for ax in axs:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
