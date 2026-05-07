import torch

def forward_channel(x: torch.Tensor, 
                    noise_std: float,
                    p_drop: float,
                    corruption: str) -> torch.Tensor:
    '''
    Implements corruption channel 
        F_AWGN(x) = x + noise_std * torch.randn_like(x)
        F_MRA(x) = T(x) + noise_std * torch.randn_like(x)
    where T is a random 2D circular shift.

    Args:
        x: torch.Tensor, shape (B, C, H, W)
        noise_std: float
        p_drop: Probability of removing a pixel from the image (0.0 to 1.0)
        corruption: str, "awgn" or "mra"

    Returns:
        torch.Tensor, shape (B, C, H, W)
    '''
    if corruption == "awgn":
        return awgn(x, noise_std)
    elif corruption == "mra":
        x = random_translate(x)
        x = awgn(x, noise_std)
        return x
    elif corruption == "drop_mra":
        x = random_translate(x)
        x = remove_pixels(x, p_drop=p_drop)
        x = awgn(x, noise_std)
        return x
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

def awgn(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    return x + noise_std * torch.randn_like(x)

def random_translate(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    rows = torch.randint(0, H, (B,))
    cols = torch.randint(0, W, (B,))
    translated = torch.stack([
        torch.roll(x[i], shifts=(rows[i].item(), cols[i].item()), dims=(-2, -1))
        for i in range(B)
    ])
    return translated

def remove_pixels(x: torch.Tensor, p_drop: float) -> torch.Tensor:
    """
    Randomly removes pixels by setting them to zero.
    Drops the entire pixel across all channels simultaneously.
    """
    if p_drop <= 0.0:
        return x
        
    B, C, H, W = x.shape
    rand_tensor = torch.rand((B, 1, H, W), device=x.device)
    mask = rand_tensor > p_drop
    return x * mask