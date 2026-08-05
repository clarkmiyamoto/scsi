import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from si import wrap_to_pi, axis_angle_to_matrix, gram_schmidt_to_matrix, matrix_to_gram_schmidt


def sample_uniform_angle(B: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Haar-uniform angle on SO(2), drawn as the literal `z ~ N(0,1)` noise the pseudocode calls
    for: a 2-D isotropic Gaussian direction is exactly uniform on the unit circle.

    Returns:
        theta: (B,) in (-pi, pi]
    """
    z = torch.randn(B, 2, device=device)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.atan2(z[:, 1], z[:, 0])


def sample_tilt_series_angles(
    n_acquisitions: int, n_tilts: int, tilt_increment: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    One CryoET-style tilt series per acquisition: T = n_tilts angles, evenly spaced by
    `tilt_increment` radians, starting from an independent Haar-uniform random offset per
    acquisition (so the dataset doesn't share one fixed set of absolute angles).

    Args:
        n_acquisitions: how many independent tilt series to draw (e.g. one GT object imaged
            `corruptions_per_object` times is `corruptions_per_object` acquisitions).
        n_tilts: T, number of tilts within one series.
        tilt_increment: angular step between consecutive tilts, in radians.

    Returns:
        theta: (n_acquisitions, n_tilts) in (-pi, pi]
    """
    theta0 = sample_uniform_angle(n_acquisitions, device)                       # (n_acq,)
    steps = torch.arange(n_tilts, device=device, dtype=theta0.dtype) * tilt_increment  # (T,)
    return wrap_to_pi(theta0[:, None] + steps[None, :])                        # (n_acq, T)


def rotate_2d(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate each (1, H, W) image in the batch by its own angle, filling the background with
    true black (-1), not grid_sample's raw zero-padding (which would be mid-gray in [-1,1]
    space). Done by rotating in a shifted (x+1) frame where zero-padding IS the background,
    then shifting back.

    Args:
        x: (B, 1, H, W) in [-1, 1]
        theta: (B,) radians

    Returns:
        (B, 1, H, W) in [-1, 1]
    """
    B = x.size(0)
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(theta)
    rot = torch.stack([
        torch.stack([cos, -sin, zeros], dim=-1),
        torch.stack([sin,  cos, zeros], dim=-1),
    ], dim=-2)  # (B, 2, 3)

    grid = F.affine_grid(rot, x.shape, align_corners=True)
    x_shifted = x + 1.0  # background -1 -> 0, matches zeros padding
    x_rot = F.grid_sample(x_shifted, grid, align_corners=True,
                          mode="bilinear", padding_mode="zeros")
    return x_rot - 1.0


def project_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel-beam projection: integrate along a fixed axis (height). Rotation supplies the
    view diversity, same fixed-projection-axis convention as simple_3d's radon_projection.

    Args:
        x: (B, 1, H, W)

    Returns:
        (B, 1, W)
    """
    return x.sum(dim=-2)


def forward_channel(
    x: torch.Tensor,
    noise_std: float,
    theta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CryoEM-style 2D->1D channel: random in-plane rotation (black fill) -> 1D projection -> AWGN.

    Args:
        x: (B, 1, H, W)
        noise_std: float
        theta: (B,) radians. If None, a fresh Haar-uniform angle is drawn per sample (used
            for synthesizing y_obs). If given, that angle is reused instead of drawing a new
            one — this is how a pool's recovered R_hat is fed back through F for ŷ = F(x̂; R̂).

    Returns:
        y: (B, 1, W)
        theta_used: (B,)
    """
    B = x.size(0)
    if theta is None:
        theta = sample_uniform_angle(B, x.device)
    x_rot = rotate_2d(x, theta)
    y = project_1d(x_rot)
    y = y + noise_std * torch.randn_like(y)
    return y, theta


########################################################
# Rotational MRA channel: same in-plane SO(2) rotation as forward_channel, but WITHOUT the 1D
# projection step -- F(x) = R∘x + W stays a full (B,1,H,W) image (classical multi-reference
# alignment: many independent noisy rotated copies of one signal, no tilt-series / shared-
# acquisition structure -- see data.build_observations_mra). forward_channel_mra reuses
# rotate_2d and sample_uniform_angle verbatim; it's forward_channel with project_1d removed,
# not a new rotation mechanism.
########################################################

def mask_to_disk(x: torch.Tensor, background: float = -1.0) -> torch.Tensor:
    """
    Zero out (set to `background`) every pixel outside the inscribed circle of a square image
    -- the largest circle that fits inside the (H, W) canvas, radius 1 in the same normalized
    [-1, 1] coordinates rotate_2d's affine_grid uses.

    Needed as a precondition for forward_channel_mra, NOT enforced by it: rotate_2d rotates a
    SQUARE canvas inside a SQUARE sampling grid (F.affine_grid/F.grid_sample). Rotation
    preserves radius, so an output pixel at radius r always samples the source at that same
    radius r -- but the SQUARE boundary is not circularly symmetric (it reaches out to radius
    sqrt(2) at the corners, only radius 1 at the edge midpoints), so which annulus-r positions
    fall inside vs. outside the source square depends on theta. Pixels outside the square get
    grid_sample's zero-padding (exactly -1 after rotate_2d's shifted-frame trick) -- a boundary
    whose exact shape is therefore a deterministic function of theta mod 90 degrees, independent
    of image content. forward_channel's project_1d sums this away into a smooth per-column bias;
    forward_channel_mra has no projection, so it would otherwise leak a second, content-
    independent cue to theta straight into y.

    Masking the SOURCE to the inscribed disk (radius <= 1) removes this leak entirely: every
    rotation of a disk-masked image keeps all real content within that same disk (rotation
    preserves radius), so the annulus between radius 1 and sqrt(2) is uniformly exactly
    `background` in the source for EVERY theta -- matching grid_sample's zero-pad value exactly,
    so the valid-vs-zero-padded boundary becomes indistinguishable from ordinary background
    regardless of rotation. See data.load_mnist_subset_mra, the caller that applies this once at
    GT-load time.

    Args:
        x: (..., H, W), assumed square in the last two dims.
        background: fill value outside the disk (always -1 here, matching [-1,1]-normalized
            MNIST).

    Returns:
        Same shape as x.
    """
    H, W = x.shape[-2], x.shape[-1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    inside = (xx ** 2 + yy ** 2) <= 1.0
    return torch.where(inside, x, torch.full_like(x, background))


def forward_channel_mra(
    x: torch.Tensor,
    noise_std: float,
    theta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotational MRA channel: F(x) = R∘x + W -- random in-plane rotation (black fill, via
    rotate_2d) followed directly by full-image AWGN, no projection. y stays (B, 1, H, W), the
    same shape as x (unlike forward_channel's (B, 1, W) 1D output).

    Callers must supply an x that's already been through mask_to_disk (see
    data.load_mnist_subset_mra) -- this function doesn't enforce that itself, but the "no
    content-independent pose leak" property documented on mask_to_disk only holds if it has.

    Args:
        x: (B, 1, H, W)
        noise_std: float
        theta: (B,) radians. If None, a fresh Haar-uniform angle is drawn per sample (used
            for synthesizing y_obs). If given, that angle is reused instead of drawing a new
            one — this is how a pool's recovered R_hat is fed back through F for ŷ = F(x̂; R̂).

    Returns:
        y: (B, 1, H, W)
        theta_used: (B,)
    """
    B = x.size(0)
    if theta is None:
        theta = sample_uniform_angle(B, x.device)
    x_rot = rotate_2d(x, theta)
    y = x_rot + noise_std * torch.randn_like(x_rot)
    return y, theta


########################################################
# CryoET-style 3D->2D channel: SO(3) rotation -> 2D projection -> AWGN. Rotation is always
# represented as a (B,3,3) matrix INTERNALLY in this module (composition of the mount+tilt
# rotations below is plain matrix multiplication — no quaternions anywhere here); the public
# boundary of forward_channel_3d uses the (B,6) Gram-Schmidt representation, matching what the
# model/interpolant consume (see si.py's module docstring for the full rationale).
########################################################

def sample_uniform_rotation_so3(B: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Haar-uniform rotation on SO(3), as a (B, 3, 3) matrix, via
    scipy.spatial.transform.Rotation.random — the same precedent already used by
    simple_3d/corruption.py and toy_3d/*.py.
    """
    R_np = Rotation.random(B).as_matrix().astype("float32")
    return torch.from_numpy(R_np).to(device)


def sample_tilt_series_rotations_so3(
    n_acquisitions: int, n_tilts: int, tilt_increment: float,
    tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    3D analogue of sample_tilt_series_angles: one random Haar-uniform "mount" rotation per
    acquisition (unknown specimen orientation, applied FIRST/inner), composed with T
    evenly-spaced rotations about a FIXED lab-frame tilt axis (applied SECOND/outer), starting
    from an independent Haar-uniform-derived random offset per acquisition — generalizing the
    2D case, where "mount" and "tilt axis" collapse into the same single SO(2) freedom.

    R_total = R_tilt(angle_i) @ R_mount — tilt OUTER (lab frame), mount INNER (specimen frame,
    applied first: the specimen is mounted at a random unknown orientation FIRST; the holder
    then tilts about ITS OWN fixed physical axis SECOND). Composition order is verified by
    test_so3_math.py — swapping it breaks the single-tilt-axis property (each acquisition would
    tilt about a different physical axis, no longer a real tilt series).

    Args:
        n_acquisitions: how many independent tilt series to draw.
        n_tilts: T, number of tilts within one series.
        tilt_increment: angular step between consecutive tilts, in radians.
        tilt_axis: the FIXED physical tilt axis, in lab-frame coordinates.

    Returns:
        R: (n_acquisitions, n_tilts, 3, 3)
    """
    R_mount = sample_uniform_rotation_so3(n_acquisitions, device)             # (n_acq, 3, 3)
    start = torch.rand(n_acquisitions, device=device) * 2 * torch.pi          # (n_acq,) Uniform(0,2pi)
    steps = torch.arange(n_tilts, device=device, dtype=start.dtype) * tilt_increment  # (T,)
    angles = start[:, None] + steps[None, :]                                  # (n_acq, T)

    axis = torch.tensor(tilt_axis, device=device, dtype=start.dtype)
    R_tilt = axis_angle_to_matrix(axis, angles.reshape(-1))                    # (n_acq*T, 3, 3)
    R_tilt = R_tilt.reshape(n_acquisitions, n_tilts, 3, 3)

    R_mount_exp = R_mount[:, None, :, :].expand(-1, n_tilts, -1, -1)          # (n_acq, T, 3, 3)
    return torch.matmul(R_tilt, R_mount_exp)


def rotate_3d(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate each (1, D, H, W) volume in the batch by its own rotation matrix, black-fill
    background with true black (-1) via the same shifted-frame trick as rotate_2d — deliberately
    NOT simple_3d's padding_mode='zeros' without the shift, which silently rotates background
    into false 0-density.

    Args:
        x: (B, 1, D, H, W) in [-1, 1]
        R: (B, 3, 3)

    Returns:
        (B, 1, D, H, W) in [-1, 1]
    """
    zeros = torch.zeros(R.size(0), 3, 1, device=x.device, dtype=R.dtype)
    theta = torch.cat([R, zeros], dim=2)  # (B, 3, 4)

    grid = F.affine_grid(theta, x.shape, align_corners=True)
    x_shifted = x + 1.0
    x_rot = F.grid_sample(x_shifted, grid, align_corners=True,
                          mode="bilinear", padding_mode="zeros")
    return x_rot - 1.0


def project_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Parallel-beam projection: integrate along the depth axis.

    Args:
        x: (B, 1, D, H, W)

    Returns:
        (B, 1, H, W)
    """
    return x.sum(dim=2)


def forward_channel_3d(
    x: torch.Tensor,
    noise_std: float,
    pose6: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CryoET-style 3D->2D channel: SO(3) rotation (black fill) -> 2D projection -> AWGN.

    Args:
        x: (B, 1, D, H, W)
        noise_std: float
        pose6: (B, 6) Gram-Schmidt rotation representation. If None, a fresh Haar-uniform
            rotation is drawn per sample (used for synthesizing y_obs). If given, that rotation
            is reused instead of drawing a new one — this is how a pool's recovered pose6_hat
            is fed back through F for ŷ = F(x̂; pose6_hat). Converted to a (B,3,3) matrix via
            si.gram_schmidt_to_matrix ONLY internally, to call rotate_3d — the public contract
            stays (B,6) throughout, matching what the model/interpolant consume.

    Returns:
        y: (B, 1, H, W)
        pose6_used: (B, 6)
    """
    B = x.size(0)
    if pose6 is None:
        R = sample_uniform_rotation_so3(B, x.device)
        pose6 = matrix_to_gram_schmidt(R)
    else:
        R = gram_schmidt_to_matrix(pose6)
    x_rot = rotate_3d(x, R)
    y = project_2d(x_rot)
    y = y + noise_std * torch.randn_like(y)
    return y, pose6
