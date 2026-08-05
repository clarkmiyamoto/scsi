import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from model import IMAGE_SIZE, VOL_SIZE
from corruption import (
    forward_channel, sample_tilt_series_angles,
    forward_channel_3d, sample_tilt_series_rotations_so3,
    forward_channel_mra, mask_to_disk,
)
from si import matrix_to_gram_schmidt


def load_mnist_subset(n_images_per_class: int, image_size: int = IMAGE_SIZE,
                      digit_classes: list[int] | None = None,
                      seed: int | None = None,
                      train: bool = True) -> torch.Tensor:
    """
    Load n_images_per_class random MNIST digits from EACH class in digit_classes,
    normalized to [-1, 1], concatenated in digit_classes order.

    Args:
        n_images_per_class: how many images to draw per class (not a total count).
        digit_classes: which MNIST classes (0-9) to draw from. None = all 10 classes.
        train: which MNIST split to draw from. True = the 60k-image train split
            (pretrain*.py's convention), False = the 10k-image test split (main*.py's
            convention) -- this is what keeps the pretraining pool and the EM dataset
            disjoint by construction. Every caller must pass this explicitly; the default
            only exists as a safety net for callers not yet updated. See
            mnist_cryoem/CLAUDE.md's dataset-split convention.
    """
    if digit_classes is None:
        digit_classes = list(range(10))
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST("./data", train=train, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    chunks = []
    for digit_class in digit_classes:
        class_idx = (dataset.targets == digit_class).nonzero(as_tuple=True)[0]
        loader = DataLoader(Subset(dataset, class_idx), batch_size=n_images_per_class,
                            shuffle=True, generator=generator)
        x_c, _ = next(iter(loader))  # (n_images_per_class, 1, image_size, image_size)
        chunks.append(x_c)
    return torch.cat(chunks, dim=0)


def build_observations(
    x_gt: torch.Tensor,
    corruptions_per_object: int,
    n_tilts: int,
    tilt_increment_deg: float,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the CryoEM channel F_CryoET to each GT object `corruptions_per_object` times —
    each repetition is an independent ACQUISITION: a full tilt series of `n_tilts` projections
    at evenly-spaced angles (step `tilt_increment_deg` degrees) from an independent random
    per-acquisition start offset, each tilt with its own independent AWGN. So a single object
    produces `corruptions_per_object * n_tilts` distinct observations (multiple tilt-series
    "shots" of the same particle).

    Args:
        x_gt: (n_images, 1, H, W)
        corruptions_per_object: number of independent tilt-series acquisitions per GT object
        n_tilts: T, number of tilts (projections) within one acquisition's series
        tilt_increment_deg: angular step between consecutive tilts, in degrees
        noise_std: float

    Returns:
        y_obs:      (n_images * corruptions_per_object * n_tilts, 1, W) — the observation set
            (this is mu; used for training)
        theta_star: (n_images * corruptions_per_object * n_tilts,)      — the true rotation
            used per observation. Diagnostic ONLY — never fed to the model, since the
            algorithm must recover the prior without knowing the true pose. Kept for eval
            plots / circular-error metrics.
        image_idx:  (n_images * corruptions_per_object * n_tilts,)      — which source object
            each observation came from, for grouped visualization.
        acq_idx:    (n_images * corruptions_per_object * n_tilts,)      — which acquisition
            (0..n_acq-1) each observation belongs to. Observations from the same acquisition
            are contiguous and already in tilt order (see `x_expanded`/`theta_flat` below),
            so `y_obs[acq_idx == a]` is one full tilt series in order — used to visualize the
            tilt-series structure (wandb_logging.py::log_reconstruction_grid) instead of an
            independent random draw over the flattened pool.
    """
    n_images = x_gt.size(0)
    n_acq = n_images * corruptions_per_object
    tilt_increment = tilt_increment_deg * torch.pi / 180.0

    x_per_acq = x_gt.repeat_interleave(corruptions_per_object, dim=0)         # (n_acq, 1, H, W)
    theta = sample_tilt_series_angles(n_acq, n_tilts, tilt_increment, x_gt.device)  # (n_acq, T)

    x_expanded = x_per_acq.repeat_interleave(n_tilts, dim=0)                  # (n_acq*T, 1, H, W)
    theta_flat = theta.reshape(-1)                                           # (n_acq*T,)
    y_obs, theta_star = forward_channel(x_expanded, noise_std=noise_std, theta=theta_flat)
    image_idx = torch.arange(n_images).repeat_interleave(corruptions_per_object * n_tilts)
    acq_idx = torch.arange(n_acq).repeat_interleave(n_tilts)
    return y_obs, theta_star, image_idx, acq_idx


def load_mnist_subset_mra(n_images_per_class: int, image_size: int = IMAGE_SIZE,
                          digit_classes: list[int] | None = None,
                          seed: int | None = None,
                          train: bool = True) -> torch.Tensor:
    """
    MRA analogue of load_mnist_subset: same MNIST load, then additionally masks each digit to
    its inscribed disk (corruption.mask_to_disk) -- REQUIRED for forward_channel_mra to satisfy
    the literal F(x) = R∘x + W spec without leaking theta through rotate_2d's square-canvas
    corner geometry (see corruption.mask_to_disk's docstring for the full geometric argument).
    Mirrors how load_mnist_volumes_3d owns its own channel-specific GT geometry fix-up (a margin
    there, to avoid SO(3)-rotation clipping; a mask here, to avoid SO(2)-rotation pose leakage)
    rather than baking either into the shared load_mnist_subset. The CryoEM/CryoET pipelines
    never needed this: rotate_2d/rotate_3d's corner artifacts get summed away by
    project_1d/project_2d; forward_channel_mra has no projection step, so the artifact would
    otherwise reach y directly.

    `train` is forwarded to load_mnist_subset unmodified -- see its docstring for the
    train/test split convention.
    """
    x = load_mnist_subset(n_images_per_class, image_size=image_size,
                          digit_classes=digit_classes, seed=seed, train=train)
    return mask_to_disk(x)


def build_observations_mra(
    x_gt: torch.Tensor,
    corruptions_per_object: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the rotational-MRA channel F_MRA to each GT object `corruptions_per_object` times --
    each repetition is a fully INDEPENDENT observation (its own Haar-uniform rotation, its own
    AWGN draw). Unlike build_observations/build_observations_3d, there is no tilt-series /
    acquisition structure here: classical MRA has no shared-axis or shared-mount concept, so
    there's no acq_idx to return.

    Args:
        x_gt: (n_images, 1, H, W) -- expected to already be disk-masked (load_mnist_subset_mra)
        corruptions_per_object: number of independent F_MRA observations per GT object
        noise_std: float

    Returns:
        y_obs:      (n_images * corruptions_per_object, 1, H, W) — the observation set (mu)
        theta_star: (n_images * corruptions_per_object,)         — the true rotation used per
            observation. Diagnostic ONLY — never fed to the model, since the algorithm must
            recover the prior without knowing the true pose.
        image_idx:  (n_images * corruptions_per_object,)         — which source object each
            observation came from, for grouped visualization.
    """
    n_images = x_gt.size(0)
    x_expanded = x_gt.repeat_interleave(corruptions_per_object, dim=0)  # (n_obs, 1, H, W)
    y_obs, theta_star = forward_channel_mra(x_expanded, noise_std=noise_std)
    image_idx = torch.arange(n_images).repeat_interleave(corruptions_per_object)
    return y_obs, theta_star, image_idx


def load_mnist_volumes_3d(n_images_per_class: int, vol_size: int = VOL_SIZE,
                          inplane_size: int | None = None,
                          depth_extent: int | None = None,
                          digit_classes: list[int] | None = None,
                          seed: int | None = None,
                          train: bool = True) -> torch.Tensor:
    """
    Build R^{p^3} MNIST volumes by extruding each 2D digit uniformly across a central depth
    band, leaving empty (-1) space at the boundary of ALL THREE axes so a generic SO(3) rotation
    doesn't clip the object against the cube's corners — a 90-degree rotation swaps depth extent
    into in-plane extent, so depth-only margin would not be enough on its own. Binding
    constraint: with in-plane ink radius r_xy and half-depth extent d, need
    sqrt(r_xy^2 + d^2) <= vol_size/2 - eps. `inplane_size`/`depth_extent` defaults below are a
    starting point for that constraint, not a guarantee — gated by the mass-invariance check in
    test_so3_math.py.

    Args:
        n_images_per_class: how many images to draw per class (not a total count).
        vol_size: p, the cube's side length.
        inplane_size: MNIST digit is loaded at THIS resolution (default: round(vol_size*0.65))
            before being zero-padded (to background -1, not literal zero) up to vol_size in H,W.
        depth_extent: thickness (in voxels) of the depth band the digit is extruded across,
            centered in D (default: round(vol_size*0.25) — a thin tablet, not a long bar;
            also closer to real cryoEM specimen geometry, thin relative to lateral extent).
        digit_classes: which MNIST classes (0-9) to draw from. None = all 10 classes.
        train: which MNIST split to draw from -- forwarded to load_mnist_subset unmodified,
            see its docstring for the train/test split convention.

    Returns:
        (N, 1, vol_size, vol_size, vol_size) in [-1, 1], background -1.
    """
    if inplane_size is None:
        inplane_size = round(vol_size * 0.65)
    if depth_extent is None:
        depth_extent = round(vol_size * 0.25)

    x2d = load_mnist_subset(n_images_per_class, image_size=inplane_size,
                            digit_classes=digit_classes, seed=seed,
                            train=train)  # (N, 1, s, s) in [-1, 1]
    N = x2d.size(0)

    pad_total = vol_size - inplane_size
    pad_lo = pad_total // 2
    pad_hi = pad_total - pad_lo
    x2d_padded = torch.nn.functional.pad(x2d, (pad_lo, pad_hi, pad_lo, pad_hi), value=-1.0)
    # (N, 1, vol_size, vol_size)

    margin = (vol_size - depth_extent) // 2
    vol = torch.full((N, 1, vol_size, vol_size, vol_size), -1.0, dtype=x2d.dtype)
    vol[:, :, margin:margin + depth_extent] = x2d_padded.unsqueeze(2).expand(
        -1, -1, depth_extent, -1, -1)
    return vol


def build_observations_3d(
    x_gt: torch.Tensor,
    corruptions_per_object: int,
    n_tilts: int,
    tilt_increment_deg: float,
    noise_std: float,
    tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    3D/CryoET analogue of build_observations: applies forward_channel_3d to each GT volume
    `corruptions_per_object` times, each repetition an independent tilt-series acquisition of
    `n_tilts` 2D projections.

    Args:
        x_gt: (n_images, 1, D, H, W)
        corruptions_per_object: number of independent tilt-series acquisitions per GT object
        n_tilts: T, number of tilts (projections) within one acquisition's series
        tilt_increment_deg: angular step between consecutive tilts, in degrees
        noise_std: float
        tilt_axis: the FIXED physical tilt axis, in lab-frame coordinates

    Returns:
        y_obs:   (n_images * corruptions_per_object * n_tilts, 1, H, W) — the observation set
        R_star:  (n_images * corruptions_per_object * n_tilts, 3, 3)   — the true rotation
            matrix used per observation. Diagnostic ONLY — never fed to the model. Kept as a
            matrix (not the 6D pose representation) since it's never fed to the interpolant and
            matrices are the natural input to si.rotation_geodesic_angle.
        image_idx: (n_images * corruptions_per_object * n_tilts,) — which source object each
            observation came from.
        acq_idx:   (n_images * corruptions_per_object * n_tilts,) — which acquisition each
            observation belongs to; contiguous and in tilt order, so y_obs[acq_idx == a] is one
            full tilt series.
    """
    n_images = x_gt.size(0)
    n_acq = n_images * corruptions_per_object
    tilt_increment = tilt_increment_deg * torch.pi / 180.0

    x_per_acq = x_gt.repeat_interleave(corruptions_per_object, dim=0)  # (n_acq, 1, D, H, W)
    R = sample_tilt_series_rotations_so3(n_acq, n_tilts, tilt_increment,
                                         tilt_axis=tilt_axis, device=x_gt.device)  # (n_acq,T,3,3)

    x_expanded = x_per_acq.repeat_interleave(n_tilts, dim=0)  # (n_acq*T, 1, D, H, W)
    R_flat = R.reshape(-1, 3, 3)                              # (n_acq*T, 3, 3)
    pose6_flat = matrix_to_gram_schmidt(R_flat)
    y_obs, _ = forward_channel_3d(x_expanded, noise_std=noise_std, pose6=pose6_flat)
    image_idx = torch.arange(n_images).repeat_interleave(corruptions_per_object * n_tilts)
    acq_idx = torch.arange(n_acq).repeat_interleave(n_tilts)
    return y_obs, R_flat, image_idx, acq_idx
