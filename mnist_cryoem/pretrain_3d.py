import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import load_mnist_volumes_3d, build_observations_3d
from model import ConditionalVelocityCryoET3D, VOL_SIZE
from corruption import forward_channel_3d
from scsi import loss_func_joint_3d
from si import matrix_to_gram_schmidt
from wandb_logging import log_train_step, log_pretrain_reconstruction_3d

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc, flush=True)
        return iterable

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

########################################################
# 3D/CryoET analogue of pretrain.py — supervised warm-start for Theta^(0), the first SCSI
# teacher, on extruded-MNIST volumes under the CryoET channel (SO(3) rotation -> 2D projection
# -> AWGN). Structurally identical to pretrain.py: ordinary stochastic-interpolant training on
# scsi.py::loss_func_joint_3d, no outer/inner loop, R drawn fresh each step, x_hat = the
# canonical (untouched) GT volume.
#
# x_hat is deliberately left canonical (not rotated) — see mnist_cryoem/CLAUDE.md for why: the
# main EM loop's M-step computes y_hat = F(x_hat; pose6_hat), which RE-APPLIES the rotation, so
# x_hat has to be canonical for that reconstruction (and this checkpoint, loaded via
# main_3d.py's --init_ckpt) to be meaningful. This is the single highest-value invariant to get
# right — it's a silent bug (no shape error) if broken, regardless of pose representation.
########################################################


########################################################
# Pluggable pool selection — same pattern as pretrain.py.
########################################################

def select_per_class(n_images_per_class: int, digit_classes: list[int] | None,
                     vol_size: int, inplane_size: int | None, depth_extent: int | None,
                     seed: int | None = None) -> torch.Tensor:
    """n_images_per_class GT volumes from EACH class in digit_classes (default: all 10).
    Always drawn from the MNIST TRAIN split (train=True) -- must stay disjoint from
    main_3d.py's EM pool, which draws from the test split. See mnist_cryoem/CLAUDE.md."""
    return load_mnist_volumes_3d(n_images_per_class, vol_size=vol_size,
                                 inplane_size=inplane_size, depth_extent=depth_extent,
                                 digit_classes=digit_classes, seed=seed, train=True)


SELECTION_STRATEGIES = {
    "per_class": select_per_class,
}


########################################################
# Classical-recon warm start (--warmstart_target classical_recon). 3D analogue of
# pretrain.py::build_classical_recon_pool -- see its docstring for the full derivation; only the
# volume/pose-representation-specific details differ, noted inline below.
########################################################

def build_classical_recon_pool_3d(
    x_gt: torch.Tensor,
    corruptions_per_object: int,
    n_tilts: int,
    tilt_increment_deg: float,
    tilt_axis: tuple[float, float, float],
    recon_filter_type: str,
    recon_calibration: str,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    3D analogue of pretrain.py::build_classical_recon_pool (see its docstring for the full
    derivation -- steps 1-4 are identical here, one dimension up). Two representation-specific
    differences from the 2D version:
      - data.build_observations_3d returns R_star as (N,3,3) rotation MATRICES, not a scalar
        angle -- classical_recon_3d.backproject needs those matrices directly (R.transpose(-1,-2)
        for R^{-1}); si.matrix_to_gram_schmidt(R_star) is applied ONLY at the end, to produce the
        (N,6) pose_pool loss_func_joint_3d actually expects (mirrors forward_channel_3d's own
        pose6-at-the-public-boundary convention).
      - tilt_axis (the FIXED lab-frame tilt axis) has no 2D analogue.

    Returns:
        x_recon_pool: (n_images*corruptions_per_object*n_tilts, 1, D, H, W)
        pose_pool:    (n_images*corruptions_per_object*n_tilts, 6) -- TRUE pose per observation,
            Gram-Schmidt representation (exact -- known because this is synthetic data)
        y_pool:       (n_images*corruptions_per_object*n_tilts, 1, H, W) -- raw D_Y observations
        image_idx:    (n_images*corruptions_per_object*n_tilts,) -- which source object each
            entry came from; only used by the wandb panel's per-object display slice
    """
    from classical_recon_3d import backproject  # lazy: see build_classical_recon_pool's note

    vol_size = x_gt.size(-1)
    y_obs, R_star, image_idx, acq_idx = build_observations_3d(
        x_gt, corruptions_per_object=corruptions_per_object, n_tilts=n_tilts,
        tilt_increment_deg=tilt_increment_deg, noise_std=noise_std, tilt_axis=tilt_axis,
    )
    n_acq = x_gt.size(0) * corruptions_per_object

    x_recon_chunks = []
    for a in range(n_acq):
        mask = acq_idx == a
        x_hat_a = backproject(y_obs[mask], R_star[mask], vol_size,
                              filtered=True, filter_type=recon_filter_type)  # (1,1,D,H,W)
        x_recon_chunks.append(x_hat_a.expand(int(mask.sum()), -1, -1, -1, -1))
    x_recon_pool = torch.cat(x_recon_chunks, dim=0)

    if recon_calibration == "affine_clamp":
        a_coef = x_gt.std() / x_recon_pool.std().clamp_min(1e-8)
        b_coef = x_gt.mean() - a_coef * x_recon_pool.mean()
        x_recon_pool = (a_coef * x_recon_pool + b_coef).clamp(-1.0, 1.0)
    elif recon_calibration != "none":
        raise ValueError(f"Unknown recon_calibration: {recon_calibration!r}")

    pose_pool = matrix_to_gram_schmidt(R_star)
    return x_recon_pool, pose_pool, y_obs, image_idx


def _classical_target_for_display_3d(
    x_recon_pool: torch.Tensor | None,
    image_idx: torch.Tensor | None,
    n_images: int,
) -> torch.Tensor | None:
    """3D analogue of pretrain.py::_classical_target_for_display."""
    if x_recon_pool is None:
        return None
    return torch.stack([x_recon_pool[image_idx == i][0] for i in range(n_images)])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised pretraining of the first SCSI teacher Theta^(0) for the "
                    "3D/CryoET generalization: ordinary stochastic-interpolant training on a "
                    "small pool of known GT extruded-MNIST volumes, learning to flow noise to "
                    "the canonical volume conditioned on F(volume; R) for a fresh random "
                    "rotation R each step."
    )
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) the pretraining pool is drawn from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_pretrain_images_per_class", type=int, default=2,
                        help="How many GT volumes make up the supervised pool, PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--selection_strategy", type=str, default="per_class",
                        choices=list(SELECTION_STRATEGIES),
                        help="How the pretraining pool {x_i} is chosen — see "
                             "SELECTION_STRATEGIES")
    parser.add_argument("--vol_size", type=int, default=VOL_SIZE,
                        help="p: side length of the R^{p^3} volume cube")
    parser.add_argument("--inplane_size", type=int, default=None,
                        help="MNIST digit is loaded at THIS resolution before zero-padding up "
                             "to --vol_size in H,W (default: round(vol_size*0.65))")
    parser.add_argument("--depth_extent", type=int, default=None,
                        help="Thickness (voxels) of the depth band the digit is extruded "
                             "across, centered in D (default: round(vol_size*0.55))")
    parser.add_argument("--noise_std", type=float, default=0.5,
                        help="AWGN std for the 2D-image observation (different scale than the "
                             "2D pipeline's 1D-projection channel — don't reuse that default)")

    # Classical-recon warm start (see build_classical_recon_pool_3d's docstring)
    parser.add_argument("--warmstart_target", type=str, default="gt",
                        choices=["gt", "classical_recon"],
                        help="'gt' (default): x1 = the literal canonical GT volume, matching "
                             "today's behavior exactly. 'classical_recon': x1 = a classical "
                             "weighted-backprojection reconstruction of a FINITE pool of the "
                             "object's own corrupted observations -- see "
                             "build_classical_recon_pool_3d.")
    parser.add_argument("--corruptions_per_object", type=int, default=1,
                        help="Only used when --warmstart_target classical_recon. Number of "
                             "independent tilt-series ACQUISITIONS per GT object -- each "
                             "acquisition gets its OWN classical reconstruction. Same flag name "
                             "as main_3d.py, but a much smaller default (main_3d.py: 50) -- each "
                             "acquisition costs one backproject() call at pool-build time.")
    parser.add_argument("--n_tilts", type=int, default=16,
                        help="Only used when --warmstart_target classical_recon. T: number of "
                             "tilts (2D projections) per acquisition's tilt series, at angles "
                             "evenly spaced by --tilt_increment_deg -- also the number of views "
                             "classical backprojection reconstructs each x_hat from. Same "
                             "name/default as main_3d.py's --n_tilts.")
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5,
                        help="Only used when --warmstart_target classical_recon. Same "
                             "name/default as main_3d.py's --tilt_increment_deg.")
    parser.add_argument("--tilt_axis", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                        help="Only used when --warmstart_target classical_recon. Same "
                             "name/default as main_3d.py's --tilt_axis.")
    parser.add_argument("--recon_filter_type", type=str, default="hann", choices=["hann", "ramp"],
                        help="Only used when --warmstart_target classical_recon. Passed to "
                             "classical_recon_3d.backproject's ramp filter.")
    parser.add_argument("--recon_calibration", type=str, default="affine_clamp",
                        choices=["affine_clamp", "none"],
                        help="Only used when --warmstart_target classical_recon. "
                             "'affine_clamp' (default): rescale the whole reconstruction pool "
                             "to the GT pool's mean/std (raw backprojection output is not on "
                             "[-1,1] scale -- see build_classical_recon_pool_3d's docstring), "
                             "then clamp to [-1,1]. 'none': use raw backproject() output as-is "
                             "-- an escape hatch for debugging, expected to train poorly.")

    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Flat supervised SGD steps (no outer/inner loop)")
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"])
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save a safety checkpoint every N steps (final state is always "
                             "saved too)")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Log a qualitative reconstruction panel to wandb every N steps "
                             "(0 disables it; the final step always logs one)")
    parser.add_argument("--sample_steps", type=int, default=20,
                        help="Euler steps for the joint ODE used only by --plot_every's "
                             "reconstruction panel (not used in training itself)")
    parser.add_argument("--out_ckpt", type=str,
                        default="mnist_cryoet_checkpoints/pretrain_theta0.pt")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Tiny run: quick smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        def _explicit(flag):
            return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

        if not _explicit("--digit_classes"):              args.digit_classes = [0]
        if not _explicit("--n_pretrain_images_per_class"): args.n_pretrain_images_per_class = 2
        if not _explicit("--vol_size"):                    args.vol_size = 16
        if not _explicit("--n_steps"):                    args.n_steps = 8
        if not _explicit("--batch_size"):                 args.batch_size = 4
        if not _explicit("--checkpoint_every"):            args.checkpoint_every = 4
        if not _explicit("--plot_every"):                  args.plot_every = 4
        if not _explicit("--sample_steps"):                args.sample_steps = 5
        if not _explicit("--n_tilts"):                     args.n_tilts = 4

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    print(f"Device: {device}")

    # ── Pretraining pool ─────────────────────────────────────────────────
    select_fn = SELECTION_STRATEGIES[args.selection_strategy]
    volume_pool = select_fn(args.n_pretrain_images_per_class, args.digit_classes,
                            args.vol_size, args.inplane_size, args.depth_extent).to(device)
    print(f"Pretrain pool: {volume_pool.size(0)} volumes "
          f"({args.n_pretrain_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))}, "
          f"strategy={args.selection_strategy}, vol_size={args.vol_size}, split=train)")

    # ── Classical-recon warm start (--warmstart_target classical_recon) ───
    x_recon_pool = pose_pool = y_pool = recon_image_idx = None
    if args.warmstart_target == "classical_recon":
        x_recon_pool, pose_pool, y_pool, recon_image_idx = build_classical_recon_pool_3d(
            volume_pool, corruptions_per_object=args.corruptions_per_object,
            n_tilts=args.n_tilts, tilt_increment_deg=args.tilt_increment_deg,
            tilt_axis=tuple(args.tilt_axis), recon_filter_type=args.recon_filter_type,
            recon_calibration=args.recon_calibration, noise_std=args.noise_std,
        )
        x_recon_pool, pose_pool, y_pool, recon_image_idx = (
            x_recon_pool.to(device), pose_pool.to(device), y_pool.to(device),
            recon_image_idx.to(device))
        print(f"Classical-recon pool: {x_recon_pool.size(0)} (x_hat, y) pairs "
              f"({volume_pool.size(0)} objects x {args.corruptions_per_object} acquisitions x "
              f"{args.n_tilts} tilts, filter={args.recon_filter_type}, "
              f"calibration={args.recon_calibration})  "
              f"x_hat stats: min={x_recon_pool.min():.3f} max={x_recon_pool.max():.3f} "
              f"mean={x_recon_pool.mean():.4f} std={x_recon_pool.std():.4f}")

    # ── Model / optimizer ────────────────────────────────────────────────
    model = ConditionalVelocityCryoET3D(vol_size=args.vol_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    if use_wandb:
        pool_size = (x_recon_pool.size(0) if args.warmstart_target == "classical_recon"
                    else volume_pool.size(0))
        wandb.init(
            project="scsi-cryoet-mnist3d-pretrain",
            config=vars(args) | {"n_params": n_params, "pool_size": pool_size,
                                 "dataset_split": "train"},
        )

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── Flat supervised stochastic-interpolant training ─────────────────
    N = x_recon_pool.size(0) if args.warmstart_target == "classical_recon" else volume_pool.size(0)
    model.train()
    running_img, running_pose = 0.0, 0.0

    for step in tqdm(range(args.n_steps), desc="pretrain"):
        idx = torch.randint(0, N, (args.batch_size,))
        if args.warmstart_target == "classical_recon":
            x_batch = x_recon_pool[idx]                   # classical WBP reconstruction, calibrated
            pose_batch = pose_pool[idx]                    # TRUE pose for that observation
            y_batch = y_pool[idx]                          # the REAL observation -- not resynthesized
        else:
            x_batch = volume_pool[idx]                     # canonical GT, untouched
            y_batch, pose_batch = forward_channel_3d(x_batch, noise_std=args.noise_std)

        loss, loss_img, loss_pose = loss_func_joint_3d(
            model, x_batch, pose_batch, y_batch,
            style=args.interpolant_style, pose_loss_weight=args.pose_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_img += loss_img.item()
        running_pose += loss_pose.item()
        log_train_step(loss_img, loss_pose, grad_norm, step, use_wandb)

        if (step + 1) % args.checkpoint_every == 0:
            torch.save(model.state_dict(), out_ckpt)

        if args.plot_every > 0 and (step + 1) % args.plot_every == 0:
            log_pretrain_reconstruction_3d(
                model, volume_pool, noise_std=args.noise_std,
                sample_steps=args.sample_steps, step=step, use_wandb=use_wandb,
                classical_target=_classical_target_for_display_3d(
                    x_recon_pool, recon_image_idx, volume_pool.size(0)),
            )

    if args.plot_every > 0 and args.n_steps % args.plot_every != 0:
        log_pretrain_reconstruction_3d(
            model, volume_pool, noise_std=args.noise_std,
            sample_steps=args.sample_steps, step=args.n_steps - 1, use_wandb=use_wandb,
            classical_target=_classical_target_for_display_3d(
                x_recon_pool, recon_image_idx, volume_pool.size(0)),
        )

    torch.save(model.state_dict(), out_ckpt)
    print(f"steps={args.n_steps}  loss_image={running_img / args.n_steps:.5f}"
          f"  loss_pose={running_pose / args.n_steps:.5f}")
    print(f"Theta^(0) saved -> {out_ckpt}")

    if use_wandb:
        wandb.finish()
    print("Done.")
