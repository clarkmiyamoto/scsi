import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import load_mnist_subset, build_observations
from model import ConditionalVelocityCryoEM, IMAGE_SIZE
from corruption import forward_channel, sample_uniform_angle
from scsi import loss_func_joint
from wandb_logging import log_train_step, log_pretrain_reconstruction
from warmup import backproject_pool, classical_target_for_display

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
# Supervised warm-start for Theta^(0), the first SCSI teacher — there is no teacher yet, so
# instead of an E-step ODE flow, (x_hat, R_hat) is built directly from known GT: R_hat drawn
# fresh each step, x_hat = the canonical (untouched) GT digit. Ordinary stochastic-interpolant
# training on scsi.py::loss_func_joint — same M-step loss the real EM loop uses, just without
# scsi.py::train_mstep's fixed-pool/permutation machinery (that exists to amortize an
# expensive ODE solve, which this script doesn't have).
#
# x_hat is deliberately left canonical (not rotated) — see mnist_cryoem/CLAUDE.md for why:
# the main EM loop's M-step computes y_hat = F(x_hat; R_hat), which RE-APPLIES the rotation,
# so x_hat has to be canonical for that reconstruction (and this checkpoint, loaded via
# main.py's --init_ckpt) to be meaningful.
########################################################


########################################################
# Pluggable pool selection — add a new strategy by writing a function with this signature
# and registering it in SELECTION_STRATEGIES; nothing else in this file needs to change.
########################################################

def select_per_class(n_images_per_class: int, digit_classes: list[int] | None,
                     seed: int | None = None) -> torch.Tensor:
    """n_images_per_class GT digits from EACH class in digit_classes (default: all 10).
    Always drawn from the MNIST TRAIN split (train=True) -- must stay disjoint from
    main.py's EM pool, which draws from the test split. See mnist_cryoem/CLAUDE.md."""
    return load_mnist_subset(n_images_per_class, digit_classes=digit_classes, seed=seed,
                             train=True)


SELECTION_STRATEGIES = {
    "per_class": select_per_class,
}


########################################################
# Classical-recon warm start (--warmstart_target classical_recon): an alternative to the
# GT-supervised target above. Instead of x1 = the literal canonical GT digit, x1 = a classical
# (filtered-backprojection) reconstruction of a FINITE set of the object's own corrupted
# observations -- built ONCE up front (backprojection is deterministic given its observations,
# so redoing it every SGD step would be pure waste), then trained on via ordinary flat SGD over
# that fixed pool, exactly like the GT-supervised path.
#
# The observation geometry (--corruptions_per_object/--n_tilts/--tilt_increment_deg) reuses the
# SAME flag names main.py's real EM pool uses -- data.build_observations is the SAME call --
# so this pool can be sized to literally match the geometry the real EM loop will actually see.
# Their DEFAULTS differ from main.py's, though (corruptions_per_object=1 here vs. 200 there):
# each acquisition costs one backproject() call at pool-build time, so this default stays cheap;
# see mnist_cryoem/CLAUDE.md for the same precedent already set by --noise_std's differing
# defaults between pretrain.py and main.py.
########################################################

def build_classical_recon_pool(
    x_gt: torch.Tensor,
    corruptions_per_object: int,
    n_tilts: int,
    tilt_increment_deg: float,
    recon_filter_type: str,
    recon_calibration: str,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds the classical-recon supervised pretraining pool D = {(x_hat, y, theta_star)} (see
    mnist_cryoem/CLAUDE.md's "Classical-recon warm start" section for the full derivation):

      1. data.build_observations draws corruptions_per_object independent tilt-series
         ACQUISITIONS per object in x_gt, each T=n_tilts observations -- this is D_Y.
      2. classical_recon.backproject inverts EACH acquisition's own n_tilts observations (at
         their TRUE angles -- a legitimate oracle here since this is synthetic data we generated
         ourselves, the same "known-pose" assumption classical_recon.py's own diagnostics
         already rely on) into ONE reconstruction x_hat per acquisition -- this is D_X. Because
         theta_star is the ABSOLUTE angle (not a tilt-series-relative offset), the reconstruction
         lands in the CANONICAL frame directly -- unlike classical_recon.py's OWN --mode
         tilt_series gallery, there is no arbitrary-absolute-rotation caveat here.
      3. Every observation within an acquisition is paired with that SAME x_hat as its
         interpolant target, broadcasting the per-acquisition reconstruction across all of its
         own observations. y is the REAL observation, never resynthesized via
         forward_channel(x_hat, ...) -- deliberately unlike train_mstep's fresh-F(x_pool)
         pattern: resynthesizing would compound the reconstruction's own approximation error
         into y too, whereas keeping the real y anchors the conditioning signal in the (noisy
         but faithful) original physics.
      4. Calibration (recon_calibration="affine_clamp", default): raw backprojection output is
         NOT on MNIST's [-1,1] scale (see classical_recon.compare's own docstring) -- a single
         global affine map, fit ONCE from the whole reconstruction pool's AGGREGATE mean/std to
         the whole GT pool's AGGREGATE mean/std (NOT a per-acquisition fit against that
         acquisition's own source image's individual mean/std -- a per-acquisition fit would
         hide the reconstruction's actual error instead of correcting only its aggregate scale),
         is applied, then the result is clamped to [-1, 1]. Note this still reads x_gt's
         aggregate statistics, not zero GT information -- a real non-synthetic pipeline would
         need to replace that one aggregate (e.g. a known prior on the target distribution's
         scale) rather than assume it's GT-free already.

    Returns:
        x_recon_pool: (n_images*corruptions_per_object*n_tilts, 1, H, W) -- x_hat broadcast per
            observation within its acquisition
        theta_pool:   (n_images*corruptions_per_object*n_tilts,) -- TRUE angle per observation
            (exact -- known because this is synthetic data we generated)
        y_pool:       (n_images*corruptions_per_object*n_tilts, 1, W) -- raw D_Y observations,
            unmodified
        image_idx:    (n_images*corruptions_per_object*n_tilts,) -- which source object each
            entry came from; not used in training, only by the wandb panel to pick one
            representative reconstruction per displayed object
    """
    # y_obs/theta_star/image_idx/acq_idx -- this is D_Y. The per-acquisition backproject +
    # calibration loop (D_Y -> D_X) is shared with main.py's --warmup_steps phase, which runs the
    # SAME logic against its own EM-loop observation pool instead of building a fresh one here --
    # see warmup.py::backproject_pool's docstring.
    y_obs, theta_star, image_idx, acq_idx = build_observations(
        x_gt, corruptions_per_object=corruptions_per_object, n_tilts=n_tilts,
        tilt_increment_deg=tilt_increment_deg, noise_std=noise_std,
    )
    x_recon_pool = backproject_pool(
        y_obs, theta_star, acq_idx, x_gt,
        recon_filter_type=recon_filter_type, recon_calibration=recon_calibration,
    )

    return x_recon_pool, theta_star, y_obs, image_idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised pretraining of the first SCSI teacher Theta^(0): ordinary "
                    "stochastic-interpolant training on a small pool of known GT MNIST "
                    "digits, learning to flow noise to the canonical digit conditioned on "
                    "F(digit; R) for a fresh random rotation R each step."
    )
    parser.add_argument("--digit_classes", type=int, nargs="+", default=None,
                        help="Which MNIST classes (0-9) the pretraining pool is drawn from "
                             "(default: all 10 classes)")
    parser.add_argument("--n_pretrain_images_per_class", type=int, default=2,
                        help="How many GT images make up the supervised pool, PER CLASS "
                             "in --digit_classes")
    parser.add_argument("--selection_strategy", type=str, default="per_class",
                        choices=list(SELECTION_STRATEGIES),
                        help="How the pretraining pool {x_i} is chosen — see "
                             "SELECTION_STRATEGIES")
    parser.add_argument("--noise_std", type=float, default=1.0)

    # Classical-recon warm start (see build_classical_recon_pool's docstring)
    parser.add_argument("--warmstart_target", type=str, default="gt",
                        choices=["gt", "classical_recon"],
                        help="'gt' (default): x1 = the literal canonical GT digit, matching "
                             "today's behavior exactly. 'classical_recon': x1 = a classical "
                             "filtered-backprojection reconstruction of a FINITE pool of the "
                             "object's own corrupted observations -- see "
                             "build_classical_recon_pool.")
    parser.add_argument("--corruptions_per_object", type=int, default=1,
                        help="Only used when --warmstart_target classical_recon. Number of "
                             "independent tilt-series ACQUISITIONS per GT object -- each "
                             "acquisition gets its OWN classical reconstruction (added target "
                             "diversity for corruptions_per_object > 1). Same flag name as "
                             "main.py, but a much smaller default (main.py: 200) -- each "
                             "acquisition costs one backproject() call at pool-build time.")
    parser.add_argument("--n_tilts", type=int, default=16,
                        help="Only used when --warmstart_target classical_recon. T: number of "
                             "tilts (1D projections) per acquisition's tilt series, at angles "
                             "evenly spaced by --tilt_increment_deg from that acquisition's "
                             "random start offset -- also the number of views classical "
                             "backprojection reconstructs each x_hat from. Same name/default as "
                             "main.py's --n_tilts, so this can be set to match the real EM "
                             "pool's geometry exactly.")
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5,
                        help="Only used when --warmstart_target classical_recon. Degrees "
                             "between consecutive tilts within one acquisition's series. Same "
                             "name/default as main.py's --tilt_increment_deg.")
    parser.add_argument("--recon_filter_type", type=str, default="hann", choices=["hann", "ramp"],
                        help="Only used when --warmstart_target classical_recon. Passed to "
                             "classical_recon.backproject's ramp filter.")
    parser.add_argument("--recon_calibration", type=str, default="affine_clamp",
                        choices=["affine_clamp", "none"],
                        help="Only used when --warmstart_target classical_recon. "
                             "'affine_clamp' (default): rescale the whole reconstruction pool "
                             "to the GT pool's mean/std (raw backprojection output is not on "
                             "MNIST's [-1,1] scale -- see build_classical_recon_pool's "
                             "docstring), then clamp to [-1,1]. 'none': use raw backproject() "
                             "output as-is -- an escape hatch for debugging, expected to train "
                             "poorly.")

    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Flat supervised SGD steps (no outer/inner loop)")
    parser.add_argument("--interpolant_style", type=str, default="linear",
                        choices=["linear", "gvp"])
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save a safety checkpoint every N steps (final state is always "
                             "saved too)")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Log a qualitative reconstruction panel to wandb every N steps "
                             "(0 disables it; the final step always logs one)")
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="Euler steps for the joint ODE used only by --plot_every's "
                             "reconstruction panel (not used in training itself)")
    parser.add_argument("--out_ckpt", type=str,
                        default="mnist_cryoem_checkpoints/pretrain_theta0.pt")
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
        if not _explicit("--n_steps"):                    args.n_steps = 8
        if not _explicit("--batch_size"):                 args.batch_size = 8
        if not _explicit("--checkpoint_every"):            args.checkpoint_every = 4
        if not _explicit("--plot_every"):                  args.plot_every = 4
        if not _explicit("--sample_steps"):                args.sample_steps = 5
        if not _explicit("--n_tilts"):                     args.n_tilts = 4

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    print(f"Device: {device}")

    # ── Pretraining pool ─────────────────────────────────────────────────
    select_fn = SELECTION_STRATEGIES[args.selection_strategy]
    image_pool = select_fn(args.n_pretrain_images_per_class, args.digit_classes).to(device)
    print(f"Pretrain pool: {image_pool.size(0)} images "
          f"({args.n_pretrain_images_per_class} per class, "
          f"classes={args.digit_classes or list(range(10))}, "
          f"strategy={args.selection_strategy}, split=train)")

    # ── Classical-recon warm start (--warmstart_target classical_recon) ───
    x_recon_pool = theta_pool = y_pool = recon_image_idx = None
    if args.warmstart_target == "classical_recon":
        x_recon_pool, theta_pool, y_pool, recon_image_idx = build_classical_recon_pool(
            image_pool, corruptions_per_object=args.corruptions_per_object,
            n_tilts=args.n_tilts, tilt_increment_deg=args.tilt_increment_deg,
            recon_filter_type=args.recon_filter_type, recon_calibration=args.recon_calibration,
            noise_std=args.noise_std,
        )
        x_recon_pool, theta_pool, y_pool, recon_image_idx = (
            x_recon_pool.to(device), theta_pool.to(device), y_pool.to(device),
            recon_image_idx.to(device))
        print(f"Classical-recon pool: {x_recon_pool.size(0)} (x_hat, y) pairs "
              f"({image_pool.size(0)} objects x {args.corruptions_per_object} acquisitions x "
              f"{args.n_tilts} tilts, filter={args.recon_filter_type}, "
              f"calibration={args.recon_calibration})  "
              f"x_hat stats: min={x_recon_pool.min():.3f} max={x_recon_pool.max():.3f} "
              f"mean={x_recon_pool.mean():.4f} std={x_recon_pool.std():.4f}")

    # ── Model / optimizer ────────────────────────────────────────────────
    model = ConditionalVelocityCryoEM(image_size=IMAGE_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    if use_wandb:
        pool_size = (x_recon_pool.size(0) if args.warmstart_target == "classical_recon"
                    else image_pool.size(0))
        wandb.init(
            project="scsi-cryoem-mnist-pretrain",
            config=vars(args) | {"n_params": n_params, "pool_size": pool_size,
                                 "dataset_split": "train"},
        )

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── Flat supervised stochastic-interpolant training ─────────────────
    N = x_recon_pool.size(0) if args.warmstart_target == "classical_recon" else image_pool.size(0)
    model.train()
    running_img, running_pose = 0.0, 0.0

    for step in tqdm(range(args.n_steps), desc="pretrain"):
        idx = torch.randint(0, N, (args.batch_size,))
        if args.warmstart_target == "classical_recon":
            x_batch = x_recon_pool[idx]                  # classical FBP reconstruction, calibrated
            theta_batch = theta_pool[idx]                # TRUE angle for that observation
            y_batch = y_pool[idx]                         # the REAL observation -- not resynthesized
        else:
            x_batch = image_pool[idx]                     # canonical GT, untouched
            theta_batch = sample_uniform_angle(args.batch_size, device)
            y_batch, _ = forward_channel(x_batch, noise_std=args.noise_std, theta=theta_batch)

        loss, loss_img, loss_pose = loss_func_joint(
            model, x_batch, theta_batch, y_batch,
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
            log_pretrain_reconstruction(
                model, image_pool, noise_std=args.noise_std,
                sample_steps=args.sample_steps, step=step, use_wandb=use_wandb,
                classical_target=classical_target_for_display(
                    x_recon_pool, recon_image_idx, image_pool.size(0)),
            )

    if args.plot_every > 0 and args.n_steps % args.plot_every != 0:
        log_pretrain_reconstruction(
            model, image_pool, noise_std=args.noise_std,
            sample_steps=args.sample_steps, step=args.n_steps - 1, use_wandb=use_wandb,
            classical_target=classical_target_for_display(
                x_recon_pool, recon_image_idx, image_pool.size(0)),
        )

    torch.save(model.state_dict(), out_ckpt)
    print(f"steps={args.n_steps}  loss_image={running_img / args.n_steps:.5f}"
          f"  loss_pose={running_pose / args.n_steps:.5f}")
    print(f"Theta^(0) saved -> {out_ckpt}")

    if use_wandb:
        wandb.finish()
    print("Done.")
