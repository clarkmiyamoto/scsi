# cryoet_igg1d — launch recipes

Unsupervised (SCSI) counterpart of the supervised image-conditioned CryoFM model on CryoBench
IgG-1D (`~/cryofm/scripts/train_igg1d.py`, `configs/igg1d/igg1d_cond_64.py`). The data is 100k
simulated particle images of an IgG antibody undergoing a 1D circular motion (100 conformations,
3.6° apart), with uniform SO(3) poses, ±30 Å shifts, experimental CTFs and SNR 0.01. The loop,
checkpointing and LR schedule are `../cryoet_mnist3d/main.py`'s. The network is cryofm's, vendored.

## What the EM loop sees

- **Observed:** the particle images and their CTF parameters (`data.build_observations`). In real
  cryo-EM the per-image CTF is estimated; the pose is not. The images are:
  - Fourier-cropped to `--resolution`;
  - phase-flipped with their own CTF, exactly as the supervised run's `phase_flip=True`;
  - divided by the noise std estimated from the image corners.
- **Split:** the same as cryofm (`--val_fraction 0.02 --split_seed 0`). The observations are
  cryofm's ~98k training images.
- **GT (metrics and panels only):** `data.build_eval_pool`, the held-out images with their
  image-frame targets, conformation labels and poses. With `--eval_n 16` these are exactly the
  images cryofm's sampling eval uses, and `eval/{raw,ema}/*` shares its `sample/*` metric names.

## Forward model (`corruption.py`)

    F(x) = vol_gain · IFFT2(|CTF_c| · FFT2(P(T_s R x))) + N(0, 1)

- `R` is a Haar-random pose and `s` a uniform shift of ±`--shift_extent_A`.
- `c` is a random row of the observations' CTF parameters, and `P` is the sum over z.
- Posing uses CryoBench's `rotate_volume` convention. `python -m scsi_new.experiments.cryoet_igg1d.corruption` checks it, and `ctf_2d`
  against `compute_ctf`; both match exactly.
- `--pair_frame image` (default): the M-step target is the posed volume `T_s R x̂`, i.e. the frame
  of the supervised model and of its z-broadcast image conditioning. `canonical` / `lift` are
  cryoet_mnist3d's alternatives.
- `x̂` carries its source image's shift, so by default it is recentred (centre of mass) before
  posing (`--no_recenter` turns this off).

`vol_gain` sets the units of the volumes the model generates. It must stay fixed for a run: it is
saved in every checkpoint and checked on load. By default it is calibrated so the pseudoinverse
warm-start volumes have unit std, like cryofm's `vol_std` standardisation but without GT. At
64 px that gives **0.0131**. `python -m scsi_new.experiments.cryoet_igg1d.data` prints it next to the GT-derived value **0.0597**,
the gain that makes GT volumes unit-std. The same check gives a matched correlation of 0.349
against a noiseless ceiling of 0.351, with residual std 1.003, so the channel explains the real
images.

## Warm start (`pseudoinverse.py`)

Each observed image is pose-blind backprojected on its own:
- CTF-corrected with a Wiener filter whose spectral SNR is estimated from the observations
  (`estimate_ssnr`);
- smeared along z through a centred ball of `--pinv_diameter_A` (default 192 Å);
- trained on as `ResampledPairs`, re-posed by the channel on every draw, as in cryoet_mnist3d.

`python -m scsi_new.experiments.cryoet_igg1d.pseudoinverse` scores it against GT. The corrected image has r 0.70 with the GT
projection (the raw image 0.28). The volume has r 0.42 with its image-frame target and 0.16 with
another image's.

## Starting point

As in mnist3d:
1. `--num_scsi_steps 0 --save_warmup_ckpt PATH --lr_horizon_scsi_steps K` (warmup only; keep K at
   the arms' EM step count, or the cosine anneals fully over the warmup).
2. `--load_warmup_ckpt PATH --ckpt_dir DIR --resume` for each EM arm, with the same
   `--warmup_n_steps_train`.

`sbatch/sched_warmup_mstep/` does this for the LR schedule × warmup × M-step ablation; its
`debug.sh` is the GPU check. `sbatch/first_run/` is the no-warm-start baseline
(`--warmup_n_steps_train 0 --vol_gain 0.0602`). A quick look at the warm start and two EM
steps:

```bash
cd <repo root>   # repo root; run as a module so package imports resolve
uv run python -m scsi_new.experiments.cryoet_igg1d.main \
    --n_observations 2000 --warmup_n_steps_train 200 --estep_num_samples 64 \
    --mstep_n_steps_train 50 --num_scsi_steps 2
```

That needs a GPU node.

## Memory and speed

- The network is cryofm's (180 M params, attention at the two coarsest levels). It runs in bf16
  autocast on CUDA (`--no_bf16` turns this off).
- cryofm measured 54 GB peak at batch 12 on an A100-80GB, hence `--warmup_batch_size 12` and
  `--mstep_batch_size 12`, and `--constraint "a100-80gb|h100|h200"`. The E-step (`--estep_batch_size 32`)
  is inference only.
- `--resolution 32` (12 Å/px) makes every step roughly 8x cheaper, but it is no longer comparable
  with the supervised 64 px run.
- `--estep_n_steps_sampling 32` roughly halves E-step and eval time.
- `--n_observations N` loads a random subset of the training images. All ~98k is 1.6 GB at 64 px
  and takes a few minutes to read from ceph.

## Metrics (`eval_metrics.py`)

All metrics are logged for the raw and EMA weights every `--viz_every` EM steps. `eval_calib/*`
holds the same scores for the GT targets themselves, i.e. the ceilings.

- `corr_target`: Pearson r with the image-frame target.
- `conf_err_deg`, `conf_acc`, `conf_acc_within_3`: the best-matching conformation, with all 100
  posed by the image's GT pose.
- `pred_conf_std`: std of the predicted conformation index. A low value means collapse.
- `fsc0.5_res_A`: FSC=0.5 resolution against the target, using cryofm's shells.
- Handedness: a single projection cannot distinguish `x` from its mirror through z = D/2, so EM can
  settle on either hand.
  - `corr_target_hand` takes the better hand per image.
  - `hand_flip_frac` is how often that is the mirror.
  - The conformation and FSC scores use the better hand.

## Checks

```bash
uv run python -m scsi_new.experiments.cryoet_igg1d.corruption      # posing / CTF conventions vs CryoBench (CPU, seconds)
uv run python -m scsi_new.experiments.cryoet_igg1d.data --save channel_check.png
    # the channel vs the real images, GT used here only: matched vs mismatched correlation,
    # the sign of the scale, residual std ~1, and the GT-derived and calibrated --vol_gain
uv run python -m scsi_new.experiments.cryoet_igg1d.pseudoinverse --save pinv.png
    # the warm start vs GT over a sweep of --pinv_diameter_A (a few minutes on one core)
```

On a login node set `OMP_NUM_THREADS=1`: the user slice is capped at one core, and torch's
default thread count makes it much slower.
