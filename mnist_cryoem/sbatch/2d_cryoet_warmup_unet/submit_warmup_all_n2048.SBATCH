#!/bin/bash
#SBATCH --job-name=warmup_all_n2048_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# ARCH SWEEP (this directory): mirrors sbatch/2d_cryoet_warmup/ file-for-file (same coverage x
# amount x steps_per_em x sample_steps matrix, same rationale -- see below, unchanged from that
# directory's own anchor file), with ONE additional fixed setting layered on every cell: --arch
# unet (model.ConditionalUNet2D in place of the default ConditionalDiT). No --patch_size flag is
# passed here -- model.py's ConditionalUNet2D takes no patch_size parameter at all, and main.py's
# own --patch_size help text says it's "Ignored under --arch unet", so passing it would be a
# silent no-op. This directory exists specifically to be compared against its "_dit_p2" sibling,
# sbatch/2d_cryoet_warmup_dit_p2/ (--arch dit --patch_size 2), which needed a separate directory
# for exactly the same reason (see that directory's anchor file). job-name/WANDB_NAME/--ckpt_dir
# all get a "_unet" suffix so runs never collide with the original matrix's checkpoints/wandb
# runs OR with the _dit_p2 sibling.
#

# In-process warmup matrix (see main.py's --warmup_steps / warmup.py): unlike sbatch/2d_cryoet/'s
# two-script matrix (pretrain.py -> checkpoint -> main.py --init_ckpt), every job here is ONE
# main.py call that runs the classical-recon warmup phase and the EM loop back to back, sharing
# one D_Y for both (see mnist_cryoem/CLAUDE.md's warmup.py row) -- there is no separate
# pretraining pool or checkpoint hand-off anywhere in this folder.
#
# THIS FILE (all_n2048) is the anchor/baseline cell for the whole matrix -- see it for the full
# rationale; every other file in this folder points back here instead of repeating it.
#
# Three axes, swept mostly one-at-a-time from this baseline:
#
#   1. Digit coverage x corrupted-data amount (the CORE 3x4 grid, 12 files):
#      coverage in {single (digit 3), few (digits 0,3,8), all (all 10, this file)}
#      amount   in {--n_images_per_class 64, 512, 2048 (this file), 4096}
#      "amount of corrupted data" = --n_images_per_class: the number of distinct GT objects
#      pulled into D_Y, which (since warmup and EM share one D_Y) sizes BOTH phases' pools at
#      once. --dataset_split train (see below) is what makes the 2048/4096 tiers possible at
#      all.
#
#   2. --steps_per_em sweep (2 extra files: steps400, steps800), held >= 200 per the
#      requirement -- 200 is this file's own value (the main.py default), so it's the sweep's
#      first data point for free; steps400/steps800 vary ONLY --steps_per_em from this cell.
#
#   3. --sample_steps (ODE integration steps) sweep (2 extra files: ode128, ode256), held >= 64
#      -- note main.py's own default is 50, which does NOT satisfy that floor, so EVERY file in
#      this folder (not just the dedicated sweep) explicitly passes --sample_steps 64 or higher;
#      this file's 64 is the sweep's baseline point, ode128/ode256 vary ONLY --sample_steps.
#
# --dataset_split train (main.py, new flag): the OLD matrix's main.py hardcoded the MNIST TEST
# split specifically to stay disjoint from pretrain.py's TRAIN-split pretraining pool (see
# mnist_cryoem/CLAUDE.md's dataset-split gotcha). That guarantee protects against a SEPARATE
# pretraining pool overlapping the EM dataset -- it doesn't apply here, since --warmup_steps
# builds its classical-recon pool from THIS run's own D_Y in-process, there is no second pool to
# overlap with. Needed anyway: the test split's per-class count (892-1135) can't fit
# --n_images_per_class up to 4096; the train split (5421-6742/class) can. CAUTION if this
# pattern is ever combined with --init_ckpt pointing at a pretrain.py checkpoint (train-split by
# construction) -- that WOULD reintroduce the overlap. None of the files in this folder use
# --init_ckpt.
#
# --corruptions_per_object 4 (NOT main.py's own EM-oriented default of 200): this is the one
# parameter every file in this folder overrides down from main.py's default, and it's not part
# of "the corruption channel" (that's --n_tilts/--tilt_increment_deg/--noise_std, held fixed at
# the old matrix's values everywhere below) -- it's a pool-size knob. The reason: warmup.py's
# backproject_pool runs ONE classical_recon.backproject() call per ACQUISITION (Python-level
# loop, not batched across acquisitions) before warmup step 1 even starts, and acquisition count
# = n_images_per_class x n_classes x corruptions_per_object. At main.py's default cpo=200 this
# was always fine for the OLD matrix (train_mstep never calls backproject at all -- x_hat comes
# from the E-step's ODE flow). It is NOT fine here: at cpo=200 the all/4096 cell alone would be
# 4096 x 10 x 200 = 8.19M backproject() calls. Measured empirically on this machine (mps,
# 2048-acquisition sample): ~0.84 ms/acquisition, no degradation at scale -- at cpo=4 the
# heaviest cell in this matrix (all/4096: 4096 x 10 x 4 = 163,840 acquisitions) costs ~2.3
# minutes of pool-build, not hours. x_recon_pool memory at that same cell is ~10.7 GB (measured:
# N_obs x 1x32x32 float32 = N_obs x 4096 bytes, confirmed against the 2048-acquisition sample's
# 134 MB), comfortably inside this file's --mem=32G with margin for everything else. cpo=4
# matches the old matrix's own pretrain.py default for the same flag.
#
# --n_tilts 16 / --tilt_increment_deg 7.5 / --noise_std 3.0: identical to every file in
# sbatch/2d_cryoet/ -- same tilt count, same total angular span (16 x 7.5 = 120 deg), same AWGN
# level. Spelled out explicitly here (even though they match main.py's own defaults) so this
# file is self-documenting and robust to any future default change.
#
# --warmup_steps 25000: the requested warmup length, fixed across the whole matrix (not swept).
#
# Compare against: single_n2048/few_n2048 (same amount, narrower coverage), all_n64/all_n512/
# all_n4096 (same coverage, other amounts), all_n2048_steps400/steps800 (steps_per_em sweep),
# all_n2048_ode128/ode256 (sample_steps sweep).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_all_n2048_unet

uv run python main.py \
    --dataset_split train \
    --n_images_per_class 2048 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --arch unet \
    --warmup_steps 25000 \
    --steps_per_em 200 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_all_n2048_unet
