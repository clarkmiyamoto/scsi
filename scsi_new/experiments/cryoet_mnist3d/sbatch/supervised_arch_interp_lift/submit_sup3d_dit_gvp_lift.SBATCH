#!/bin/bash
#SBATCH --job-name=sup3d_dit_gvp_lift
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Supervised baseline for experiments/cryoet_mnist3d, LIFTED-TARGET variant -- cell 1/4 of the
# arch x interpolant 2x2 (DiT vs UNet backbone, GVP vs Linear interpolant).
# This cell: --arch dit  --interpolant_style gvp  --lift.
#
# --lift trains on (R.x, F(x)) instead of the canonical (x, F(x)): R is a FRESH independent
# Haar-uniform SO(3) rotation per volume (corruption.build_pair_sample), uncorrelated with the
# channel's own discarded mount. The tilt series y pins the clean volume only up to a global
# rotation, so this symmetrises the training target over SO(3) and the learned b_t(.|y) becomes
# rotation-invariant -- y fixes the shape, not its orientation. Everything else is identical to
# the canonical grid in ../supervised_arch_interp/; run both to see how lifting moves the 2x2.
# This grid, not the canonical one, is the true upper bound for the pose-blind SCSI loop in main.py.
#
# Fixed across all 4 cells: digit 3 only, 23,000 images/class, 300,000 optimiser steps,
# batch 16, visualise + checkpoint every 20,000 steps, viz ODE-step sweep
# {16,32,64,128,256,512,1024}. Full grid + caveats: ./README.md and ../supervised_arch_interp/README.md
#
# --arch dit -> ConditionalDiTCryoET3D (model.py): volumetric 3D-patch DiT, full 3D
# self-attention over (32/4)**3 = 512 tokens, adaLN-Zero timestep conditioning. ~33M params
# at the 384/12/6 default (close to the UNet's ~37M). Far lighter on GPU memory than the
# video-UNet cells -- batch 16 is comfortable on any modern card here.
#
# --time=48:00:00 is a FLAT BUDGET, not a measurement (no CUDA available to profile). 300k
# steps + the viz sweep is ~2x the biggest cell of ../warmup_mstep_grid/, which used a flat
# 24h. main_supervised.py checkpoints (model+EMA+optim+sched) at every 20k step, but has NO
# --resume_from: a walltime kill leaves an inspectable step_<n>.pt and no way to continue
# without new code. Read train/ step-rate off wandb after the first checkpoint and resubmit
# with a real --time (or a smaller --n_steps_train) if it will not finish.
#
# --lift is already the main_supervised.py default; it is passed explicitly so the submitted
# job is self-documenting and independent of that default.

cd /scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d

uv run python main_supervised.py \
    --arch dit \
    --interpolant_style gvp \
    --lift \
    --digit_classes 3 \
    --n_images_per_class 23000 \
    --n_steps_train 300000 \
    --batch_size 16 \
    --log_every 20000 \
    --viz_ema \
    --viz_n_steps_sampling 16 32 64 128 256 512 1024 \
    --checkpoint_steps 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000 \
    --checkpoint_dir checkpoints/cryoet_mnist3d_supervised/dit_gvp_lift \
    --wandb_project scsi-cryoet-mnist3d-supervised-arch-interp-lift \
    --wandb_run_name dit_gvp_lift
