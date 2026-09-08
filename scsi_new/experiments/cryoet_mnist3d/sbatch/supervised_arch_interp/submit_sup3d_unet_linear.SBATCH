#!/bin/bash
#SBATCH --job-name=sup3d_unet_linear
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

# Supervised upper-bound baseline for experiments/cryoet_mnist3d -- cell 4/4 of the
# arch x interpolant 2x2 (DiT vs UNet backbone, GVP vs Linear interpolant).
# This cell: --arch unet  --interpolant_style linear.
# Fixed across all 4 cells: digit 3 only, 23,000 images/class, 300,000 optimiser steps,
# batch 16, visualise + checkpoint every 20,000 steps, viz ODE-step sweep
# {16,32,64,128,256,512,1024}. Full grid + caveats: ./README.md
#
# --arch unet -> ConditionalVelocityCryoET3D (model.py): the diffusers UNet3DConditionModel
# video-UNet, main.py's default backbone. ~37M params. It reshapes (B,C,D,H,W)->(B*D,C,H,W)
# for its spatial convs, so GPU memory scales with batch*D: --batch_size 16 was profiled at
# ~77% of a 48GB L40 in ../warmup_mstep_grid/. --gres=gpu:1 is generic -- a job landing on a
# <40GB card (e.g. V100 32GB) WILL OOM here; add a typed --gres (sinfo -o "%n %G") or drop
# --batch_size to 8 (change it in the unet_gvp cell too, to keep the pair comparable).
#
# --time=48:00:00 is a FLAT BUDGET, not a measurement. 300k steps + the viz sweep is ~2x the
# biggest cell of ../warmup_mstep_grid/, which used a flat 24h. main_supervised.py checkpoints
# (model+EMA+optim+sched) at every 20k step, but has NO --resume_from: a walltime kill leaves
# an inspectable step_<n>.pt and no way to continue without new code. Read train/ step-rate
# off wandb after the first checkpoint and resubmit with a real --time (or a smaller
# --n_steps_train) if it will not finish.

cd /scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d

uv run python main_supervised.py \
    --arch unet \
    --interpolant_style linear \
    --digit_classes 3 \
    --n_images_per_class 23000 \
    --n_steps_train 300000 \
    --batch_size 16 \
    --log_every 20000 \
    --viz_ema \
    --viz_n_steps_sampling 16 32 64 128 256 512 1024 \
    --checkpoint_steps 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000 \
    --checkpoint_dir checkpoints/cryoet_mnist3d_supervised/unet_linear \
    --wandb_project scsi-cryoet-mnist3d-supervised-arch-interp \
    --wandb_run_name unet_linear
