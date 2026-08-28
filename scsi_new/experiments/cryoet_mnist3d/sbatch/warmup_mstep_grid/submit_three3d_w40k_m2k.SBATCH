#!/bin/bash
#SBATCH --job-name=three3d_w40k_m2k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D warmup x M-step "amount of training" grid, digit class 3 only (6,000 images/class).
# 3D->2D counterpart of ../../../cryoet_mnist/sbatch/warmup_mstep_grid/. That grid runs
# num_scsi_steps=40; here it is 12 -- a vol_size 32 EM step is ~30x the compute (see RECIPIES.md),
# so the M-step axis also tops out at 10k rather than the 2D grid's 20k.
# This cell: warmup_n_steps_train=40000, mstep_n_steps_train=2000, num_scsi_steps=12 (fixed).
# Full grid + caveats: ./README.md
#
# --time=24:00:00 is a flat budget across the whole grid (user's choice, not a measured number).
# --warmup_batch_size 16 is passed explicitly: args.py overrides the mstep/estep batch defaults
# for 3D but NOT the warmup one, which would otherwise fall back to 258 and OOM in warmup.
# Batch 16 was profiled at ~77% of a 48GB L40; --gres=gpu:1 is generic, so a job scheduled onto
# a <40GB card (e.g. V100 32GB) will OOM -- add a --constraint or drop the batch sizes to 12 if
# that happens. main.py has no checkpointing: a walltime kill loses the weights entirely; only
# the per-EM-step reconstruction/trajectory panels already logged to wandb survive.

cd /scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist3d

uv run python main.py \
    --digit_classes 3 \
    --n_images_per_class 6000 \
    --warmup_batch_size 16 \
    --mstep_batch_size 16 \
    --estep_batch_size 32 \
    --warmup_n_steps_train 40000 \
    --mstep_n_steps_train 2000 \
    --num_scsi_steps 12 \
    --wandb_project scsi-cryoet-mnist3d-three-warmup-mstep-grid \
    --wandb_run_name w40k_m2k
