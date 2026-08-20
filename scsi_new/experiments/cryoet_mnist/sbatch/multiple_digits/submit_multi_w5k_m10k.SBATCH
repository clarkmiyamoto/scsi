#!/bin/bash
#SBATCH --job-name=multi_w5k_m10k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Warmup x M-step "amount of training" grid, all 10 digit classes (6,000 images/class each).
# This cell: warmup_n_steps_train=5000, mstep_n_steps_train=10000, num_scsi_steps=40 (fixed).
# Full grid + caveats: ../multiple_digits/README.md
#
# --time=12:00:00 is a flat budget across the whole grid (set at the user's explicit request,
# not measured) -- throughput extrapolated from a sibling codebase's comparable run suggests
# several cells in this grid, especially the higher mstep_n_steps_train ones, likely will NOT
# finish all 40 EM steps inside 12h. main.py has no checkpointing, so a walltime kill loses the
# run's weights entirely; whatever reconstruction/trajectory panels had already been logged to
# wandb (every EM step) are what survive.

cd /scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist

uv run main.py \
    --n_images_per_class 6000 \
    --warmup_n_steps_train 5000 \
    --mstep_n_steps_train 10000 \
    --num_scsi_steps 40 \
    --wandb_project scsi-cryoet-mnist-multi-warmup-mstep-grid \
    --wandb_run_name w5k_m10k
