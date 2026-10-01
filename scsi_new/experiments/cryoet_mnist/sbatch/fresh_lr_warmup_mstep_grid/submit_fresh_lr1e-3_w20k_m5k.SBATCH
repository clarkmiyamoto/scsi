#!/bin/bash -l
#SBATCH --job-name=fresh_lr1e-3_w20k_m5k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=rocky9&(h100|a100-80gb)
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_mnist/slurm-%x-%j.out
#SBATCH --error=/mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_mnist/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Fresh-student lr x warmup x M-step grid, digit class 3 only (6,000 images), --student_init fresh.
# This cell: mstep_lr=1e-3, warmup_n_steps_train=20000, mstep_n_steps_train=5000, num_scsi_steps=40.
# warmup_lr is held at 3e-4 in every cell. Full grid + caveats: README.md in this directory.

module load python/3.13.2 uv

cd /mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_mnist

uv run main.py \
    --student_init fresh \
    --digit_classes 3 \
    --n_images_per_class 6000 \
    --warmup_lr 3e-4 \
    --warmup_n_steps_train 20000 \
    --mstep_lr 1e-3 \
    --mstep_n_steps_train 5000 \
    --num_scsi_steps 40 \
    --wandb_project scsi-cryoet-mnist-three-fresh-lr-warmup-mstep \
    --wandb_run_name lr1e-3_w20k_m5k
