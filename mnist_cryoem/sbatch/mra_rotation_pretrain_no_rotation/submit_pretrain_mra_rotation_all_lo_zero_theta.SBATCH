#!/bin/bash
#SBATCH --job-name=pretrain_mra_rotation_all_lo_zero_theta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix for rotational-MRA channel, axis: class coverage x pool size/class.
# all_lo_zero_theta: same as all_lo (ALL 10 digit classes, 32 images/class) but with
# --zero_theta: theta is fixed to 0 every step instead of sampled, so F_MRA degenerates to
# y = x + W (no rotation) -- isolates the image branch from pose-branch noise.
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_mra_rotation_all_lo_zero_theta

uv run python pretrain_mra_rotation.py \
    --n_pretrain_images_per_class 32 \
    --n_steps 10000 \
    --batch_size 256 \
    --zero_theta \
    --out_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_all_lo_zero_theta.pt
