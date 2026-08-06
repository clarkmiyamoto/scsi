#!/bin/bash
#SBATCH --job-name=pretrain_mra_rotation_single_hi_zero_theta
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
# single_hi_zero_theta: same as single_hi (digit {3} ONLY, 516 images/class) but with
# --zero_theta: theta is fixed to 0 every step instead of sampled, so F_MRA degenerates to
# y = x + W (no rotation) -- isolates the image branch from pose-branch noise.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_mra_rotation_single_hi_zero_theta

uv run python pretrain_mra_rotation.py \
    --digit_classes 3 \
    --n_pretrain_images_per_class 516 \
    --n_steps 10000 \
    --batch_size 256 \
    --zero_theta \
    --out_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_single_hi_zero_theta.pt
