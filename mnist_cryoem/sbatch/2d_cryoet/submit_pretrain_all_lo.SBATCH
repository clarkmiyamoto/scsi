#!/bin/bash
#SBATCH --job-name=pretrain_all_lo
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

# Pretrain matrix, axis: class coverage x pool size/class (see mnist_cryoem/CLAUDE.md).
# all_lo: ALL 10 digit classes, 32 images/class -- full coverage, starved.
# Compare against: all_hi (same coverage, 516 img/class), single_lo (same pool size, 1 class).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_lo

uv run python pretrain.py \
    --n_pretrain_images_per_class 32 \
    --n_steps 10000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_all_lo.pt
