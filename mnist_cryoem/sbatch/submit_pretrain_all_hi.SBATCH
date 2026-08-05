#!/bin/bash
#SBATCH --job-name=pretrain_all_hi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix, axis: class coverage x pool size/class (see mnist_cryoem/CLAUDE.md).
# all_hi: ALL 10 digit classes, 516 images/class -- full coverage, well-fed. This is the
# "best" checkpoint in the matrix -- the ceiling every other checkpoint is compared against.
# Compare against: all_lo (same coverage, 32 img/class), few_hi (same pool size, 3 classes).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_hi

uv run python pretrain.py \
    --n_pretrain_images_per_class 516 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_all_hi.pt
