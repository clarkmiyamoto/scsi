#!/bin/bash
#SBATCH --job-name=em_mra_rotation_single_hi_matched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Baseline matrix (rotational-MRA), Group A (matched): pretrain coverage == EM problem coverage.
# single_hi_matched: pretrained on {3} only (516 img/class), EM problem also {3} only.
# Compare against: single_lo_matched (same class, 32 img/class), few_hi_matched (same pool size, 3 classes).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_single_hi_matched

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_single_hi.pt \
    --digit_classes 3 \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_single_hi_matched
