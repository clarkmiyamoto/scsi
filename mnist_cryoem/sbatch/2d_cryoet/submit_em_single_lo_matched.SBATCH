#!/bin/bash
#SBATCH --job-name=em_single_lo_matched
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

# Baseline matrix, Group A (matched): pretrain coverage == EM problem coverage.
# single_lo_matched: pretrained on {3} only (32 img/class, starved), EM problem also {3}.
# Compare against: single_hi_matched (same class, well-fed pretrain).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_single_lo_matched

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_single_lo.pt \
    --digit_classes 3 \
    --ckpt_dir mnist_cryoem_checkpoints/em_single_lo_matched
