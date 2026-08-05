#!/bin/bash
#SBATCH --job-name=em_mra_rotation_few_hi_matched
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
# few_hi_matched: pretrained on {0,3,8} (516 img/class), EM problem also {0,3,8}.
# Compare against: all_hi_matched (all 10 classes), single_hi_matched (1 class only).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_few_hi_matched

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_few_hi.pt \
    --digit_classes 0 3 8 \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_few_hi_matched
