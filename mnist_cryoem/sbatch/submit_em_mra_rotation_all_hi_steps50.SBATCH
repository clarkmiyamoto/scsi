#!/bin/bash
#SBATCH --job-name=em_mra_rotation_all_hi_steps50
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

# Baseline matrix (rotational-MRA), Group D (steps_per_em sweep): how many M-step SGD steps per
# E-step regeneration is actually needed? Pretrain (all_hi) and EM problem (all 10 classes) held
# fixed at the same setting as em_mra_rotation_all_hi_matched -- only steps_per_em changes.
# steps_per_em=50: heavier amortization -- one Phi^(k-1) pool regeneration is reused across
# 50 SGD steps before refreshing. More M-step drift, less E-step overhead.
# Compare against em_mra_rotation_all_hi_steps1, em_mra_rotation_all_hi_steps10,
# em_mra_rotation_all_hi_matched (steps_per_em=200).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_all_hi_steps50

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_all_hi.pt \
    --steps_per_em 50 \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_all_hi_steps50
