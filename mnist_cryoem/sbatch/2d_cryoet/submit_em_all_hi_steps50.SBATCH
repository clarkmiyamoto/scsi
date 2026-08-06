#!/bin/bash
#SBATCH --job-name=em_all_hi_steps50
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

# Baseline matrix, Group D (steps_per_em sweep). See em_all_hi_steps1 for the full comment.
# steps_per_em=50.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_all_hi_steps50

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_all_hi.pt \
    --steps_per_em 50 \
    --ckpt_dir mnist_cryoem_checkpoints/em_all_hi_steps50
