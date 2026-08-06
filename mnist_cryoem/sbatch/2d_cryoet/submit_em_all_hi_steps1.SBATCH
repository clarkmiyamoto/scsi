#!/bin/bash
#SBATCH --job-name=em_all_hi_steps1
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

# Baseline matrix, Group D (steps_per_em sweep): how many M-step SGD steps per E-step
# regeneration is actually needed? Pretrain (all_hi) and EM problem (all 10 classes) held
# fixed at the same setting as em_all_hi_matched -- only steps_per_em changes.
# steps_per_em=1: literal EM pseudocode -- a completely fresh Phi^(k-1)-proposed pool every
# single SGD step. Most faithful to the algorithm, most E-step (ODE solve) overhead.
# Compare against em_all_hi_steps10, em_all_hi_steps50, em_all_hi_matched (steps_per_em=200).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_all_hi_steps1

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_all_hi.pt \
    --steps_per_em 1 \
    --ckpt_dir mnist_cryoem_checkpoints/em_all_hi_steps1
