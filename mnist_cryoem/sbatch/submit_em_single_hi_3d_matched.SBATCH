#!/bin/bash
#SBATCH --job-name=em_single_hi_3d_matched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET pretraining-amount matrix, Group A (matched): pretrain coverage == EM problem
# coverage -- ceiling reference for how well SCSI/EM recovers what it was already pretrained
# on. single_hi_3d_matched: pretrained on {3} only (516 vol/class), EM problem also
# restricted to {3}. Compare against: single_lo_3d_matched (same class, starved pretrain),
# few_hi_3d_matched.
#
# --n_em_steps/--steps_per_em bumped from the initial 100/40 to 200/100 (--time bumped to
# 13:00:00 to match) -- see submit_em_all_hi_3d_matched.SBATCH for the throughput-scaling
# explanation. Same K/steps_per_em used here so the comparison stays apples-to-apples.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_single_hi_3d_matched

uv run python main_3d.py \
    --init_ckpt mnist_cryoet_checkpoints/pretrain_single_hi_3d.pt \
    --digit_classes 3 \
    --n_em_steps 200 \
    --steps_per_em 100 \
    --ckpt_dir mnist_cryoet_checkpoints/em_single_hi_3d_matched
