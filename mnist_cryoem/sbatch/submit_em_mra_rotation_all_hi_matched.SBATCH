#!/bin/bash
#SBATCH --job-name=em_mra_rotation_all_hi_matched
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
# all_hi_matched: pretrained on all 10 classes (516 img/class), EM problem also all 10.
# This is the ceiling run of the MRA matrix. Also the steps_per_em=200 (default) point for
# Group D's sweep -- compare against em_mra_rotation_all_hi_steps{1,10,50}.
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_all_hi_matched

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_all_hi.pt \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_all_hi_matched
