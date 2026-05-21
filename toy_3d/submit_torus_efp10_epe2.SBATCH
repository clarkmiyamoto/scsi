#!/bin/bash
#SBATCH --job-name=torus_efp10_epe2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

REPO_DIR=toy_3d
TARGET_FILE=scsi_train.py

cd /scratch/cm6627/scsi/$REPO_DIR

uv run $TARGET_FILE \
    --shapes torus \
    --batch_size 64 \
    --epochs_first_pass 10 \
    --epochs_per_em 2 \
    --n_em_steps 400 \
    --bootstrap revolve
