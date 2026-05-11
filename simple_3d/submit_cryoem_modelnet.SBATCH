#!/bin/bash
#SBATCH --job-name=cryoem_3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

REPO_DIR=simple_3d
TARGET_FILE=main.py

cd /scratch/cm6627/scsi/$REPO_DIR

uv run $TARGET_FILE \
    --model=dit3d \
    --n_em_steps=500 \
    --epochs_per_em=5 \
    --epochs_first_pass=20 \
    --interpolant_style=linear
