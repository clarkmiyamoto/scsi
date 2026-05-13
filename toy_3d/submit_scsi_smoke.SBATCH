#!/bin/bash
#SBATCH --job-name=toy3d_scsi_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

REPO_DIR=toy_3d
TARGET_FILE=scsi_train.py

cd /scratch/cm6627/scsi/$REPO_DIR

uv run $TARGET_FILE --debug
