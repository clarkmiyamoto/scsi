#!/bin/bash
#SBATCH --job-name=mnist_transformer     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)

REPO_DIR=scsi
TARGET_FILE=scsi_mnist_transformer.py

cd /scratch/cm6627/$REPO_DIR

uv run $TARGET_FILE --corruption=awgn --epochs_per_em=2 --epochs_first_pass=10
