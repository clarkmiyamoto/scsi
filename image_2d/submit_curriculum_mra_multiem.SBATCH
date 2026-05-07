#!/bin/bash
#SBATCH --job-name=curriculum_mra_multiem      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)

REPO_DIR=image_2d
TARGET_FILE=main_warmup_curriculum.py

cd /scratch/cm6627/scsi/$REPO_DIR

# Multiple EM steps per curriculum level for more refinement at each ell level
uv run $TARGET_FILE \
    --corruption=mra \
    --n_ell_levels=10 \
    --epochs_per_level=20 \
    --em_steps_per_level=5 \
    --ell_min=0.0 \
    --ell_max=1.0 \
    --n_em_steps=2000 \
    --epochs_per_em=5