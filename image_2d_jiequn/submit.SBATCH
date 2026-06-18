#!/bin/bash
#SBATCH --job-name=mnist_transformer     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)

REPO_DIR=image_2d_jiequn
TARGET_FILE=main.py

cd /scratch/cm6627/scsi/$REPO_DIR

uv run $TARGET_FILE --K 16 --tilt_span_deg 60 --epsilon 0.0 --train_steps 50000 --warmup_steps 6000 --transport_steps 100 --ode_steps 64 --alpha 1.0 --lr 3e-4 --batch_size 256 --model_channels 32 --ema_decay 0.99
