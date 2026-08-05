#!/bin/bash
#SBATCH --job-name=pretrain_all_lo_3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET counterpart of submit_pretrain_all_lo.SBATCH -- same pretrain matrix cell (ALL 10
# digit classes, 32 volumes/class -- full coverage, starved), via pretrain_3d.py.
# Compare against: all_hi_3d (same coverage, 516 vol/class), single_lo_3d (same pool size, 1
# class). NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_lo_3d

uv run python pretrain_3d.py \
    --n_pretrain_images_per_class 32 \
    --n_steps 9000 \
    --batch_size 8 \
    --out_ckpt mnist_cryoet_checkpoints/pretrain_all_lo_3d.pt
