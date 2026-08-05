#!/bin/bash
#SBATCH --job-name=pretrain_single_lo_3d
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

# 3D/CryoET counterpart of submit_pretrain_single_lo.SBATCH -- same pretrain matrix cell
# (digit {3} ONLY, 32 volumes/class -- narrowest prior, most starved), via pretrain_3d.py.
# Compare against: single_hi_3d (same class, 516 vol/class), all_lo_3d (same pool size, all
# 10 classes).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_single_lo_3d

uv run python pretrain_3d.py \
    --digit_classes 3 \
    --n_pretrain_images_per_class 32 \
    --n_steps 9000 \
    --batch_size 8 \
    --out_ckpt mnist_cryoet_checkpoints/pretrain_single_lo_3d.pt
