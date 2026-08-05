#!/bin/bash
#SBATCH --job-name=pretrain_all_lo_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# UNet counterpart of submit_pretrain_all_lo.SBATCH -- same pretrain matrix cell (ALL 10
# digit classes, 32 images/class, starved), image branch swapped to ConditionalUNet2D via
# --image_arch unet (see model.py::build_image_branch). Compare against pretrain_all_lo.pt
# (DiT) and pretrain_all_hi_unet.pt (same arch, well-fed).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_lo_unet

uv run python pretrain.py \
    --image_arch unet \
    --n_pretrain_images_per_class 32 \
    --n_steps 10000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_all_lo_unet.pt
