#!/bin/bash
#SBATCH --job-name=pretrain_single_lo_unet
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

# UNet counterpart of submit_pretrain_single_lo.SBATCH -- same pretrain matrix cell (digit
# {3} ONLY, 32 images/class, most starved), image branch swapped to ConditionalUNet2D via
# --image_arch unet (see model.py::build_image_branch). Compare against
# pretrain_single_lo.pt (DiT) and pretrain_single_hi_unet.pt (same arch, well-fed).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_single_lo_unet

uv run python pretrain.py \
    --image_arch unet \
    --digit_classes 3 \
    --n_pretrain_images_per_class 32 \
    --n_steps 10000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_single_lo_unet.pt
