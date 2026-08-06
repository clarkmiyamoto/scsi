#!/bin/bash
#SBATCH --job-name=pretrain_few_hi_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# UNet counterpart of submit_pretrain_few_hi.SBATCH -- same pretrain matrix cell (digits
# {0, 3, 8}, 516 images/class), image branch swapped to ConditionalUNet2D via --image_arch
# unet (see model.py::build_image_branch). Compare against pretrain_few_hi.pt (DiT).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_few_hi_unet

uv run python pretrain.py \
    --image_arch unet \
    --digit_classes 0 3 8 \
    --n_pretrain_images_per_class 516 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_few_hi_unet.pt
