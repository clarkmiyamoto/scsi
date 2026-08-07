#!/bin/bash
#SBATCH --job-name=warmup_all_n512_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# _unet variant: adds --arch unet on top of sbatch/2d_cryoet_warmup/'s matrix (no --patch_size --
# DiT-only, ignored under --arch unet) -- see submit_warmup_all_n2048.SBATCH in this directory
# for the full arch rationale.
#

# Warmup matrix, core 3x4 grid (coverage x amount). See submit_warmup_all_n2048.SBATCH for the
# full matrix rationale.
#
# all_n512: all 10 digit classes (--digit_classes omitted -> default), --n_images_per_class 512.
# Compare against: all_n64/n2048/n4096 (same coverage, other amounts), single_n512/few_n512
# (same amount, narrower coverage).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_all_n512_unet

uv run python main.py \
    --dataset_split train \
    --n_images_per_class 512 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --arch unet \
    --warmup_steps 25000 \
    --steps_per_em 200 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_all_n512_unet
