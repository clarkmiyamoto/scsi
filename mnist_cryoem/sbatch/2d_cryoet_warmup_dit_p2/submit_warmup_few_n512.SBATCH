#!/bin/bash
#SBATCH --job-name=warmup_few_n512_dit_p2
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

# _dit_p2 variant: adds --arch dit --patch_size 2 on top of sbatch/2d_cryoet_warmup/'s matrix --
# see submit_warmup_all_n2048.SBATCH in this directory for the full arch/patch_size rationale.
#

# Warmup matrix, core 3x4 grid (coverage x amount). See submit_warmup_all_n2048.SBATCH for the
# full matrix rationale.
#
# few_n512: digits {0, 3, 8}, --n_images_per_class 512.
# Compare against: few_n64/n2048/n4096 (same classes, other amounts), single_n512/all_n512
# (same amount, narrower/broader coverage).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_few_n512_dit_p2

uv run python main.py \
    --dataset_split train \
    --digit_classes 0 3 8 \
    --n_images_per_class 512 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --arch dit \
    --patch_size 2 \
    --warmup_steps 25000 \
    --steps_per_em 200 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_few_n512_dit_p2
