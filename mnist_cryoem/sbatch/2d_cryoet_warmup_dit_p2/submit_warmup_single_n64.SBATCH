#!/bin/bash
#SBATCH --job-name=warmup_single_n64_dit_p2
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
# full matrix rationale (--dataset_split train, --corruptions_per_object 4, timing/memory
# measurements, --sample_steps floor).
#
# single_n64: digit {3} only, --n_images_per_class 64 -- narrowest coverage, smallest amount of
# corrupted data in the matrix.
# Compare against: single_n512/n2048/n4096 (same class, more data), few_n64/all_n64 (same
# amount, broader coverage).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_single_n64_dit_p2

uv run python main.py \
    --dataset_split train \
    --digit_classes 3 \
    --n_images_per_class 64 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --arch dit \
    --patch_size 2 \
    --warmup_steps 25000 \
    --steps_per_em 200 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_single_n64_dit_p2
