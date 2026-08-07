#!/bin/bash
#SBATCH --job-name=warmup_all_n2048_steps400_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# _unet variant: adds --arch unet on top of sbatch/2d_cryoet_warmup/'s matrix (no --patch_size --
# DiT-only, ignored under --arch unet) -- see submit_warmup_all_n2048.SBATCH in this directory
# for the full arch rationale.
#

# Warmup matrix, --steps_per_em sweep: how many M-step SGD steps per E-step regeneration,
# floored at >= 200 per the requirement. Coverage/amount/--sample_steps held at
# submit_warmup_all_n2048.SBATCH's baseline (all 10 classes, --n_images_per_class 2048,
# --sample_steps 64) -- only --steps_per_em changes. That file's own value (200, the main.py
# default) is this sweep's first data point for free.
# steps_per_em=400: 2x the baseline M-step count (n_em_steps default 200 x 400 = 80,000 M-step
# SGD steps total, same 200 E-step ODE solves as every other cell) -- --time bumped from 16h to
# 20h accordingly (conservative; the E-step's ODE-solve cost doesn't grow with steps_per_em).
# Compare against: submit_warmup_all_n2048.SBATCH (steps_per_em=200), steps800 (steps_per_em=800).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_all_n2048_steps400_unet

uv run python main.py \
    --dataset_split train \
    --n_images_per_class 2048 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --arch unet \
    --warmup_steps 25000 \
    --steps_per_em 400 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_all_n2048_steps400_unet
