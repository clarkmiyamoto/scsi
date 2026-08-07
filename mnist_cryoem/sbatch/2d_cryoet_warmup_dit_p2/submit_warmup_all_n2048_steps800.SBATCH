#!/bin/bash
#SBATCH --job-name=warmup_all_n2048_steps800_dit_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=28:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# _dit_p2 variant: adds --arch dit --patch_size 2 on top of sbatch/2d_cryoet_warmup/'s matrix --
# see submit_warmup_all_n2048.SBATCH in this directory for the full arch/patch_size rationale.
#

# Warmup matrix, --steps_per_em sweep. See submit_warmup_all_n2048_steps400.SBATCH for the full
# sweep rationale (same baseline: all 10 classes, --n_images_per_class 2048, --sample_steps 64).
# steps_per_em=800: 4x the baseline M-step count (n_em_steps default 200 x 800 = 160,000 M-step
# SGD steps total) -- --time bumped to 28h accordingly.
# Compare against: submit_warmup_all_n2048.SBATCH (steps_per_em=200), steps400 (steps_per_em=400).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_all_n2048_steps800_dit_p2

uv run python main.py \
    --dataset_split train \
    --n_images_per_class 2048 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --arch dit \
    --patch_size 2 \
    --warmup_steps 25000 \
    --steps_per_em 800 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_all_n2048_steps800_dit_p2
