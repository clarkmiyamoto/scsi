#!/bin/bash
#SBATCH --job-name=warmup_few_n4096
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

# Warmup matrix, core 3x4 grid (coverage x amount). See submit_warmup_all_n2048.SBATCH for the
# full matrix rationale.
#
# few_n4096: digits {0, 3, 8}, --n_images_per_class 4096 -- 4096 x 3 x 4 = 49,152 backproject()
# acquisitions at pool-build time (~40s measured rate), x_recon_pool ~3.2 GB.
# Compare against: few_n64/n512/n2048 (same classes, other amounts), single_n4096/all_n4096
# (same amount, narrower/broader coverage).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=warmup_few_n4096

uv run python main.py \
    --dataset_split train \
    --digit_classes 0 3 8 \
    --n_images_per_class 4096 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --warmup_steps 25000 \
    --steps_per_em 200 \
    --sample_steps 64 \
    --ckpt_dir mnist_cryoem_checkpoints/warmup_few_n4096
