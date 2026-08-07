#!/bin/bash
#SBATCH --job-name=pretrain_all_lo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix v2 (classical-recon warm start). See submit_pretrain_single_lo.SBATCH for the
# full matrix rationale.
#
# all_lo: ALL 10 digit classes, 32 images/class -- full coverage, starved (320 objects x 4
# acquisitions = 1280 backproject() calls at pool-build time).
# Compare against: all_hi (same coverage, 516 img/class), single_lo (same pool size, 1 class).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_lo

uv run python pretrain.py \
    --warmstart_target classical_recon \
    --n_pretrain_images_per_class 32 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_all_lo.pt
