#!/bin/bash
#SBATCH --job-name=pretrain_few_hi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix v2 (classical-recon warm start). See submit_pretrain_single_lo.SBATCH for the
# full matrix rationale.
#
# few_hi: digits {0, 3, 8} (visually distinct strokes), 516 images/class -- partial coverage,
# well-fed. --time bumped to 03:00:00: 1548 objects x 4 acquisitions = 6192 unbatched
# backproject() calls at pool-build time.
# Compare against: single_hi (1 of these 3 classes only), all_hi (all 10 classes),
# few_hi_cpo1 (same cell, corruptions_per_object=1 -- what does the extra reconstruction
# diversity actually buy?).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_few_hi

uv run python pretrain.py \
    --warmstart_target classical_recon \
    --digit_classes 0 3 8 \
    --n_pretrain_images_per_class 516 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_few_hi.pt
