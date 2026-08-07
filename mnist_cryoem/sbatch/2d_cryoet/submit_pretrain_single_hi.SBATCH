#!/bin/bash
#SBATCH --job-name=pretrain_single_hi
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

# Pretrain matrix v2 (classical-recon warm start). See submit_pretrain_single_lo.SBATCH for the
# full matrix rationale (why classical_recon replaces GT warmstart here, why n_steps is fixed
# at 20000 across every cell, why corruptions_per_object=4).
#
# single_hi: digit {3} ONLY, 516 images/class -- narrow prior, well-fed. --time bumped to
# 02:00:00 vs single_lo's 01:00:00: 516 objects x 4 acquisitions = 2064 unbatched backproject()
# calls at pool-build time before step 1 even starts.
# Compare against: single_lo (same class, starved pretrain), few_hi (same pool size, 3 classes).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_single_hi

uv run python pretrain.py \
    --warmstart_target classical_recon \
    --digit_classes 3 \
    --n_pretrain_images_per_class 516 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_single_hi.pt
