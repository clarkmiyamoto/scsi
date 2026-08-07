#!/bin/bash
#SBATCH --job-name=pretrain_few_hi_cpo1
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

# Pretrain matrix v2, reconstruction-richness ablation (NOT a class-coverage/pool-size cell --
# see submit_pretrain_single_lo.SBATCH for that axis). Same GT pool as few_hi (digits
# {0,3,8}, 516 img/class) but --corruptions_per_object 1 instead of 4: every object gets
# exactly ONE tilt-series reconstruction (one randomly-oriented missing wedge) instead of 4
# independently-oriented ones. Isolates whether the extra reconstruction diversity
# corruptions_per_object=4 buys is doing real work, or whether a single classical
# reconstruction per object is already enough signal for pretraining.
# --time 02:00:00 (vs few_hi's 03:00:00): pool-build is 4x cheaper (1548 backproject() calls
# instead of 6192).
# Compare against: few_hi (same cell, corruptions_per_object=4) -- downstream comparison is
# submit_em_few_hi_cpo1_matched.SBATCH vs submit_em_few_hi_matched.SBATCH.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_few_hi_cpo1

uv run python pretrain.py \
    --warmstart_target classical_recon \
    --digit_classes 0 3 8 \
    --n_pretrain_images_per_class 516 \
    --corruptions_per_object 1 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_few_hi_cpo1.pt
