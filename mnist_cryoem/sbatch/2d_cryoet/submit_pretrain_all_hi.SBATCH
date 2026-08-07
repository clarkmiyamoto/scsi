#!/bin/bash
#SBATCH --job-name=pretrain_all_hi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix v2 (classical-recon warm start). See submit_pretrain_single_lo.SBATCH for the
# full matrix rationale.
#
# all_hi: ALL 10 digit classes, 516 images/class -- full coverage, well-fed. This is the
# "best" classical-recon checkpoint in the matrix -- the ceiling every other classical-recon
# cell is compared against (for the true, GT-oracle ceiling see pretrain_all_hi_gt.pt).
# --time bumped to 04:00:00: 5160 objects x 4 acquisitions = 20640 unbatched backproject()
# calls at pool-build time before step 1 even starts -- by far the heaviest pool-build in
# the matrix.
# Compare against: all_lo (same coverage, 32 img/class), few_hi (same pool size, 3 classes),
# all_hi_gt (same coverage/pool size, GT-supervised instead of classical-recon).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_hi

uv run python pretrain.py \
    --warmstart_target classical_recon \
    --n_pretrain_images_per_class 516 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_all_hi.pt
