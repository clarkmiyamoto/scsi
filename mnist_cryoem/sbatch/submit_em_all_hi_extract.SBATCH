#!/bin/bash
#SBATCH --job-name=em_all_hi_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Baseline matrix, Group C (subclass extraction): pretrained broader than the EM problem --
# can SCSI/EM pick out just one class from a multi-class prior?
# all_hi_extract: pretrained on all 10 classes, EM problem restricted to {3} only.
# Compare against all_hi_matched (EM problem == full pretrain coverage) and
# single_hi_matched (same target class, but narrow pretrain instead of broad).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_all_hi_extract

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_all_hi.pt \
    --digit_classes 3 \
    --ckpt_dir mnist_cryoem_checkpoints/em_all_hi_extract
