#!/bin/bash
#SBATCH --job-name=em_few_hi_extract
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
# few_hi_extract: pretrained on {0,3,8}, EM problem restricted to {8} only.
# Compare against few_hi_matched (EM problem == full pretrain coverage).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_few_hi_extract

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_few_hi.pt \
    --digit_classes 8 \
    --ckpt_dir mnist_cryoem_checkpoints/em_few_hi_extract
