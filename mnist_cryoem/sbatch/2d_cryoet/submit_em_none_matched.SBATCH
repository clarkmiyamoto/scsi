#!/bin/bash
#SBATCH --job-name=em_none_matched
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

# Baseline matrix, Group E (cold-start floor): NO --init_ckpt -- Theta^(0) is a random init,
# same architecture/hyperparameters as every other Group A "matched" run, EM problem is all
# 10 digit classes. This is the zero point every pretrain-matrix cell should be measured
# against: "how much pretraining data do we need" has no answer without knowing what EM alone,
# with no warm start at all, already gets you. If em_all_lo_matched (32 img/class,
# classical-recon pretrain) doesn't clearly beat this, 32 img/class of pretraining is buying
# nothing.
# Compare against: em_all_lo_matched, em_all_hi_matched, em_all_hi_gt_matched.
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_none_matched

uv run python main.py \
    --ckpt_dir mnist_cryoem_checkpoints/em_none_matched
