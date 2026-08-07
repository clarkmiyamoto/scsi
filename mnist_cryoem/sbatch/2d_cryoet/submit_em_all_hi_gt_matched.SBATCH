#!/bin/bash
#SBATCH --job-name=em_all_hi_gt_matched
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

# Baseline matrix, Group A (matched), GT-oracle ceiling: pretrained on pretrain_all_hi_gt.pt
# (--warmstart_target gt, literal clean digits, all 10 classes, 516 img/class) instead of
# pretrain_all_hi.pt (classical_recon). This is the TRUE ceiling of the whole matrix -- the
# most any pretrain cell could possibly achieve, since it's the only one whose Theta^(0) ever
# saw an actual clean digit. em_all_hi_matched (classical_recon) is compared against THIS run,
# not treated as the ceiling itself, to quantify the classical-recon-vs-GT warmstart gap that
# survives all the way through the EM loop.
# Compare against: em_all_hi_matched (same coverage/pool size, classical_recon pretrain
# instead of gt), em_none_matched (no pretraining at all).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_all_hi_gt_matched

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_all_hi_gt.pt \
    --ckpt_dir mnist_cryoem_checkpoints/em_all_hi_gt_matched
