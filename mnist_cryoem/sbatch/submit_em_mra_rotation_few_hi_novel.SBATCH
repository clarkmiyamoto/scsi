#!/bin/bash
#SBATCH --job-name=em_mra_rotation_few_hi_novel
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

# Baseline matrix (rotational-MRA), Group B (novel-class generalization): pretrained narrower
# than the EM problem -- does SCSI/EM discover a class it never saw during pretraining?
# few_hi_novel: pretrained on {0,3,8}, EM problem sees {0,3,8,5} (5 is novel).
# Compare against few_hi_matched (same pretrain, no novel class).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_few_hi_novel

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_few_hi.pt \
    --digit_classes 0 3 8 5 \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_few_hi_novel
