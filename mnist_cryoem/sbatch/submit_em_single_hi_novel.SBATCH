#!/bin/bash
#SBATCH --job-name=em_single_hi_novel
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

# Baseline matrix, Group B (novel-class generalization): pretrained narrower than the EM
# problem -- does SCSI/EM discover a class it never saw during pretraining?
# single_hi_novel: pretrained on {3} only, EM problem sees {3,7} (7 is novel).
# Expectation per hypothesis: recovers 3 well, likely struggles on 7. Compare against
# single_hi_matched (same pretrain, no novel class).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_single_hi_novel

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_single_hi.pt \
    --digit_classes 3 7 \
    --ckpt_dir mnist_cryoem_checkpoints/em_single_hi_novel
