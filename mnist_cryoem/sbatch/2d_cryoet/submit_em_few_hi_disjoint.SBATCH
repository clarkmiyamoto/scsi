#!/bin/bash
#SBATCH --job-name=em_few_hi_disjoint
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

# Baseline matrix, Group B (novel-class generalization), harsh end: em_few_hi_novel only adds
# ONE novel class on top of full overlap with the pretrain classes ({0,3,8} -> {0,3,8,5}). This
# run has ZERO class overlap with pretraining -- pretrained on {0,3,8}, EM problem is entirely
# {1,4,7}. Tests whether SCSI/EM can build a working prior for digit shapes the teacher never
# saw a single stroke of, relying purely on the pretrained model's general "digit-shaped
# blob" prior plus the EM loop's own data-driven refinement.
# Expectation per hypothesis: much weaker than em_few_hi_novel (partial overlap) and
# potentially close to em_none_matched (no prior at all) if the pretrained prior doesn't
# transfer across classes.
# Compare against: em_few_hi_novel (partial overlap), em_few_hi_matched (no novel classes),
# em_none_matched (no pretraining at all).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_few_hi_disjoint

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_few_hi.pt \
    --digit_classes 1 4 7 \
    --ckpt_dir mnist_cryoem_checkpoints/em_few_hi_disjoint
