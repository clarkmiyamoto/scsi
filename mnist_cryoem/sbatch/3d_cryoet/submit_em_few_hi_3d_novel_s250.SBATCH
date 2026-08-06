#!/bin/bash
#SBATCH --job-name=em_few_hi_3d_novel_s250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET pretraining-amount matrix, Group B (novel-class generalization): pretrained
# narrower than the EM problem -- does SCSI/EM discover a class it never saw during
# pretraining? few_hi_3d_novel: pretrained on {0,3,8}, EM problem sees {0,3,8,5} (5 is
# novel). Compare against few_hi_3d_matched (same pretrain, no novel class).
#
# --n_em_steps/--steps_per_em bumped from the initial 100/40 to 200/250 here (--time bumped to
# 32:00:00 to match) -- see submit_em_all_hi_3d_matched.SBATCH for the throughput-scaling
# explanation. Same K/steps_per_em used throughout the 3D matrix so comparisons stay
# apples-to-apples.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_few_hi_3d_novel_s250

uv run python main_3d.py \
    --init_ckpt mnist_cryoet_checkpoints/pretrain_few_hi_3d.pt \
    --digit_classes 0 3 8 5 \
    --n_em_steps 200 \
    --steps_per_em 250 \
    --ckpt_dir mnist_cryoet_checkpoints/em_few_hi_3d_novel_s250
