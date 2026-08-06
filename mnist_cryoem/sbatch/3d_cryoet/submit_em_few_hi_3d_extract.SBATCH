#!/bin/bash
#SBATCH --job-name=em_few_hi_3d_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET pretraining-amount matrix, Group C (subclass extraction): pretrained broader than
# the EM problem -- can SCSI/EM pick out just one class from a multi-class prior?
# few_hi_3d_extract: pretrained on {0,3,8}, EM problem restricted to {8} only.
# Compare against few_hi_3d_matched (EM problem == full pretrain coverage).
#
# --n_em_steps/--steps_per_em bumped from the initial 100/40 to 200/100 (--time bumped to
# 13:00:00 to match) -- see submit_em_all_hi_3d_matched.SBATCH for the throughput-scaling
# explanation. Same K/steps_per_em used throughout the 3D matrix so comparisons stay
# apples-to-apples.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_few_hi_3d_extract

uv run python main_3d.py \
    --init_ckpt mnist_cryoet_checkpoints/pretrain_few_hi_3d.pt \
    --digit_classes 8 \
    --n_em_steps 200 \
    --steps_per_em 100 \
    --ckpt_dir mnist_cryoet_checkpoints/em_few_hi_3d_extract
