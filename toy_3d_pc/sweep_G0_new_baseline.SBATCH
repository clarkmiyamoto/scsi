#!/bin/bash
#SBATCH --job-name=G0_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cm6627/scsi/toy_3d_pc/%j.out
#SBATCH --error=/scratch/cm6627/scsi/toy_3d_pc/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Group G — NEW BASELINE: explicit GVP + ema=0.999 (all other values are defaults).
# All A-E sweeps ran under the OLD defaults (linear interpolant, ema=0.995).
# This establishes the reference point for groups G/H/I under the new default regime.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --interpolant-style gvp \
    --ema-decay 0.999 \
    --lr 2e-4 \
    --sample-steps 64 \
    --em-steps 100 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/G0_new_baseline.pt \
    --eval-dir toy_3d_pc_eval_G0_new_baseline \
    --wandb-name G0_new_baseline
