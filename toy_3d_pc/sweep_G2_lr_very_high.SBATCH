#!/bin/bash
#SBATCH --job-name=G2_lr_vhigh
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

# Group G — Learning rate: lr=1e-3 (5x default).
# Hypothesis: same as G1 but more aggressive. If GVP+0.999 EMA stabilizes the
# EM loop sufficiently, 1e-3 could halve the required em-steps. Watch for
# divergence / oscillating loss as the failure mode.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --lr 1e-3 \
    --out toy_3d_pc_checkpoints/G2_lr_very_high.pt \
    --eval-dir toy_3d_pc_eval_G2_lr_very_high \
    --wandb-name G2_lr_very_high
