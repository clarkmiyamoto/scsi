#!/bin/bash
#SBATCH --job-name=G1_lr_high
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

# Group G — Learning rate: lr=5e-4 (2.5x default).
# Hypothesis: GVP preserves unit variance at all t, so velocity field gradients are
# smoother than linear OT's (which peaks at intermediate t). A higher LR should be
# stable under GVP and may let the inner loop converge in fewer training steps,
# cutting the EM iters needed to reach the same loss.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --lr 5e-4 \
    --out toy_3d_pc_checkpoints/G1_lr_high.pt \
    --eval-dir toy_3d_pc_eval_G1_lr_high \
    --wandb-name G1_lr_high
