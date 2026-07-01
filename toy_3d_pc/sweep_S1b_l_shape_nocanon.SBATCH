#!/bin/bash
#SBATCH --job-name=S1b_l_shape_nocanon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization ablation. Control for S1_l_shape_canon: identical flags,
# --canonicalize omitted. n-objects/em-steps changed vs old Group L baseline, so this
# control is needed rather than reusing L4_l_shape for the comparison.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape \
    --n-objects 2048 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/S1b_l_shape_nocanon.pt \
    --eval-dir toy_3d_pc_eval_S1b_l_shape_nocanon \
    --wandb-name S1b_l_shape_nocanon
