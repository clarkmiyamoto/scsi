#!/bin/bash
#SBATCH --job-name=M2_l_shape_r12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group M — splat radius vs. shape thinness: l_shape, radius=0.12 (looser than default 0.08).
# Baseline (radius=0.08) is covered by sweep_L4_l_shape.SBATCH.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape \
    --n-objects 400 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 100 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --radius 0.12 \
    --out toy_3d_pc_checkpoints/M2_l_shape_r12.pt \
    --eval-dir toy_3d_pc_eval_M2_l_shape_r12 \
    --wandb-name M2_l_shape_r12
