#!/bin/bash
#SBATCH --job-name=Q19_l_shape_npoints_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group Q — l_shape deep-dive: n-points=1024 (denser).
# Baseline (l_shape, all defaults) is covered by sweep_L4_l_shape.SBATCH (+ M1/M2 radius, O1/O2 quantile already scripted).

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape \
    --n-objects 400 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --n-points 1024 \
    --out toy_3d_pc_checkpoints/Q19_l_shape_npoints_high.pt \
    --eval-dir toy_3d_pc_eval_Q19_l_shape_npoints_high \
    --wandb-name Q19_l_shape_npoints_high
