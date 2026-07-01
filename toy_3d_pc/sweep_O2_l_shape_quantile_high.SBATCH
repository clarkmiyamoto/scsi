#!/bin/bash
#SBATCH --job-name=O2_l_shape_q30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group O — F-dagger tomo-quantile per shape class: l_shape, tomo-quantile=0.3 (tighter carve).
# Baseline (tomo-quantile=0.15) is covered by sweep_L4_l_shape.SBATCH.

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
    --tomo-quantile 0.3 \
    --out toy_3d_pc_checkpoints/O2_l_shape_q30.pt \
    --eval-dir toy_3d_pc_eval_O2_l_shape_q30 \
    --wandb-name O2_l_shape_q30
