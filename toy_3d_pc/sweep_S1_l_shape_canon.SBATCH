#!/bin/bash
#SBATCH --job-name=S1_l_shape_canon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization ablation. Baseline recipe + --canonicalize, l_shape solo.
# Control counterpart (identical flags, no --canonicalize): S1b_l_shape_nocanon.
# n-objects bumped 400->2048 and em-steps 100->500 relative to the old Group L baseline
# (matches the Q/R re-ablation standard); l_shape has no continuous symmetry so its
# in-plane angle is well-defined under canonicalize (see CLAUDE.md canonicalize.py note).

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
    --canonicalize \
    --out toy_3d_pc_checkpoints/S1_l_shape_canon.pt \
    --eval-dir toy_3d_pc_eval_S1_l_shape_canon \
    --wandb-name S1_l_shape_canon
