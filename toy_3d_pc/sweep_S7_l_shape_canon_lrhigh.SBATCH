#!/bin/bash
#SBATCH --job-name=S7_l_shape_canon_lrhigh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization interaction sweep, l_shape solo. Axis: higher lr (5e-4).
# Hypothesis: a cleaner (derotated) regression target gives a less noisy gradient signal,
# so the 1e-3+ lr regime that hurt pre-canonicalize (parameters.md constraint #4) may
# become viable at least at 5e-4. Compare against S1_l_shape_canon (baseline lr 2e-4).

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
    --lr 5e-4 \
    --canonicalize \
    --out toy_3d_pc_checkpoints/S7_l_shape_canon_lrhigh.pt \
    --eval-dir toy_3d_pc_eval_S7_l_shape_canon_lrhigh \
    --wandb-name S7_l_shape_canon_lrhigh
