#!/bin/bash
#SBATCH --job-name=S5_l_shape_canon_emafast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization interaction sweep, l_shape solo. Axis: faster outer EMA.
# Hypothesis: a stabler (derotated) interpolant target lets Theta_EMA track Theta more
# closely without the instability that made fast EMA (0.9-0.99) hurt pre-canonicalize
# (parameters.md constraint #2). Compare against S1_l_shape_canon (baseline ema 0.999).

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape \
    --n-objects 2048 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.995 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --canonicalize \
    --out toy_3d_pc_checkpoints/S5_l_shape_canon_emafast.pt \
    --eval-dir toy_3d_pc_eval_S5_l_shape_canon_emafast \
    --wandb-name S5_l_shape_canon_emafast
