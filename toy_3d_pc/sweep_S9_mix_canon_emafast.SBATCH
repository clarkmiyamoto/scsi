#!/bin/bash
#SBATCH --job-name=S9_mix_canon_emafast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization interaction sweep, l_shape+torus mixture. Axis: faster EMA.
# Mirrors S5_l_shape_canon_emafast; compare against S3_mix_canon.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape torus \
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
    --out toy_3d_pc_checkpoints/S9_mix_canon_emafast.pt \
    --eval-dir toy_3d_pc_eval_S9_mix_canon_emafast \
    --wandb-name S9_mix_canon_emafast
