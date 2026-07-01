#!/bin/bash
#SBATCH --job-name=S10_mix_canon_ode8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization interaction sweep, l_shape+torus mixture. Axis: cheap ODE.
# Mirrors S6_l_shape_canon_ode8; compare against S3_mix_canon.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape torus \
    --n-objects 2048 \
    --interpolant-style gvp \
    --sample-steps 8 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --canonicalize \
    --out toy_3d_pc_checkpoints/S10_mix_canon_ode8.pt \
    --eval-dir toy_3d_pc_eval_S10_mix_canon_ode8 \
    --wandb-name S10_mix_canon_ode8
