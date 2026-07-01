#!/bin/bash
#SBATCH --job-name=L3_trefoil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group L — Five-shape baseline: trefoil, 400 corrupted samples, usual defaults.
# Explicit: GVP interpolant, 16 ODE sample-steps, regular (default) coupling.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape trefoil \
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
    --out toy_3d_pc_checkpoints/L_trefoil.pt \
    --eval-dir toy_3d_pc_eval_L_trefoil \
    --wandb-name L_trefoil
