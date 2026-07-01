#!/bin/bash
#SBATCH --job-name=R11_trefoil_linear_interp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group R — trefoil deep-dive: linear interpolant instead of GVP.
# Baseline (trefoil, all defaults) is covered by sweep_L3_trefoil.SBATCH (+ N1/N2 n-points already scripted).

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape trefoil \
    --n-objects 400 \
    --interpolant-style linear \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/R11_trefoil_linear_interp.pt \
    --eval-dir toy_3d_pc_eval_R11_trefoil_linear_interp \
    --wandb-name R11_trefoil_linear_interp
