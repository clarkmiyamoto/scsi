#!/bin/bash
#SBATCH --job-name=R1_trefoil_coupling_none
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group R — trefoil deep-dive: obs/noise coupling off.
# Baseline (trefoil, all defaults) is covered by sweep_L3_trefoil.SBATCH (+ N1/N2 n-points already scripted).

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape trefoil \
    --n-objects 400 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.0 \
    --alpha-y 0.0 \
    --ema-decay 0.999 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/R1_trefoil_coupling_none.pt \
    --eval-dir toy_3d_pc_eval_R1_trefoil_coupling_none \
    --wandb-name R1_trefoil_coupling_none
