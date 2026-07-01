#!/bin/bash
#SBATCH --job-name=S2_torus_canon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization ablation. Baseline recipe + --canonicalize, torus solo.
# NOTE (per CLAUDE.md): torus is continuously symmetric, so pca_canonicalize's in-plane
# angle is ill-defined here -- expected outcome is no-op-or-worse vs S2b, not an improvement.
# Included anyway as a direct empirical check. Control: S2b_torus_nocanon.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
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
    --out toy_3d_pc_checkpoints/S2_torus_canon.pt \
    --eval-dir toy_3d_pc_eval_S2_torus_canon \
    --wandb-name S2_torus_canon
