#!/bin/bash
#SBATCH --job-name=N1_trefoil_n256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group N — point-cloud density vs. shape complexity: trefoil, n_points=256 (below default 512).
# Baseline (n_points=512) is covered by sweep_L3_trefoil.SBATCH.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape trefoil \
    --n-objects 400 \
    --n-points 256 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 100 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/N1_trefoil_n256.pt \
    --eval-dir toy_3d_pc_eval_N1_trefoil_n256 \
    --wandb-name N1_trefoil_n256
