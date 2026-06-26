#!/bin/bash
#SBATCH --job-name=B2_ema_mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group B — EMA decay sweep: moderate prior adaptation (10x faster than default 0.995)
# H2: ema_decay=0.99 — hypothesis: better prior convergence than 0.995 without instability

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --ema-decay 0.99 \
    --out toy_3d_pc_checkpoints/B2_ema_mid.pt \
    --eval-dir toy_3d_pc_eval_B2_ema_mid \
    --wandb-name B2_ema_mid
