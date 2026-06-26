#!/bin/bash
#SBATCH --job-name=B1_ema_fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group B — EMA decay sweep: very fast prior adaptation
# H2: ema_decay=0.9 — prior updates aggressively each outer EM step; may destabilize

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --ema-decay 0.9 \
    --out toy_3d_pc_checkpoints/B1_ema_fast.pt \
    --eval-dir toy_3d_pc_eval_B1_ema_fast \
    --wandb-name B1_ema_fast
