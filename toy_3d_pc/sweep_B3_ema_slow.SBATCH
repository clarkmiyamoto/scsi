#!/bin/bash
#SBATCH --job-name=B3_ema_slow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group B — EMA decay sweep: very slow prior adaptation
# H2: ema_decay=0.999 — prior barely updates; after 100 outer steps it has only incorporated
# ~10% of the current model weights; may stall the EM loop

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --ema-decay 0.999 \
    --out toy_3d_pc_checkpoints/B3_ema_slow.pt \
    --eval-dir toy_3d_pc_eval_B3_ema_slow \
    --wandb-name B3_ema_slow
