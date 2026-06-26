#!/bin/bash
#SBATCH --job-name=D3_heavy_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

# Group D — Pretrain initialization: heavy warm-start (5000 steps, 2.5x more than default)
# H4: tests whether extended Algorithm 1 produces a meaningfully better Theta^(0)
# Extra 2 hours allocated due to longer pretrain phase

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --pretrain-steps 5000 \
    --out toy_3d_pc_checkpoints/D3_heavy_pretrain.pt \
    --eval-dir toy_3d_pc_eval_D3_heavy_pretrain \
    --wandb-name D3_heavy_pretrain
