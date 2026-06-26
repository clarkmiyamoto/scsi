#!/bin/bash
#SBATCH --job-name=D2_light_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group D — Pretrain initialization: light warm-start (500 steps, 4x less than default)
# H4: tests whether the first ~25% of default warm-start is most of the benefit

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --pretrain-steps 500 \
    --out toy_3d_pc_checkpoints/D2_light_pretrain.pt \
    --eval-dir toy_3d_pc_eval_D2_light_pretrain \
    --wandb-name D2_light_pretrain
