#!/bin/bash
#SBATCH --job-name=A0_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group A — Coupling sweep: BASELINE (all defaults explicit for reference)
# H1: alpha_z=0.05, alpha_y=0.05 (default coupling)

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.995 \
    --sample-steps 64 \
    --pretrain-steps 2000 \
    --em-steps 100 \
    --training-steps 200 \
    --interpolant-style linear \
    --out toy_3d_pc_checkpoints/A0_baseline.pt \
    --eval-dir toy_3d_pc_eval_A0_baseline \
    --wandb-name A0_baseline
