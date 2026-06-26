#!/bin/bash
#SBATCH --job-name=A4_both_mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group A — Coupling sweep: both alpha_z and alpha_y at moderate level
# H1: alpha_z=0.2, alpha_y=0.2 — symmetric moderate coupling for both noise and observation

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --alpha-z 0.2 \
    --alpha-y 0.2 \
    --out toy_3d_pc_checkpoints/A4_both_mid.pt \
    --eval-dir toy_3d_pc_eval_A4_both_mid \
    --wandb-name A4_both_mid
