#!/bin/bash
#SBATCH --job-name=A5_full_obs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group A — Coupling sweep: always use real observation (oracle upper bound on obs coupling)
# H1: alpha_z=0.0, alpha_y=1.0 — conditioning is always the true y; tests whether self-generated
# ŷ = F(x̂) hurts at all vs. using ground-truth observations throughout training

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --alpha-z 0.0 \
    --alpha-y 1.0 \
    --out toy_3d_pc_checkpoints/A5_full_obs.pt \
    --eval-dir toy_3d_pc_eval_A5_full_obs \
    --wandb-name A5_full_obs
