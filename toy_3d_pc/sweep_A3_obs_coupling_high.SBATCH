#!/bin/bash
#SBATCH --job-name=A3_obs_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group A — Coupling sweep: high obs coupling
# H1: alpha_z=0.05, alpha_y=0.5 — 50% of training steps use real observation as conditioning

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --alpha-z 0.05 \
    --alpha-y 0.5 \
    --out toy_3d_pc_checkpoints/A3_obs_coupling_high.pt \
    --eval-dir toy_3d_pc_eval_A3_obs_coupling_high \
    --wandb-name A3_obs_coupling_high
