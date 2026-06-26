#!/bin/bash
#SBATCH --job-name=A2_obs_mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group A — Coupling sweep: moderate obs coupling
# H1: alpha_z=0.05, alpha_y=0.2 — 20% of training steps use real observation as conditioning

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --alpha-z 0.05 \
    --alpha-y 0.2 \
    --out toy_3d_pc_checkpoints/A2_obs_coupling_mid.pt \
    --eval-dir toy_3d_pc_eval_A2_obs_coupling_mid \
    --wandb-name A2_obs_coupling_mid
