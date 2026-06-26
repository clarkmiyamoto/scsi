#!/bin/bash
#SBATCH --job-name=A1_no_coupling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group A — Coupling sweep: no coupling at all (ablation)
# H1: alpha_z=0.0, alpha_y=0.0 — pure self-generated conditioning, maximum self-consistency bias

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --alpha-z 0.0 \
    --alpha-y 0.0 \
    --out toy_3d_pc_checkpoints/A1_no_coupling.pt \
    --eval-dir toy_3d_pc_eval_A1_no_coupling \
    --wandb-name A1_no_coupling
