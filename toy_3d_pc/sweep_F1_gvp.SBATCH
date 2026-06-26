#!/bin/bash
#SBATCH --job-name=F1_gvp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group F — Interpolant style: geometric variance-preserving (GVP) cosine schedule
# H6: gvp maintains unit variance throughout the ODE (alpha^2 + beta^2 = 1 at all t)
# unlike linear OT which has intermediate variance ~0.5 at t=0.5; may give smoother
# velocity fields for point cloud targets

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --interpolant-style gvp \
    --out toy_3d_pc_checkpoints/F1_gvp.pt \
    --eval-dir toy_3d_pc_eval_F1_gvp \
    --wandb-name F1_gvp
