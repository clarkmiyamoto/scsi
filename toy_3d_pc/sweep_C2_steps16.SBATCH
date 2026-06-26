#!/bin/bash
#SBATCH --job-name=C2_steps16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group C — ODE fidelity sweep: 16 Euler steps (4x cheaper than default 64)
# H3: hypothesis this is the sweet spot — cheap enough to allow more EM iters
# in a fixed time budget while still producing reasonable x_hat targets

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --sample-steps 16 \
    --out toy_3d_pc_checkpoints/C2_steps16.pt \
    --eval-dir toy_3d_pc_eval_C2_steps16 \
    --wandb-name C2_steps16
