#!/bin/bash
#SBATCH --job-name=C3_steps32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group C — ODE fidelity sweep: 32 Euler steps (2x cheaper than default 64)
# H3: should be close to 64-step quality; useful to quantify the quality gap

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --sample-steps 32 \
    --out toy_3d_pc_checkpoints/C3_steps32.pt \
    --eval-dir toy_3d_pc_eval_C3_steps32 \
    --wandb-name C3_steps32
