#!/bin/bash
#SBATCH --job-name=C1_steps8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group C — ODE fidelity sweep: 8 Euler steps (8x cheaper than default 64)
# H3: very coarse ODE; x_hat training targets have high variance — tests if coarse
# transport is still informative enough for the velocity field to learn

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --sample-steps 8 \
    --out toy_3d_pc_checkpoints/C1_steps8.pt \
    --eval-dir toy_3d_pc_eval_C1_steps8 \
    --wandb-name C1_steps8
