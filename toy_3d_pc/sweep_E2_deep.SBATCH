#!/bin/bash
#SBATCH --job-name=E2_deep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group E — Budget structure: fewer outer EM iters, more inner SGD steps
# H5: em_steps=50, training_steps=400 — same total SGD steps as baseline (20,000)
# but 2x fewer EMA prior updates; each E-step has 2x more time to converge

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --em-steps 50 \
    --training-steps 400 \
    --out toy_3d_pc_checkpoints/E2_deep.pt \
    --eval-dir toy_3d_pc_eval_E2_deep \
    --wandb-name E2_deep
