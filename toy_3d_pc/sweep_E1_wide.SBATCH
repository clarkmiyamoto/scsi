#!/bin/bash
#SBATCH --job-name=E1_wide
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group E — Budget structure: more outer EM iters, fewer inner SGD steps
# H5: em_steps=200, training_steps=100 — same total SGD steps as baseline (20,000)
# but 2x more EMA prior updates; tests if frequent prior refreshes help convergence

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --em-steps 200 \
    --training-steps 100 \
    --out toy_3d_pc_checkpoints/E1_wide.pt \
    --eval-dir toy_3d_pc_eval_E1_wide \
    --wandb-name E1_wide
