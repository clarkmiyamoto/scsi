#!/bin/bash
#SBATCH --job-name=D1_no_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group D — Pretrain initialization: no warm-start (Algorithm 1 skipped)
# H4: EM starts from a randomly initialized velocity field; tests how much
# the F-dagger bootstrapped warm-start matters for downstream convergence

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --pretrain-steps 0 \
    --out toy_3d_pc_checkpoints/D1_no_pretrain.pt \
    --eval-dir toy_3d_pc_eval_D1_no_pretrain \
    --wandb-name D1_no_pretrain
