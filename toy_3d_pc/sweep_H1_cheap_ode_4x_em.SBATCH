#!/bin/bash
#SBATCH --job-name=H1_ode16_em400
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cm6627/scsi/toy_3d_pc/%j.out
#SBATCH --error=/scratch/cm6627/scsi/toy_3d_pc/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Group H — Iso-compute cheap ODE + more EM iters: sample_steps=16, em_steps=400.
# Hypothesis: GVP's cosine schedule has no sharp kinks, so 16 Euler steps ≈ 64 steps
# in x-hat quality. The saved 4x compute per inner step is reinvested into 4x more
# outer EM prior updates (400 vs 100), which directly attacks slow convergence.
# Total ODE evals: 400 × 200 × 16 = 1,280,000 — identical to baseline 100 × 200 × 64.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --sample-steps 16 \
    --em-steps 400 \
    --out toy_3d_pc_checkpoints/H1_cheap_ode_4x_em.pt \
    --eval-dir toy_3d_pc_eval_H1_cheap_ode_4x_em \
    --wandb-name H1_cheap_ode_4x_em
