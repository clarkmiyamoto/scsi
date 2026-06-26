#!/bin/bash
#SBATCH --job-name=H2_ode8_em800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/cm6627/scsi/toy_3d_pc/%j.out
#SBATCH --error=/scratch/cm6627/scsi/toy_3d_pc/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Group H — Iso-compute cheap ODE + more EM iters: sample_steps=8, em_steps=800.
# Hypothesis: push the cheap-ODE idea to its limit. 8 Euler steps may produce noisier
# x-hat targets but the 8x increase in EM prior updates could compensate. Extra 2 hours
# allocated for per-step overhead at 800 EM iters (logging, eval, etc.).
# Total ODE evals: 800 × 200 × 8 = 1,280,000 — identical to baseline.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --sample-steps 8 \
    --em-steps 800 \
    --out toy_3d_pc_checkpoints/H2_cheap_ode_8x_em.pt \
    --eval-dir toy_3d_pc_eval_H2_cheap_ode_8x_em \
    --wandb-name H2_cheap_ode_8x_em
