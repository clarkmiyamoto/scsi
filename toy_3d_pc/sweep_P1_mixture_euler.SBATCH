#!/bin/bash
#SBATCH --job-name=P1_mix_euler16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group P — cross-shape mixture generalization + integrator: all 5 shapes as a uniform
# mixture, Euler integrator, 16 sample-steps (baseline arm, matches Group L per-shape recipe).

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus dumbbell trefoil l_shape t_shape \
    --n-objects 400 \
    --interpolant-style gvp \
    --integrator euler \
    --sample-steps 16 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 100 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/P1_mix_euler16.pt \
    --eval-dir toy_3d_pc_eval_P1_mix_euler16 \
    --wandb-name P1_mix_euler16
