#!/bin/bash
#SBATCH --job-name=P3_mix_heun8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group P — cross-shape mixture generalization + integrator: all 5 shapes as a uniform
# mixture, Heun integrator, 8 sample-steps -- tests whether Heun's 2nd-order accuracy lets
# sample-steps drop below the Euler-only floor (16) found in Group K.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus dumbbell trefoil l_shape t_shape \
    --n-objects 400 \
    --interpolant-style gvp \
    --integrator heun \
    --sample-steps 8 \
    --alpha-z 0.05 \
    --alpha-y 0.05 \
    --ema-decay 0.999 \
    --em-steps 100 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --out toy_3d_pc_checkpoints/P3_mix_heun8.pt \
    --eval-dir toy_3d_pc_eval_P3_mix_heun8 \
    --wandb-name P3_mix_heun8
