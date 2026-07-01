#!/bin/bash
#SBATCH --job-name=S4_l_shape_canon_couplingoff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Group S — canonicalization interaction sweep, l_shape solo. Axis: coupling off.
# Hypothesis: canonicalize removes the rotation-ambiguity that alpha_z/alpha_y coupling
# was compensating for (parameters.md constraint #5), so alpha=0 may no longer hurt once
# --canonicalize is on. Compare against S1_l_shape_canon (baseline coupling 0.05/0.05).

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape l_shape \
    --n-objects 2048 \
    --interpolant-style gvp \
    --sample-steps 16 \
    --alpha-z 0.0 \
    --alpha-y 0.0 \
    --ema-decay 0.999 \
    --em-steps 500 \
    --training-steps 200 \
    --pretrain-steps 2000 \
    --tomo-quantile 0.15 \
    --canonicalize \
    --out toy_3d_pc_checkpoints/S4_l_shape_canon_couplingoff.pt \
    --eval-dir toy_3d_pc_eval_S4_l_shape_canon_couplingoff \
    --wandb-name S4_l_shape_canon_couplingoff
