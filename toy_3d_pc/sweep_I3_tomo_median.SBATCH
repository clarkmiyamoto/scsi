#!/bin/bash
#SBATCH --job-name=I3_tomo_median
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

# Group I — Tomo-quantile: tomo_quantile=0.5 (median over tilts = most conservative).
# Hypothesis: only voxels with occupancy ≥ median across ALL K tilts are kept.
# For a torus with K=32 tilts, the torus ring appears in roughly half the tilt views
# (the other half occlude part of the ring), so median carving may sharply over-carve.
# This tests the lower bound of carving quality; if it still helps warm-start, F-dagger
# is robust; if it collapses the cloud, that sets the upper bound for this parameter.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --tomo-quantile 0.5 \
    --out toy_3d_pc_checkpoints/I3_tomo_median.pt \
    --eval-dir toy_3d_pc_eval_I3_tomo_median \
    --wandb-name I3_tomo_median
