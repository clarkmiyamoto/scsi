#!/bin/bash
#SBATCH --job-name=I2_tomo_tight
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

# Group I — Tomo-quantile: tomo_quantile=0.3 (tighter than default 0.15).
# Hypothesis: stricter space-carving discards more borderline voxels, producing a
# cleaner but sparser pseudo-inverse cloud. Cleaner initialization → fewer EM steps
# to converge, at the cost of potentially missing some torus surface points.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --tomo-quantile 0.3 \
    --out toy_3d_pc_checkpoints/I2_tomo_tight.pt \
    --eval-dir toy_3d_pc_eval_I2_tomo_tight \
    --wandb-name I2_tomo_tight
