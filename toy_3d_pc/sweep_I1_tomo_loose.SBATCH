#!/bin/bash
#SBATCH --job-name=I1_tomo_loose
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

# Group I — Tomo-quantile (F-dagger quality): tomo_quantile=0.05 (very loose carving).
# Hypothesis: a lower quantile retains more voxels in the back-projection → denser
# but noisier pseudo-inverse clouds. This gives the warm-start more signal to fit but
# may introduce spurious density outside the torus. Prediction: fast early convergence
# that plateaus if the noise overwhelms the structure.

cd /scratch/cm6627/scsi

uv run python -m toy_3d_pc scsi \
    --shape torus \
    --tomo-quantile 0.05 \
    --out toy_3d_pc_checkpoints/I1_tomo_loose.pt \
    --eval-dir toy_3d_pc_eval_I1_tomo_loose \
    --wandb-name I1_tomo_loose
