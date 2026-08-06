#!/bin/bash
#SBATCH --job-name=em_all_hi_3d_matched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET pretraining-amount matrix, Group A (matched): pretrain coverage == EM problem
# coverage. all_hi_3d_matched: pretrained on all 10 classes (516 vol/class), EM problem also
# all 10. Ceiling run of the 3D matrix -- compare against all_lo_3d_matched (same coverage,
# starved pretrain).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.
#
# --n_em_steps/--steps_per_em bumped from the initial 100/40 to 200/100 -- n_em_steps now
# matches main_3d.py's algorithmic default (200); steps_per_em stays below its default (200)
# since E-step cost is pool_size * sample_steps full 3D-UNet forward passes per EM iteration,
# where pool_size = min(N_obs, steps_per_em * batch_size). The 100/40 setting measured ~2.5h
# in practice (squeue/sacct); scaling that observed throughput linearly with the
# K*steps_per_em product (4,000 -> 20,000, 5x) targets ~12.5h, hence --time=13:00:00. Applied
# identically to all five *_3d_matched runs so the comparison stays apples-to-apples. The
# *_s250 copies (steps_per_em=250, product 50,000, 12.5x of the original) target ~31h, hence
# --time=32:00:00 there.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_all_hi_3d_matched

uv run python main_3d.py \
    --init_ckpt mnist_cryoet_checkpoints/pretrain_all_hi_3d.pt \
    --n_em_steps 200 \
    --steps_per_em 100 \
    --ckpt_dir mnist_cryoet_checkpoints/em_all_hi_3d_matched
