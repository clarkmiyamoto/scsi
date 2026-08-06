#!/bin/bash
#SBATCH --job-name=em_mra_rotation_single_lo_matched_zero_theta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Zero-theta-pretrain warm-start ablation (rotational-MRA), Group A (matched): pretrain
# coverage == EM problem coverage. Same as em_mra_rotation_single_lo_matched, but
# warm-started from pretrain_mra_rotation_single_lo_zero_theta.pt -- pretrained with theta
# fixed to 0 (F_MRA degenerates to y = x + W, image branch only, pose head never trained on
# nonzero rotation) and starved (32 img/class). The EM stage itself is unchanged (real
# sampled rotation): tests whether the pose branch can bootstrap from scratch during EM
# despite this image-only, data-starved warm start. Compare against
# em_mra_rotation_single_lo_matched (same EM problem, ordinary pretrain) and
# em_mra_rotation_single_hi_matched_zero_theta (same class, 516 img/class).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_single_lo_matched_zero_theta

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_single_lo_zero_theta.pt \
    --digit_classes 3 \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_single_lo_matched_zero_theta
