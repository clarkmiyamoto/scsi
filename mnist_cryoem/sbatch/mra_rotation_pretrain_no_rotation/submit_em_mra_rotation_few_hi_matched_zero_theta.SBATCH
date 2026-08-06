#!/bin/bash
#SBATCH --job-name=em_mra_rotation_few_hi_matched_zero_theta
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

# Zero-theta + no-pose-head pretrain warm-start ablation (rotational-MRA), Group A (matched):
# pretrain coverage == EM problem coverage. Same as em_mra_rotation_few_hi_matched, but
# warm-started from pretrain_mra_rotation_few_hi_zero_theta.pt -- pretrained with theta fixed
# to 0 (F_MRA degenerates to y = x + W, no rotation) AND --no_pose_head (no pose branch at
# all, pure image-only Theta^(0)). The EM stage also runs --no_pose_head (required to load a
# checkpoint with no pose_branch.* keys) against the unchanged real-rotation forward channel,
# so train_mstep_mra marginalizes over R instead of estimating it: tests whether the image
# branch alone, warm-started fully pose-free, can still recover a good prior under real
# rotations. Compare against em_mra_rotation_few_hi_matched (same EM problem, ordinary
# pretrain+pose).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_few_hi_matched_zero_theta

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_few_hi_zero_theta.pt \
    --digit_classes 0 3 8 \
    --no_pose_head \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_few_hi_matched_zero_theta
