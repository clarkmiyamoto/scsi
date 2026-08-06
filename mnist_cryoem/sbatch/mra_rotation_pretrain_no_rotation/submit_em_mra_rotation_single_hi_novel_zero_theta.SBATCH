#!/bin/bash
#SBATCH --job-name=em_mra_rotation_single_hi_novel_zero_theta
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

# Zero-theta + no-pose-head pretrain warm-start ablation (rotational-MRA), Group B
# (novel-class generalization): pretrained narrower than the EM problem. Same as
# em_mra_rotation_single_hi_novel, but warm-started from
# pretrain_mra_rotation_single_hi_zero_theta.pt -- pretrained with theta fixed to 0 (F_MRA
# degenerates to y = x + W, no rotation), --no_pose_head (no pose branch at all, pure
# image-only Theta^(0)), and the model never saw digit 7 at all. The EM stage also runs
# --no_pose_head (required to load a checkpoint with no pose_branch.* keys) against the
# unchanged real-rotation forward channel (digits {3,7}), so train_mstep_mra marginalizes
# over R throughout: tests whether SCSI/EM can discover a novel class from a fully pose-free,
# image-only prior. Compare against em_mra_rotation_single_hi_novel (same EM problem,
# ordinary pretrain+pose) and em_mra_rotation_single_hi_matched_zero_theta (same pretrain, no
# novel class).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_single_hi_novel_zero_theta

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_single_hi_zero_theta.pt \
    --digit_classes 3 7 \
    --no_pose_head \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_single_hi_novel_zero_theta
