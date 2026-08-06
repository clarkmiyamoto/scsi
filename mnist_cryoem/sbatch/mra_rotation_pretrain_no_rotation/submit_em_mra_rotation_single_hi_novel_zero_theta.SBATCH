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

# Zero-theta-pretrain warm-start ablation (rotational-MRA), Group B (novel-class
# generalization): pretrained narrower than the EM problem. Same as
# em_mra_rotation_single_hi_novel, but warm-started from
# pretrain_mra_rotation_single_hi_zero_theta.pt -- pretrained with theta fixed to 0 (F_MRA
# degenerates to y = x + W, image branch only, pose head never trained on nonzero rotation,
# and the model never saw digit 7 at all). The EM stage itself is unchanged (real sampled
# rotation, digits {3,7}): tests whether SCSI/EM can both discover a novel class AND
# bootstrap the pose branch from scratch, starting from this image-only warm start. Compare
# against em_mra_rotation_single_hi_novel (same EM problem, ordinary pretrain) and
# em_mra_rotation_single_hi_matched_zero_theta (same pretrain, no novel class).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_mra_rotation_single_hi_novel_zero_theta

uv run python main_mra_rotation.py \
    --init_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_single_hi_zero_theta.pt \
    --digit_classes 3 7 \
    --ckpt_dir mnist_mra_rotation_checkpoints/em_single_hi_novel_zero_theta
