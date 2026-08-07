#!/bin/bash
#SBATCH --job-name=em_few_hi_cpo1_matched
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

# Baseline matrix, Group A (matched), reconstruction-richness ablation: pretrained on
# pretrain_few_hi_cpo1.pt (corruptions_per_object=1, one reconstruction per object) instead of
# pretrain_few_hi.pt (corruptions_per_object=4, four independently-oriented reconstructions
# per object). Same EM problem ({0,3,8}) as em_few_hi_matched. Tests whether the extra
# reconstruction diversity corruptions_per_object=4 buys at pretrain time actually survives
# into better downstream EM performance, or whether the EM loop's own cross-acquisition
# pooling already recovers whatever a richer pretrain pool would have bought.
# Compare against: em_few_hi_matched (same EM problem, corruptions_per_object=4 pretrain).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_few_hi_cpo1_matched

uv run python main.py \
    --init_ckpt mnist_cryoem_checkpoints/pretrain_few_hi_cpo1.pt \
    --digit_classes 0 3 8 \
    --ckpt_dir mnist_cryoem_checkpoints/em_few_hi_cpo1_matched
