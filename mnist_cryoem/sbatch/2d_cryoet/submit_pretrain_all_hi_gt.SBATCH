#!/bin/bash
#SBATCH --job-name=pretrain_all_hi_gt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix v2, GT-oracle ceiling (NOT a classical-recon cell -- --warmstart_target gt,
# the old default behavior). Same GT pool as all_hi (all 10 classes, 516 img/class), but x1 is
# the literal canonical GT digit instead of a classical reconstruction of noisy observations.
# Kept as the ONE gt cell in an otherwise all-classical_recon matrix specifically to answer:
# how much does classical_recon warmstart cost you relative to a warmstart that gets to see
# clean data directly? --time 02:00:00: gt mode has no backproject() pool-build stage at all
# (theta is drawn fresh every step, x1 never touches corruption.py), so this is as cheap as the
# old all_hi.pt used to be.
# Compare against: all_hi (same coverage/pool size, classical_recon instead of gt) --
# downstream comparison is submit_em_all_hi_gt_matched.SBATCH vs submit_em_all_hi_matched.SBATCH.
# NOTE: --digit_classes omitted -> defaults to all 10 classes.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_all_hi_gt

uv run python pretrain.py \
    --warmstart_target gt \
    --n_pretrain_images_per_class 516 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_all_hi_gt.pt
