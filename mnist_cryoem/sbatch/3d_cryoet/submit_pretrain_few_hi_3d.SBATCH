#!/bin/bash
#SBATCH --job-name=pretrain_few_hi_3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET counterpart of submit_pretrain_few_hi.SBATCH -- same pretrain matrix cell (digits
# {0, 3, 8}, 516 volumes/class -- partial coverage, well-fed), via pretrain_3d.py.
# Compare against: single_hi_3d (1 of these 3 classes only), all_hi_3d (all 10 classes).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_few_hi_3d

uv run python pretrain_3d.py \
    --digit_classes 0 3 8 \
    --n_pretrain_images_per_class 516 \
    --n_steps 18000 \
    --batch_size 8 \
    --out_ckpt mnist_cryoet_checkpoints/pretrain_few_hi_3d.pt
