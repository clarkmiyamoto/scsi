#!/bin/bash
#SBATCH --job-name=pretrain_mra_rotation_few_hi
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

# Pretrain matrix for rotational-MRA channel, axis: class coverage x pool size/class.
# few_hi: digits {0, 3, 8} (visually distinct strokes), 516 images/class -- partial coverage,
# well-fed. Compare against: single_hi (1 of these 3 classes only), all_hi (all 10 classes).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_mra_rotation_few_hi

uv run python pretrain_mra_rotation.py \
    --digit_classes 0 3 8 \
    --n_pretrain_images_per_class 516 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_mra_rotation_checkpoints/pretrain_mra_rotation_few_hi.pt
