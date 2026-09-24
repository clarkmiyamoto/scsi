#!/bin/bash
#SBATCH --job-name=ten_warmup_w20k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/cm6627/scsi_tenclass/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/cm6627/scsi_tenclass/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Ten-class sweep, warmup-length side job (20k steps, the digit-3 value). Nothing chains on
# it: compare its em-0 eval/ against warmup_w40k / warmup_w80k to see how warmup length should
# scale with 10 classes. Saves its checkpoint so a follow-up arm can start from it.
# Design: ./README.md

CODE=/scratch/cm6627/scsi_tenclass/scsi_new/experiments/cryoet_mnist3d
CKPT=/scratch/cm6627/scsi_tenclass/checkpoints
cd $CODE

uv run --project /scratch/cm6627/scsi python main.py \
    --digit_classes 0 1 2 3 4 5 6 7 8 9 \
    --n_images_per_class 5000 \
    --warmup_batch_size 16 \
    --mstep_batch_size 16 \
    --estep_batch_size 32 \
    --eval_n 100 \
    --viz_n_pool 30 \
    --viz_n_display 10 \
    --warmup_n_steps_train 20000 \
    --mstep_n_steps_train 5000 \
    --lr_horizon_scsi_steps 12 \
    --num_scsi_steps 0 \
    --save_warmup_ckpt $CKPT/warmup_w20k.pt \
    --wandb_project scsi-cryoet-mnist3d-ten-sweep \
    --wandb_run_name warmup_w20k
