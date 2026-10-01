#!/bin/bash -l
#SBATCH --job-name=fresh3d_lr3e-4_w25k_m25k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=rocky9&(h100|a100-80gb)
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_mnist3d/logs/slurm-%x-%j.out
#SBATCH --error=/mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_mnist3d/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D fresh-student lr x (warmup = M-step length) grid, digit class 3 only (23,000 images),
# --student_init fresh. This cell: mstep_lr=3e-4, warmup_n_steps_train = mstep_n_steps_train
# = 25000, num_scsi_steps=12. warmup_lr is held at 3e-4 in every cell.
# --time is ~1.5x an estimate (~62h on an L40S-class GPU), not a measurement. --ckpt_dir +
# --resume: submit_all.sh chains a continuation of this same file that picks up from latest.pt
# after a walltime kill, or exits at once if the run already finished.
# Full grid + caveats: README.md in this directory.

module load python/3.13.2 uv

cd /mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_mnist3d

uv run main.py \
    --student_init fresh \
    --digit_classes 3 \
    --n_images_per_class 23000 \
    --warmup_batch_size 16 \
    --mstep_batch_size 16 \
    --estep_batch_size 32 \
    --warmup_lr 3e-4 \
    --warmup_n_steps_train 25000 \
    --mstep_lr 3e-4 \
    --mstep_n_steps_train 25000 \
    --num_scsi_steps 12 \
    --ckpt_dir /mnt/ceph/users/cmiyamoto/scsi_checkpoints/cryoet_mnist3d/fresh_lr_tied_steps_grid/lr3e-4_w25k_m25k \
    --resume \
    --wandb_project scsi-cryoet-mnist3d-three-fresh-lr-steps \
    --wandb_run_name lr3e-4_w25k_m25k
