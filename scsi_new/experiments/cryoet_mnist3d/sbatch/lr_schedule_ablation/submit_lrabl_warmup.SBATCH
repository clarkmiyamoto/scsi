#!/bin/bash
#SBATCH --job-name=lrabl_warmup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/cm6627/scsi_lr_ablation/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/cm6627/scsi_lr_ablation/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# LR / schedule ablation, stage 1: the ONE shared warm start. Every arm
# (submit_lrabl_<arm>.SBATCH) loads the checkpoint this writes, so all arms start EM from
# identical weights, optimizer moments and RNG state and differ only in their EM-phase schedule.
# Full design + caveats: ./README.md
#
# --num_scsi_steps 0: warmup + em-0 panels/eval only. --lr_horizon_scsi_steps 12 trains the
# warmup on the original 12-step recipe's cosine (3e-4 -> 2.58e-4 over the 20k steps), so
# the cos_h12 arm continues that recipe unchanged.
# Everything that fixes the observations / model must match the arms' flags exactly;
# --load_warmup_ckpt refuses a mismatch.

CODE=/scratch/cm6627/scsi_lr_ablation/scsi_new/experiments/cryoet_mnist3d
CKPT=/scratch/cm6627/scsi_lr_ablation/checkpoints
cd $CODE

uv run --project /scratch/cm6627/scsi python main.py \
    --digit_classes 3 \
    --n_images_per_class 23000 \
    --warmup_batch_size 16 \
    --mstep_batch_size 16 \
    --estep_batch_size 32 \
    --warmup_n_steps_train 20000 \
    --mstep_n_steps_train 5000 \
    --num_scsi_steps 0 \
    --lr_horizon_scsi_steps 12 \
    --save_warmup_ckpt $CKPT/warmup_w20k.pt \
    --wandb_project scsi-cryoet-mnist3d-three-lr-schedule \
    --wandb_run_name warmup_w20k
