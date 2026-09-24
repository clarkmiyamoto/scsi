#!/bin/bash
#SBATCH --job-name=ten_s2k_m5k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/cm6627/scsi_tenclass/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/cm6627/scsi_tenclass/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Ten-class sweep arm `s2k_m5k`.
# The digit-3 recipe's E-step / M-step sizes, unchanged (2000 samples, 5000 steps -> ~200
# posterior samples per class per EM step). Does the one-class recipe transfer as is?
#
# Loads the shared warm start from submit_ten_warmup_w40k.SBATCH, then runs 24 EM steps.
# submit_all.sh queues this file twice: after the warmup (--dependency=afterok), and again after
# that first job ends (--dependency=afterany). With --resume the second copy continues from
# $CKPT/s2k_m5k/latest.pt after a walltime kill, or exits at once if the arm already finished.
# Design, arm table and caveats: ./README.md

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
    --warmup_n_steps_train 40000 \
    --estep_num_samples 2000 \
    --mstep_n_steps_train 5000 \
    --num_scsi_steps 24 \
    --lr_schedule cosine \
    --lr_horizon_scsi_steps 12 \
    --load_warmup_ckpt $CKPT/warmup_w40k.pt \
    --ckpt_dir $CKPT/s2k_m5k \
    --resume \
    --wandb_project scsi-cryoet-mnist3d-ten-sweep \
    --wandb_run_name s2k_m5k
