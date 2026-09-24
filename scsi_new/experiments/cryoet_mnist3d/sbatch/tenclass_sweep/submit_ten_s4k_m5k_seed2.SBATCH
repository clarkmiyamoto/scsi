#!/bin/bash
#SBATCH --job-name=ten_s4k_m5k_seed2
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

# Ten-class sweep arm `s4k_m5k_seed2`.
# s4k_m5k with a different EM-phase RNG stream (--em_seed 2): the noise floor. Differences
# between arms smaller than s4k_m5k vs this one are noise.
#
# Loads the shared warm start from submit_ten_warmup_w40k.SBATCH, then runs 24 EM steps.
# submit_all.sh queues this file twice: after the warmup (--dependency=afterok), and again after
# that first job ends (--dependency=afterany). With --resume the second copy continues from
# $CKPT/s4k_m5k_seed2/latest.pt after a walltime kill, or exits at once if the arm already finished.
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
    --estep_num_samples 4000 \
    --mstep_n_steps_train 5000 \
    --num_scsi_steps 24 \
    --lr_schedule cosine \
    --lr_horizon_scsi_steps 12 \
    --em_seed 2 \
    --load_warmup_ckpt $CKPT/warmup_w40k.pt \
    --ckpt_dir $CKPT/s4k_m5k_seed2 \
    --resume \
    --wandb_project scsi-cryoet-mnist3d-ten-sweep \
    --wandb_run_name s4k_m5k_seed2
