#!/bin/bash
#SBATCH --job-name=lrabl_const_1e-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/cm6627/scsi_lr_ablation/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/cm6627/scsi_lr_ablation/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# LR / schedule ablation arm `const_1e-4`.
# Constant 1e-4. Same mean LR over EM 1-12 as cos_h12 (1.0e-4 vs 1.1e-4) without the anneal
# tail -- separates 'less total LR' from 'ending at a small LR'.
#
# Loads the shared warm start from submit_lrabl_warmup.SBATCH (submit_all.sh chains this job
# after it with --dependency=afterok), then runs 24 EM steps. Everything but the schedule flags
# below is identical across arms. Full design + caveats: ./README.md
# --ckpt_dir keeps a rolling latest.pt (model + EMA + optimizer), overwritten every EM step.

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
    --num_scsi_steps 24 \
    --load_warmup_ckpt $CKPT/warmup_w20k.pt \
    --ckpt_dir $CKPT/const_1e-4 \
    --lr_schedule constant \
    --mstep_lr 1e-4 \
    --wandb_project scsi-cryoet-mnist3d-three-lr-schedule \
    --wandb_run_name const_1e-4
