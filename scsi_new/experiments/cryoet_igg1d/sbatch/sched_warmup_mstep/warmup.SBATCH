#!/bin/bash
# Stage 1 of the schedule / warmup / M-step ablation: ONE shared warm start of W steps
#   sbatch warmup.SBATCH 30000        # submit_all.sh runs both W = 30000 and W = 40000
# Trains on ResampledPairs of pseudoinverse.py's reconstructions and writes
# $CKPT_ROOT/warmup_w<W/1000>k.pt (model, EMA, AdamW moments, global step, RNG, vol_gain). Every
# arm with this W loads it, so they start EM from identical weights and differ only in EM-phase
# flags. --num_scsi_steps 0: warmup + em-0 panels / eval only.
#
# The warmup LR is the global cosine of the base recipe (--lr_schedule cosine, 12 EM steps of
# 5000 M-step steps): 3e-4 -> 2.3e-4 over 30k steps, -> 2.0e-4 over 40k. --lr_horizon_scsi_steps
# 12 matters here: with --num_scsi_steps 0 the cosine would otherwise anneal fully over warmup.
# Design and caveats: ./README.md
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80gb|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/logs/%x_%j.out
set -euo pipefail

W=${1:?usage: sbatch warmup.SBATCH <warmup steps>}
# CODE: the snapshot submit_all.sh made; the live checkout when run by hand.
REPO=${REPO:-/mnt/home/cmiyamoto/scsi}
CODE=${CODE:-$REPO}
RUNS=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d
CKPT_ROOT=$RUNS/checkpoints/sched_warmup_mstep
NAME=warmup_w$((W / 1000))k

module load python/3.13.2 uv
export WANDB_DIR=$RUNS/wandb
mkdir -p "$WANDB_DIR" "$CKPT_ROOT"
cd "$CODE"

echo "host $(hostname)  job $SLURM_JOB_ID  code $CODE  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv run --project "$REPO" python -m scsi_new.experiments.cryoet_igg1d.main \
    --warmup_n_steps_train "$W" \
    --mstep_n_steps_train 5000 \
    --lr_schedule cosine \
    --lr_horizon_scsi_steps 12 \
    --num_scsi_steps 0 \
    --save_warmup_ckpt "$CKPT_ROOT/$NAME.pt" \
    --wandb_project scsi-cryoet-igg1d-ablation \
    --wandb_run_name "$NAME"
