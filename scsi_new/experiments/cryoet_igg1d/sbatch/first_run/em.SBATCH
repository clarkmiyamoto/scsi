#!/bin/bash
# First IgG-1D SCSI run: EM from a randomly initialised network (no warm start, the baseline
# for ../sched_warmup_mstep), 12 EM steps of 2000 posterior samples and 5000 M-step steps at 64 px.
# To start from a warm start instead, replace START with
#   START="--load_warmup_ckpt $RUNS/checkpoints/<warmup>.pt"
# (the checkpoint carries vol_gain) and give the run a new name / ckpt dir.
#
#   bash submit_all.sh          # this file plus afterany continuations (see README.md)
#   sbatch em.SBATCH            # one 2-day job; mkdir -p the --output directory first
#
# --resume: a later job continues from $CKPT/latest.pt (same wandb run), or exits at once if the
# run is finished. Design, cost estimates and caveats: ./README.md
#SBATCH --job-name=igg1d-em-scratch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80gb|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/logs/%x_%j.out
set -euo pipefail

REPO=${REPO:-/mnt/home/cmiyamoto/scsi}   # checkout whose pyproject.toml / uv env is used
CODE=$REPO                                    # directory the job runs from (python -m needs the repo root)
RUNS=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d
NAME=em_scratch
CKPT=$RUNS/checkpoints/$NAME
VOL_GAIN=${VOL_GAIN:-0.0602}
START="--warmup_n_steps_train 0 --vol_gain $VOL_GAIN"

module load python/3.13.2 uv
export WANDB_DIR=$RUNS/wandb
mkdir -p "$WANDB_DIR"
cd "$CODE"

echo "host $(hostname)  job $SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv run --project "$REPO" python -m scsi_new.experiments.cryoet_igg1d.main \
    $START \
    --estep_num_samples 2000 \
    --mstep_n_steps_train 5000 \
    --num_scsi_steps 12 \
    --lr_schedule cosine \
    --ckpt_dir "$CKPT" \
    --resume \
    --wandb_project scsi-cryoet-igg1d \
    --wandb_run_name "$NAME" \
    "$@"
