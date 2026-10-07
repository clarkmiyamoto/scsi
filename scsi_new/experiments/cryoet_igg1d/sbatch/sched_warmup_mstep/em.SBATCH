#!/bin/bash
# Stage 2 of the schedule / warmup / M-step ablation: one EM arm
#   sbatch em.SBATCH <lr_schedule> <warmup steps> <M-step steps> [main.py args]
#   sbatch em.SBATCH cosine_per_mstep 30000 10000
#   SUFFIX=_seed2 sbatch em.SBATCH cosine 30000 5000 --em_seed 2    # a replicate, own name
# Loads warmup.SBATCH's checkpoint for that warmup length (vol_gain comes with it), then runs 12
# EM steps of 2000 posterior samples. Only --lr_schedule and --mstep_n_steps_train differ
# between arms that share a warm start.
#
# --resume: a later job with the same arguments continues from $CKPT_ROOT/<arm>/latest.pt (same
# wandb run), or exits at once if the arm is finished; submit_all.sh chains one with
# --dependency=afterany. --lr_horizon_scsi_steps 12 keeps the cosine's end at EM 12 even if
# --num_scsi_steps is later raised to extend an arm. Design and caveats: ./README.md
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80gb|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/logs/%x_%j.out
set -euo pipefail

USAGE="usage: sbatch em.SBATCH <cosine|constant|cosine_per_mstep> <warmup steps> <M-step steps>"
SCHED=${1:?$USAGE}
W=${2:?$USAGE}
M=${3:?$USAGE}
case "$SCHED" in
    cosine) TAG=cos ;;
    constant) TAG=const ;;
    cosine_per_mstep) TAG=cospm ;;
    *) echo "$USAGE" >&2; exit 2 ;;
esac
# CODE: the snapshot submit_all.sh made; the live checkout when run by hand. A continuation of
# an arm should run the same code as its first job: pass the same CODE.
REPO=${REPO:-/mnt/home/cmiyamoto/scsi}
CODE=${CODE:-$REPO}
RUNS=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d
CKPT_ROOT=$RUNS/checkpoints/sched_warmup_mstep
WARMUP=$CKPT_ROOT/warmup_w$((W / 1000))k.pt
NAME=w$((W / 1000))k_m$((M / 1000))k_$TAG${SUFFIX:-}

module load python/3.13.2 uv
export WANDB_DIR=$RUNS/wandb
mkdir -p "$WANDB_DIR"
cd "$CODE"

echo "host $(hostname)  job $SLURM_JOB_ID  arm $NAME  code $CODE  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv run --project "$REPO" python -m scsi_new.experiments.cryoet_igg1d.main \
    --warmup_n_steps_train "$W" \
    --load_warmup_ckpt "$WARMUP" \
    --mstep_n_steps_train "$M" \
    --lr_schedule "$SCHED" \
    --lr_horizon_scsi_steps 12 \
    --num_scsi_steps 12 \
    --ckpt_dir "$CKPT_ROOT/$NAME" \
    --resume \
    --wandb_project scsi-cryoet-igg1d-ablation \
    --wandb_run_name "$NAME" \
    "${@:4}"
