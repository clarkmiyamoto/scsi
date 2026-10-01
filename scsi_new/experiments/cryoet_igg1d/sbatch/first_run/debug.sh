#!/bin/bash
# Short GPU check of experiments/cryoet_igg1d/main.py, run by hand on an interactive node:
#   salloc -p gpu --gres=gpu:1 -C "a100-80gb|h100|h200" --cpus-per-task=16 --mem=64G --time 1:00:00
#   bash /mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_igg1d/sbatch/first_run/debug.sh
# Checks the channel conventions, then runs 2 EM steps at the production batch sizes on 2000
# observations, and restarts with --num_scsi_steps 3 to check that --resume continues at EM 3.
# Each EM step prints its E-step / M-step / eval times and peak GPU memory; ../README.md turns
# them into wall-time estimates. Everything is also written to $RUNS/logs/debug_*.log.
#
# Matches em.SBATCH, which has no warm start: --warmup_n_steps_train 0 and an explicit --vol_gain
# (the GT-derived value `python data.py` prints; see ../../RECIPIES.md). For the warm start, see
# ../sched_warmup_mstep/debug.sh.
set -euo pipefail

CODE=/mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_igg1d
RUNS=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d
EXP=debug_$(date +%Y%m%d_%H%M%S)
VOL_GAIN=${VOL_GAIN:-0.0602}
mkdir -p "$RUNS/logs" "$RUNS/wandb"
exec > >(tee -a "$RUNS/logs/$EXP.log") 2>&1

module load python/3.13.2 uv
export WANDB_DIR=$RUNS/wandb
cd "$CODE"
PY="uv run --project /mnt/home/cmiyamoto/scsi python"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name())"

if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs api.wandb.ai ~/.netrc; then
    echo "No wandb login found, logging offline (run 'wandb login' once to log online)"
    export WANDB_MODE=offline
fi

$PY corruption.py

OPTS="--n_observations 2000 --warmup_n_steps_train 0 --vol_gain $VOL_GAIN
      --estep_num_samples 64 --mstep_n_steps_train 50
      --eval_n 8 --viz_n_pool 8 --viz_n_display 4
      --ckpt_dir $RUNS/checkpoints/$EXP --resume
      --wandb_project scsi-cryoet-igg1d-debug --wandb_run_name $EXP"

$PY main.py $OPTS --num_scsi_steps 2
echo "=== restarting to check --resume continues at EM 3 ==="
$PY main.py $OPTS --num_scsi_steps 3
echo "=== debug run finished OK: $RUNS/checkpoints/$EXP ==="
