#!/bin/bash
# GPU check of this ablation's pipeline before submit_all.sh, run by hand on an interactive node:
#   salloc -p gpu --gres=gpu:1 -C "a100-80gb|h100|h200" --cpus-per-task=16 --mem=64G --time 1:00:00
#   bash /mnt/home/cmiyamoto/scsi/scsi_new/experiments/cryoet_igg1d/sbatch/sched_warmup_mstep/debug.sh
# At the production model and batch sizes, on 2000 observations: a 300-step pseudoinverse warm
# start saved as warmup.SBATCH does (prints its it/s), then each lr_schedule loads it for one EM
# step as em.SBATCH does, then the cosine arm resumes to EM 2. Read off the `warmup: ... it/s`
# and `[em k/...] estep ... mstep ...` lines; README.md turns them into wall times. Everything is
# also written to $RUNS/logs/debug_sched_*.log.
set -euo pipefail

REPO=${REPO:-/mnt/home/cmiyamoto/scsi}   # checkout whose pyproject.toml / uv env is used
CODE=$REPO                                    # directory the job runs from (python -m needs the repo root)
RUNS=/mnt/ceph/users/cmiyamoto/scsi_runs/igg1d
EXP=debug_sched_$(date +%Y%m%d_%H%M%S)
CKPT=$RUNS/checkpoints/$EXP
mkdir -p "$RUNS/logs" "$RUNS/wandb" "$CKPT"
exec > >(tee -a "$RUNS/logs/$EXP.log") 2>&1

module load python/3.13.2 uv
export WANDB_DIR=$RUNS/wandb
cd "$CODE"
PY="uv run --project $REPO python"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name())"

if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs api.wandb.ai ~/.netrc; then
    echo "No wandb login found, logging offline (run 'wandb login' once to log online)"
    export WANDB_MODE=offline
fi

COMMON="--n_observations 2000 --warmup_n_steps_train 300 --lr_horizon_scsi_steps 12
        --eval_n 8 --viz_n_pool 8 --viz_n_display 4 --wandb_project scsi-cryoet-igg1d-debug"

echo "=== warmup only (warmup.SBATCH) ==="
$PY -m scsi_new.experiments.cryoet_igg1d.main $COMMON --mstep_n_steps_train 5000 --lr_schedule cosine --num_scsi_steps 0 \
    --save_warmup_ckpt "$CKPT/warmup.pt" --wandb_run_name "${EXP}_warmup"

EM="$COMMON --load_warmup_ckpt $CKPT/warmup.pt --estep_num_samples 64 --mstep_n_steps_train 50 --resume"
for SCHED in cosine constant cosine_per_mstep; do
    echo "=== one EM step, --lr_schedule $SCHED (em.SBATCH) ==="
    $PY -m scsi_new.experiments.cryoet_igg1d.main $EM --lr_schedule $SCHED --num_scsi_steps 1 --ckpt_dir "$CKPT/$SCHED" \
        --wandb_run_name "${EXP}_$SCHED"
done
echo "=== restarting the cosine arm: should resume at EM step 1 and run EM 2 ==="
$PY -m scsi_new.experiments.cryoet_igg1d.main $EM --lr_schedule cosine --num_scsi_steps 2 --ckpt_dir "$CKPT/cosine" \
    --wandb_run_name "${EXP}_cosine"
echo "=== debug run finished OK: $CKPT ==="
