#!/bin/bash
# Submit the schedule / warmup / M-step ablation:
#   - a snapshot of the code, so editing the checkout does not change queued or running jobs;
#   - one warmup.SBATCH per warmup length;
#   - every arm (lr_schedule x M-step steps) chained after its warmup with --dependency=afterok,
#     plus N_JOBS - 1 continuations each, --dependency=afterany on the one before.
# If a warmup fails, its arms stay pending (DependencyNeverSatisfied): scancel them AND their
# continuations, which afterany would otherwise start once the first job is gone.
#
#   bash submit_all.sh                         # the full 2 x 3 x 2 grid
#   WARMUPS="30000" SCHEDS="cosine" bash submit_all.sh    # a subset
set -euo pipefail
cd "$(dirname "$0")"
WARMUPS=${WARMUPS:-"30000 40000"}
SCHEDS=${SCHEDS:-"cosine constant cosine_per_mstep"}
MSTEPS=${MSTEPS:-"5000 10000"}
N_JOBS=${N_JOBS:-2}   # 3-day jobs; see README.md for the wall-time estimate

SRC=/mnt/home/cmiyamoto/scsi/scsi_new
SNAP=/mnt/home/cmiyamoto/scsi_snapshots/igg1d_sched_warmup_mstep_$(date +%Y%m%d_%H%M%S)/scsi_new
mkdir -p "$SNAP/experiments" /mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/logs
rsync -a --exclude __pycache__ "$SRC"/*.py "$SNAP/"
rsync -a --exclude __pycache__ --exclude wandb "$SRC/experiments/cryoet_igg1d" "$SNAP/experiments/"
export CODE=$SNAP/experiments/cryoet_igg1d
echo "code snapshot: $CODE"

for W in $WARMUPS; do
    wid=$(sbatch --parsable --job-name="igg1d-warmup-w$((W / 1000))k" warmup.SBATCH "$W")
    echo "warmup w$((W / 1000))k: $wid"
    for SCHED in $SCHEDS; do
        for M in $MSTEPS; do
            name="igg1d-$SCHED-w$((W / 1000))k-m$((M / 1000))k"
            prev=$(sbatch --parsable --job-name="$name" --dependency=afterok:"$wid" \
                   em.SBATCH "$SCHED" "$W" "$M")
            ids=$prev
            for _ in $(seq 2 "$N_JOBS"); do
                prev=$(sbatch --parsable --job-name="$name" --dependency=afterany:"$prev" \
                       em.SBATCH "$SCHED" "$W" "$M")
                ids="$ids -> $prev"
            done
            echo "  $name: $ids"
        done
    done
done
