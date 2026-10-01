#!/bin/bash
# Submit em.SBATCH plus N_JOBS - 1 continuations, each --dependency=afterany on the one before.
# The continuations run the same file, whose --resume picks up $CKPT/latest.pt after a walltime
# kill, or exits in a few minutes if the run already finished. Extra args go to main.py.
#
#   bash submit_all.sh
#   N_JOBS=3 bash submit_all.sh
set -euo pipefail
cd "$(dirname "$0")"
N_JOBS=${N_JOBS:-2}   # 2-day jobs; see README.md for the wall-time estimate
mkdir -p /mnt/ceph/users/cmiyamoto/scsi_runs/igg1d/logs

prev=$(sbatch --parsable em.SBATCH "$@")
ids=$prev
for _ in $(seq 2 "$N_JOBS"); do
    prev=$(sbatch --parsable --dependency=afterany:"$prev" em.SBATCH "$@")
    ids="$ids -> $prev"
done
echo "em.SBATCH: $ids"
