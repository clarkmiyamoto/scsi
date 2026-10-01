#!/bin/bash -l
# Submit cells of this grid from a Rusty login node:
#   ./submit_all.sh                                  # all 12 cells
#   ./submit_all.sh submit_fresh3d_lr3e-4_*.SBATCH   # a subset, e.g. one LR row
#
# Each cell is queued N_JOBS times (default 2): the first job, then continuations of the same
# file, each --dependency=afterany on the one before. --resume makes a continuation pick up from
# the cell's latest.pt after a walltime kill, or exit within a minute if the cell already finished.
#
# Syncs the uv env and checks for EMNIST once up front, so 12 jobs that start together don't race on
# uv's env sync or data.py's first-use download.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_JOBS=${N_JOBS:-2}

module load python/3.13.2 uv
cd "$HERE/../.."
mkdir -p logs
uv run python -c "from torchvision import datasets; datasets.EMNIST('./data', split='digits', train=True, download=True)"

cd "$HERE"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=(submit_fresh3d_*.SBATCH)
for f in "${files[@]}"; do
    prev=$(sbatch --parsable "$f")
    ids=$prev
    for _ in $(seq 2 "$N_JOBS"); do
        prev=$(sbatch --parsable --dependency=afterany:"$prev" "$f")
        ids="$ids -> $prev"
    done
    echo "$f: $ids"
done
