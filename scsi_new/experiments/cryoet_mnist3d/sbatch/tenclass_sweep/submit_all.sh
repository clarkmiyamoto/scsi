#!/bin/bash
# Submit the ten-class sweep:
#   - the shared w40k warmup, with every arm chained after it (--dependency=afterok);
#   - N_JOBS - 1 continuations per arm, each --dependency=afterany on the one before. They run
#     the same SBATCH file, whose --resume continues from the arm's latest.pt after a walltime
#     kill, or exits in about a minute if the arm already finished;
#   - the w20k / w80k warmup-only side jobs (em-0 eval only), unchained.
# If the w40k warmup fails, its arms stay pending with reason DependencyNeverSatisfied.
# scancel them AND their continuations: once an arm's first job is gone, afterany lets the
# continuation start, and it will fail on the missing warmup_w40k.pt.
set -euo pipefail
cd "$(dirname "$0")"
N_JOBS=${N_JOBS:-2}   # 48h jobs per arm; 2 covers the slowest arm (~68h on an L40S)
mkdir -p /scratch/cm6627/scsi_tenclass/logs /scratch/cm6627/scsi_tenclass/checkpoints

WARMUP=$(sbatch --parsable submit_ten_warmup_w40k.SBATCH)
echo "submit_ten_warmup_w40k.SBATCH: $WARMUP"
for f in submit_ten_warmup_w20k.SBATCH submit_ten_warmup_w80k.SBATCH; do
    echo "$f: $(sbatch --parsable "$f")"
done

for f in submit_ten_*.SBATCH; do
    case "$f" in submit_ten_warmup_*) continue ;; esac
    prev=$(sbatch --parsable --dependency=afterok:"$WARMUP" "$f")
    ids=$prev
    for _ in $(seq 2 "$N_JOBS"); do
        prev=$(sbatch --parsable --dependency=afterany:"$prev" "$f")
        ids="$ids -> $prev"
    done
    echo "$f: $ids"
done
