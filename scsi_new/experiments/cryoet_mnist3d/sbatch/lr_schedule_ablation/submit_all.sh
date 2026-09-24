#!/bin/bash
# Submit the LR / schedule ablation: the shared warmup job, then every arm chained after it with
# --dependency=afterok, so the arms queue now and start only once warmup_w20k.pt exists. If the
# warmup fails, the arms stay pending with reason DependencyNeverSatisfied -- scancel them.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /scratch/cm6627/scsi_lr_ablation/logs /scratch/cm6627/scsi_lr_ablation/checkpoints

WARMUP=$(sbatch --parsable submit_lrabl_warmup.SBATCH)
echo "submit_lrabl_warmup.SBATCH: $WARMUP"
for f in submit_lrabl_*.SBATCH; do
    [ "$f" = submit_lrabl_warmup.SBATCH ] && continue
    echo "$f: $(sbatch --parsable --dependency=afterok:"$WARMUP" "$f")"
done
