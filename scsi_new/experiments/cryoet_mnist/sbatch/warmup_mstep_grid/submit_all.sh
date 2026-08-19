#!/bin/bash
# Submits all 20 cells of the warmup x mstep grid. Run this from the cluster (from anywhere --
# it cd's into place itself).
#
# Why this exists instead of just `sbatch submit_three_*.SBATCH`: data.py downloads MNIST to a
# cwd-relative ./data on first use. If 20 jobs land on nodes at the same time, they can race on
# that same download/extract and corrupt each other's copy. This script forces the download to
# happen once, synchronously, before any SBATCH job is queued.
set -euo pipefail

PROJECT_DIR="/scratch/cm6627/scsi/scsi_new/experiments/cryoet_mnist"
GRID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Pre-downloading/caching MNIST (digit 3) into ${PROJECT_DIR}/data ..."
(cd "$PROJECT_DIR" && uv run python data.py --digit_classes 3 --n_images_per_class 2 --out_dir /tmp/_grid_predownload_viz)

echo "Submitting 20 grid jobs..."
for f in "$GRID_DIR"/submit_three_w*.SBATCH; do
    sbatch "$f"
done
