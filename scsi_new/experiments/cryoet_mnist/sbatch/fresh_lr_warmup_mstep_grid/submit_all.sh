#!/bin/bash -l
# Submit every cell of this grid from a Rusty login node:
#   ./submit_all.sh
#
# Downloads MNIST into cryoet_mnist/data/ (and syncs the uv env) once up front, so the 27 jobs
# don't race each other on data.py's first-use download when several start at the same moment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load python/3.13.2 uv
cd "$HERE/../.."
uv run python -c "from torchvision import datasets; datasets.MNIST('./data', train=True, download=True)"

for f in "$HERE"/submit_fresh_*.SBATCH; do
    sbatch "$f"
done
