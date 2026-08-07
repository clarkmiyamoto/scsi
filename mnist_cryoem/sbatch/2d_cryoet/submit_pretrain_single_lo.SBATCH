#!/bin/bash
#SBATCH --job-name=pretrain_single_lo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# Pretrain matrix v2 (classical-recon warm start), axis: class coverage x pool size/class.
# Replaces the old GT-supervised matrix (submit_pretrain_*.SBATCH, --warmstart_target gt) and
# the old submit_pretrain_*_unet.SBATCH files, which are DELETED -- they passed --image_arch,
# a flag pretrain.py has never had (would die on argparse; no EM script referenced their
# checkpoints anyway). See mnist_cryoem/CLAUDE.md's "Classical-recon warm start" section: x1 is
# now a filtered-backprojection reconstruction of the object's OWN noisy tilt-series
# observations, never the literal clean digit -- this checkpoint has to learn to denoise/sharpen
# a real reconstruction, not just memorize GT. --n_tilts/--tilt_increment_deg/--noise_std are
# pinned to main.py's real EM geometry so the pretraining pseudo-targets have exactly the
# fidelity the EM loop will actually see (never sharper). --corruptions_per_object 4 gives each
# object 4 independently-oriented reconstructions (tilt_series_difficulty.py found
# per-acquisition error is variance-, not bias-dominated -- different wedges are complementary,
# not redundant). n_steps is fixed at 20000 across EVERY cell in this matrix (the old matrix
# confounded pool size with training length: lo=10000, hi=20000 steps) so pool size/coverage is
# the only thing that varies cell to cell.
#
# single_lo: digit {3} ONLY, 32 images/class -- narrowest prior, most starved.
# Compare against: single_hi (same class, 516 img/class), all_lo (same pool size, all 10
# classes), few_hi_cpo1 (reconstruction-richness ablation), all_hi_gt (GT-oracle ceiling).

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=pretrain_single_lo

uv run python pretrain.py \
    --warmstart_target classical_recon \
    --digit_classes 3 \
    --n_pretrain_images_per_class 32 \
    --corruptions_per_object 4 \
    --n_tilts 16 \
    --tilt_increment_deg 7.5 \
    --noise_std 3.0 \
    --n_steps 20000 \
    --batch_size 256 \
    --out_ckpt mnist_cryoem_checkpoints/pretrain_single_lo.pt
