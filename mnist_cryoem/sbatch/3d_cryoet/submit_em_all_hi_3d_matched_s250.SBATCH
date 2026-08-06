#!/bin/bash
#SBATCH --job-name=em_all_hi_3d_matched_s250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
#SBATCH --output=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.out
#SBATCH --error=/scratch/cm6627/scsi/mnist_cryoem/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

# 3D/CryoET pretraining-amount matrix, Group A (matched): pretrain coverage == EM problem
# coverage. all_hi_3d_matched: pretrained on all 10 classes (516 vol/class), EM problem also
# all 10. Ceiling run of the 3D matrix -- compare against all_lo_3d_matched (same coverage,
# starved pretrain).
# NOTE: --digit_classes omitted -> defaults to all 10 classes.
#
# --steps_per_em=250 copy of submit_em_all_hi_3d_matched.SBATCH (--n_em_steps 200/
# --steps_per_em 100 there). n_em_steps matches main_3d.py's algorithmic default (200);
# steps_per_em here (250) exceeds its default (200) since E-step cost is pool_size *
# sample_steps full 3D-UNet forward passes per EM iteration, where pool_size = min(N_obs,
# steps_per_em * batch_size). The original 100/40 setting measured ~2.5h in practice
# (squeue/sacct); scaling that observed throughput linearly with the K*steps_per_em product
# (4,000 -> 50,000, 12.5x) targets ~31h, hence --time=32:00:00. The 200/100 sibling
# (product 20,000, 5x) targets ~12.5h, --time=13:00:00 there. Applied identically to all
# five *_3d_matched_s250 runs so the comparison stays apples-to-apples.

cd /scratch/cm6627/scsi/mnist_cryoem

export WANDB_NAME=em_all_hi_3d_matched_s250

uv run python main_3d.py \
    --init_ckpt mnist_cryoet_checkpoints/pretrain_all_hi_3d.pt \
    --n_em_steps 200 \
    --steps_per_em 250 \
    --ckpt_dir mnist_cryoet_checkpoints/em_all_hi_3d_matched_s250
