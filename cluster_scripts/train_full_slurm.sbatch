#!/bin/bash
#SBATCH --job-name=pi05_marker_abs_rtc_0527
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodelist=iris-hgx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kewalk@stanford.edu
#SBATCH --output=/iris/u/kewalk/openpi_trossen/logs/marker_abs_rtc_0527/%x-%j.out
#SBATCH --error=/iris/u/kewalk/openpi_trossen/logs/marker_abs_rtc_0527/%x-%j.err

set -euo pipefail

cd /iris/u/kewalk/openpi_trossen
source /iris/u/kewalk/.bashrc.user

export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb
export UV_CACHE_DIR=/tmp/${USER}/uv-cache
mkdir -p "${WANDB_DIR}" "${UV_CACHE_DIR}" /iris/u/kewalk/openpi_trossen/logs/marker_abs_rtc_0527

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME ===="
nvidia-smi
echo "================================================"

echo "==== Computing norm stats ===="
uv run scripts/compute_norm_stats.py \
    --config-name pi05_trossen_marker_handover_full_abs_rtc

echo "==== Starting training ===="
# uv run scripts/train.py \
#     pi05_trossen_pack_with_human_full \
#     --exp-name=pi05_pack_with_human_full_0515 \
#     --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_trossen_marker_handover_full_abs_rtc \
    --exp-name=pi05_marker_handover_abs_rtc_0527 \
    --overwrite
