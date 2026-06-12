#!/bin/bash
#SBATCH --job-name=pi05_block_mem_subtask_delta_rtc
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodelist=iris-hgx-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kewalk@stanford.edu
#SBATCH --output=/iris/u/kewalk/openpi_trossen/logs/block_mem_subtask_delta_rtc/%x-%j.out
#SBATCH --error=/iris/u/kewalk/openpi_trossen/logs/block_mem_subtask_delta_rtc/%x-%j.err

set -euo pipefail

cd /iris/u/kewalk/openpi_trossen
source /iris/u/kewalk/.bashrc.user

export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb
export UV_CACHE_DIR=/tmp/${USER}/uv-cache
mkdir -p "${WANDB_DIR}" "${UV_CACHE_DIR}" /iris/u/kewalk/openpi_trossen/logs/block_mem_subtask_delta_rtc

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME ===="
nvidia-smi
echo "================================================"

# Norm stats are reused from pi05_trossen_block_mem_full_delta_rtc (same dataset,
# same delta-action scheme; stats do not depend on prompts), so no compute step.

echo "==== Starting training ===="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_trossen_block_mem_subtask_full_delta_rtc \
    --exp-name=pi05_block_mem_subtask_delta_rtc \
    --overwrite
