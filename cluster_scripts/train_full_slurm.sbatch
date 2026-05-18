#!/bin/bash
#SBATCH --job-name=pi05_human_robot_pack_full_0514
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodelist=iris-hgx-1,iris-hgx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/iris/u/kewalk/openpi_trossen/logs/human_robot_pack_full_0514/%x-%j.out
#SBATCH --error=/iris/u/kewalk/openpi_trossen/logs/human_robot_pack_full_0514/%x-%j.err

set -euo pipefail

cd /iris/u/kewalk/openpi_trossen
source /iris/u/kewalk/.bashrc.user

export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME ===="
nvidia-smi
echo "================================================"

# echo "==== Computing norm stats ===="
# uv run scripts/compute_norm_stats.py \
#     --config-name pi05_trossen_pack_with_human_full

echo "==== Starting training ===="
uv run scripts/train.py \
    pi05_trossen_pack_with_human_full \
    --exp-name=pi05_pack_with_human_full_0515 \
    --overwrite
