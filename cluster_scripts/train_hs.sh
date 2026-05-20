#!/bin/bash
#SBATCH --job-name=pi05_human_robot_collab
#SBATCH --output=/iris/u/kewalk/openpi_trossen/logs/placement/pi05-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris-hgx-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G                  # adjust if you need less/more
#SBATCH --cpus-per-task=12         # tune if you want more data-loader threads
#SBATCH --mail-user=kewalk@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=iris

set -e

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME ===="
nvidia-smi
echo "================================================"

# go to your project
cd /iris/u/kewalk/openpi_trossen
source /iris/u/kewalk/.bashrc.user
export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb
# load your env
source .venv/bin/activate

uv run cluster_scripts/train_hs.py
