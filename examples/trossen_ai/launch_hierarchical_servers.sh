#!/bin/bash
# Launch BOTH hierarchical-policy servers on one GPU node (H100/H200):
#   :8000  pi0.5 low-level policy  (openpi uv env, JAX)
#   :8001  Qwen3-VL subtask server (qwen3vl conda env, torch)
#
# JAX is started FIRST with a reduced memory fraction so torch Qwen (~17GB)
# fits beside it on the same card.
#
# Usage (on the GPU node):
#   bash examples/trossen_ai/launch_hierarchical_servers.sh
# Then from the robot workstation:
#   python examples/trossen_ai/eval_real_hierarchical.py \
#       --policy_host <this-node> --subtask_host <this-node> --test

set -euo pipefail

OPENPI_ROOT=${OPENPI_ROOT:-/iris/u/kewalk/openpi_trossen}
QWEN_ROOT=${QWEN_ROOT:-/iris/projects/humanoid/Qwen3-VL/qwen-vl-finetune}
QWEN_ENV=${QWEN_ENV:-/iris/projects/humanoid/miniconda3/envs/qwen3vl}

PI05_CONFIG=${PI05_CONFIG:-pi05_trossen_block_mem_subtask_full_delta_rtc}
PI05_CKPT=${PI05_CKPT:-checkpoints/pi05_trossen_block_mem_subtask_full_delta_rtc/pi05_block_mem_subtask_delta_rtc/5000}
QWEN_CKPT=${QWEN_CKPT:-output/qwen3vl-8b-trossen-merge-block-subtask-0528-3cls-hist5-5f-3xh100/checkpoint-400}

PI05_PORT=${PI05_PORT:-8000}
QWEN_PORT=${QWEN_PORT:-8001}
# pi0.5 needs ~8GB; 0.45 of an 80GB H100 leaves plenty for Qwen. On smaller
# cards lower this (e.g. 0.3) -- JAX must claim its share BEFORE Qwen loads.
XLA_FRACTION=${XLA_FRACTION:-0.45}

export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/iris/projects/humanoid/trossen_data}
export HF_HOME=${HF_HOME:-/iris/projects/humanoid/.cache/qwen3vl/huggingface}

LOG_DIR=${LOG_DIR:-/tmp/${USER}/hierarchical_servers}
mkdir -p "${LOG_DIR}"

cleanup() {
    echo "Shutting down servers..."
    [ -n "${PI05_PID:-}" ] && kill "${PI05_PID}" 2>/dev/null || true
    [ -n "${QWEN_PID:-}" ] && kill "${QWEN_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting pi0.5 policy server on :${PI05_PORT} (XLA fraction ${XLA_FRACTION})"
(
    cd "${OPENPI_ROOT}"
    XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_FRACTION} uv run scripts/serve_policy_hierarchical.py \
        --config="${PI05_CONFIG}" --dir="${PI05_CKPT}" --port="${PI05_PORT}"
) >"${LOG_DIR}/pi05.log" 2>&1 &
PI05_PID=$!

echo "==> Starting Qwen subtask server on :${QWEN_PORT}"
(
    cd "${QWEN_ROOT}"
    "${QWEN_ENV}/bin/python" tools/serve_subtask_predictor.py \
        --model-name-or-path "${QWEN_CKPT}" --port "${QWEN_PORT}"
) >"${LOG_DIR}/qwen.log" 2>&1 &
QWEN_PID=$!

wait_healthy() {
    local name=$1 port=$2 pid=$3
    for _ in $(seq 1 120); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} exited early. Last log lines:"
            tail -5 "${LOG_DIR}/${name}.log"
            exit 1
        fi
        if curl -sf "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            echo "==> ${name} ready on :${port}"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: ${name} did not become healthy in 10 minutes"; exit 1
}

wait_healthy pi05 "${PI05_PORT}" "${PI05_PID}"
wait_healthy qwen "${QWEN_PORT}" "${QWEN_PID}"

echo
echo "Both servers ready on $(hostname):"
echo "  pi0.5 policy : ws://$(hostname):${PI05_PORT}"
echo "  Qwen subtask : ws://$(hostname):${QWEN_PORT}"
echo "Logs: ${LOG_DIR}/{pi05,qwen}.log   (Ctrl-C stops both)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
wait
