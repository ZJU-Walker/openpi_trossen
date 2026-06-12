#!/bin/bash
# Run the offline hierarchical eval (stage 2) over the val episodes, with both
# Qwen-predicted prompts and oracle GT prompts, then print a comparison table.
#
# Prereq: stage-1 predictions JSONs (one per episode) from
#   Qwen3-VL/qwen-vl-finetune/tools/predict_subtasks_lerobot.py
#
# Usage:
#   cd /iris/u/kewalk/openpi_trossen
#   bash examples/trossen_ai/run_hierarchical_eval.sh <stage1_preds_dir> [episodes...]

set -euo pipefail

STAGE1_DIR=${1:?usage: run_hierarchical_eval.sh <stage1_preds_dir> [episodes...]}
shift
EPISODES=("$@")
if [ ${#EPISODES[@]} -eq 0 ]; then
    EPISODES=(27 28 29 30 56 57 58 59 60)
fi

CONFIG=${CONFIG:-pi05_trossen_block_mem_subtask_full_delta_rtc}
CKPT=${CKPT:-checkpoints/pi05_trossen_block_mem_subtask_full_delta_rtc/pi05_block_mem_subtask_delta_rtc/7000}
CHUNK=${CHUNK:-50}

export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/iris/projects/humanoid/trossen_data}

for ep in "${EPISODES[@]}"; do
    printf -v ep6 "%06d" "$ep"
    for source in gt qwen; do
        extra=()
        if [ "$source" = "qwen" ]; then
            json="${STAGE1_DIR}/subtask_preds_ep${ep6}.json"
            if [ ! -f "$json" ]; then
                echo "SKIP episode $ep ($source): missing $json"
                continue
            fi
            extra=(--subtask-json "$json")
        fi
        echo "==== episode $ep | prompts=$source ===="
        uv run examples/trossen_ai/eval_offline_hierarchical.py \
            --config-name "$CONFIG" \
            --checkpoint-dir "$CKPT" \
            --episode-index "$ep" \
            --prompt-source "$source" \
            --chunk-size "$CHUNK" \
            "${extra[@]}"
    done
done

# --- Comparison table from the summary.jsons ---
step=$(basename "$CKPT")
exp=$(basename "$(dirname "$CKPT")")
cfg=$(basename "$(dirname "$(dirname "$CKPT")")")
OUT_ROOT="eval_outputs/${cfg}_${exp}_${step}"

python3 - "$OUT_ROOT" "${EPISODES[@]}" <<'EOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = [int(e) for e in sys.argv[2:]]
print(f"\n=== qwen vs oracle(gt) prompts | overall MAE per episode ===")
print(f"{'episode':>8} {'gt MAE':>10} {'qwen MAE':>10} {'qwen/gt':>8}")
gts, qwens = [], []
for ep in episodes:
    vals = {}
    for source in ("gt", "qwen"):
        p = root / f"episode_{ep}_{source}" / "summary.json"
        if p.exists():
            vals[source] = json.loads(p.read_text())["overall_mae"]
    g, q = vals.get("gt"), vals.get("qwen")
    ratio = f"{q / g:8.2f}" if g and q else "       -"
    print(f"{ep:>8} {g if g is not None else float('nan'):>10.5f} "
          f"{q if q is not None else float('nan'):>10.5f} {ratio}")
    if g is not None: gts.append(g)
    if q is not None: qwens.append(q)
if gts and qwens:
    print(f"{'mean':>8} {sum(gts)/len(gts):>10.5f} {sum(qwens)/len(qwens):>10.5f}")
EOF
