#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-results/fa3-kv-aliasing-i0}"
I0_MODEL_ID="${I0_MODEL_ID:-Qwen/Qwen3.8-27B}"
I0_BATCH_SIZE="${I0_BATCH_SIZE:-16}"
I0_CONTEXT_LENGTH="${I0_CONTEXT_LENGTH:-16384}"
I0_QUERY_LENGTH="${I0_QUERY_LENGTH:-1}"
I0_SHARED_PREFIX_RATIO="${I0_SHARED_PREFIX_RATIO:-0.9}"
I0_PAGE_SIZE="${I0_PAGE_SIZE:-1}"
I0_NUM_QUERY_HEADS="${I0_NUM_QUERY_HEADS:-24}"
I0_NUM_KV_HEADS="${I0_NUM_KV_HEADS:-4}"
I0_HEAD_DIM="${I0_HEAD_DIM:-256}"
I0_SEEDS="${I0_SEEDS:-17 29 41}"
I0_REPETITIONS="${I0_REPETITIONS:-50}"
I0_WARMUP="${I0_WARMUP:-10}"
I0_L2_THRASH_MIB="${I0_L2_THRASH_MIB:-128}"
I0_MINIMUM_EFFECT_PERCENT="${I0_MINIMUM_EFFECT_PERCENT:-2.0}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run I0 from a dirty worktree; commit the exact experiment first." >&2
  exit 2
fi

if [[ -e "${RESULT_ROOT}/i0-started" ]]; then
  echo "Refusing to reuse I0 result root: ${RESULT_ROOT}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}/metadata"
touch "${RESULT_ROOT}/i0-started"

git rev-parse HEAD >"${RESULT_ROOT}/metadata/git-head.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${RESULT_ROOT}/metadata/started-at-utc.txt"
nvidia-smi -q >"${RESULT_ROOT}/metadata/nvidia-smi-q.txt"
python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}))' \
  >"${RESULT_ROOT}/metadata/runtime.json"

nvidia-smi \
  --query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used \
  --format=csv \
  --loop-ms=500 \
  --filename="${RESULT_ROOT}/metadata/gpu-telemetry.csv" &
TELEMETRY_PID=$!

stop_telemetry() {
  kill "${TELEMETRY_PID}" 2>/dev/null || true
  wait "${TELEMETRY_PID}" 2>/dev/null || true
}
trap stop_telemetry EXIT

python -m sglang.benchmark.fa3_kv_aliasing_i0 run \
  --output-dir "${RESULT_ROOT}" \
  --model-id "${I0_MODEL_ID}" \
  --batch-size "${I0_BATCH_SIZE}" \
  --context-length "${I0_CONTEXT_LENGTH}" \
  --query-length "${I0_QUERY_LENGTH}" \
  --shared-prefix-ratio "${I0_SHARED_PREFIX_RATIO}" \
  --page-size "${I0_PAGE_SIZE}" \
  --num-query-heads "${I0_NUM_QUERY_HEADS}" \
  --num-kv-heads "${I0_NUM_KV_HEADS}" \
  --head-dim "${I0_HEAD_DIM}" \
  --seeds ${I0_SEEDS} \
  --repetitions "${I0_REPETITIONS}" \
  --warmup "${I0_WARMUP}" \
  --l2-thrash-mib "${I0_L2_THRASH_MIB}" \
  --minimum-effect-percent "${I0_MINIMUM_EFFECT_PERCENT}"

stop_telemetry
trap - EXIT
date -u +%Y-%m-%dT%H:%M:%SZ >"${RESULT_ROOT}/metadata/completed-at-utc.txt"
touch "${RESULT_ROOT}/i0-complete"
