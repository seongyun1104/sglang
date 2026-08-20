#!/usr/bin/env bash
set -euo pipefail

# M1 counterbalanced fixed-K sweep. Timing order is ascending-K AB followed by
# descending-K BA. Footprint accounting is a separate process and never shares a
# server with timing collection.

TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"
WORKLOAD_FILE="${WORKLOAD_FILE:-benchmark/radix_kv_sharing_m0/m05_workloads.json}"
RESULT_ROOT="${RESULT_ROOT:-results/radix-kv-sharing-m1}"
PORT="${PORT:-30000}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-900}"
M1_MINIMUM_EFFECT_PERCENT="${M1_MINIMUM_EFFECT_PERCENT:-2.0}"
M1_SMOKE_ONLY="${M1_SMOKE_ONLY:-0}"
M1_STEPS="${M1_STEPS:-0 1 2 3 4 5}"
M1_WORKLOADS="${M1_WORKLOADS:-code_completion structured_json}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

read -r -a steps <<<"${M1_STEPS}"
read -r -a workloads <<<"${M1_WORKLOADS}"
if [[ "${M1_SMOKE_ONLY}" == "1" ]]; then
  steps=(2)
  workloads=(code_completion)
fi

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run M1 from a dirty worktree; commit the exact experiment first." >&2
  exit 2
fi

if [[ -e "${RESULT_ROOT}/m1-started" ]]; then
  echo "Refusing to reuse M1 result root: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p \
  "${RESULT_ROOT}/plan" \
  "${RESULT_ROOT}/logs" \
  "${RESULT_ROOT}/metadata" \
  "${RESULT_ROOT}/timing/forward" \
  "${RESULT_ROOT}/timing/reverse" \
  "${RESULT_ROOT}/footprint"
touch "${RESULT_ROOT}/m1-started"

for workload in "${workloads[@]}"; do
  python -m sglang.benchmark.radix_kv_sharing_m1 plan \
    --workload-file "${WORKLOAD_FILE}" \
    --workload-id "${workload}" \
    --plan-dir "${RESULT_ROOT}/plan/${workload}"
done

nvidia-smi -q >"${RESULT_ROOT}/metadata/nvidia-smi-q.txt"
python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}))' \
  >"${RESULT_ROOT}/metadata/runtime.json"
git rev-parse HEAD >"${RESULT_ROOT}/metadata/git-head.txt"
printf '%s\n' \
  'forward: ascending K, shared -> duplicated' \
  'reverse: descending K, duplicated -> shared' \
  'footprint: ascending K, shared -> duplicated' \
  'primary denominator: draft_gpu_ms + target_verify_gpu_ms' \
  'sensitivity denominator: spec_cycle_gpu_ms' \
  'ignore_eos: false' \
  >"${RESULT_ROOT}/metadata/execution-contract.txt"

server_pid=""
telemetry_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" || true
    wait "${server_pid}" || true
  fi
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill "${telemetry_pid}" || true
    wait "${telemetry_pid}" || true
  fi
}
trap cleanup EXIT INT TERM

nvidia-smi \
  --query-gpu=timestamp,index,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu \
  --format=csv -lms 500 >"${RESULT_ROOT}/metadata/gpu-telemetry.csv" &
telemetry_pid=$!

wait_for_server() {
  local log_path="$1"
  local started_at
  started_at=$(date +%s)
  while ! curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -n 200 "${log_path}"
      return 1
    fi
    if (( $(date +%s) - started_at > SERVER_TIMEOUT_SECONDS )); then
      tail -n 200 "${log_path}"
      return 1
    fi
    sleep 2
  done
}

run_server() {
  local component="$1"
  local round="$2"
  local ordinal="$3"
  local step="$4"
  local layout="$5"
  local output_root="$6"
  local log_path="${RESULT_ROOT}/logs/${component}-${round}-${ordinal}-${layout}-k${step}.log"
  local layout_args=()
  local output_override_args=()
  if [[ "${layout}" == "duplicated" ]]; then
    layout_args=(--disable-radix-cache)
  fi
  if [[ "${component}" == "kv_footprint" ]]; then
    output_override_args=(--output-length-override 16)
  fi

  SGLANG_RADIX_KV_M0_RECORD="${component}" \
    python -m sglang.launch_server \
      --model-path "${TARGET_MODEL}" \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path "${DRAFT_MODEL}" \
      --speculative-num-steps "${step}" \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens "$((step + 1))" \
      --attention-backend fa3 \
      --speculative-draft-attention-backend fa3 \
      --context-length 20000 \
      --dtype bfloat16 \
      --host 127.0.0.1 \
      --port "${PORT}" \
      "${layout_args[@]}" >"${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${log_path}"

  for workload in "${workloads[@]}"; do
    python -m sglang.benchmark.radix_kv_sharing_m1 run-matching \
      --plan-dir "${RESULT_ROOT}/plan/${workload}" \
      --base-url "http://127.0.0.1:${PORT}" \
      --output-dir "${output_root}/${workload}" \
      "${output_override_args[@]}"
  done

  kill "${server_pid}"
  wait "${server_pid}" || true
  server_pid=""
}

ordinal=0
for step in "${steps[@]}"; do
  ordinal=$((ordinal + 1))
  run_server spec_cycle_gpu_time forward "${ordinal}" "${step}" shared "${RESULT_ROOT}/timing/forward"
  ordinal=$((ordinal + 1))
  run_server spec_cycle_gpu_time forward "${ordinal}" "${step}" duplicated "${RESULT_ROOT}/timing/forward"
done

if [[ "${M1_SMOKE_ONLY}" != "1" ]]; then
  for ((index=${#steps[@]} - 1; index >= 0; index--)); do
    step="${steps[index]}"
    ordinal=$((ordinal + 1))
    run_server spec_cycle_gpu_time reverse "${ordinal}" "${step}" duplicated "${RESULT_ROOT}/timing/reverse"
    ordinal=$((ordinal + 1))
    run_server spec_cycle_gpu_time reverse "${ordinal}" "${step}" shared "${RESULT_ROOT}/timing/reverse"
  done
fi

for step in "${steps[@]}"; do
  ordinal=$((ordinal + 1))
  run_server kv_footprint accounting "${ordinal}" "${step}" shared "${RESULT_ROOT}/footprint"
  ordinal=$((ordinal + 1))
  run_server kv_footprint accounting "${ordinal}" "${step}" duplicated "${RESULT_ROOT}/footprint"
done

if [[ "${M1_SMOKE_ONLY}" == "1" ]]; then
  python -m sglang.benchmark.radix_kv_sharing_m1 analyze-smoke \
    --timing-captures "${RESULT_ROOT}/timing/forward" \
    --footprint-captures "${RESULT_ROOT}/footprint" \
    >"${RESULT_ROOT}/smoke-summary.json"
  touch "${RESULT_ROOT}/m1-smoke-complete"
else
  python -m sglang.benchmark.radix_kv_sharing_m1 analyze \
    --forward-captures "${RESULT_ROOT}/timing/forward" \
    --reverse-captures "${RESULT_ROOT}/timing/reverse" \
    --footprint-captures "${RESULT_ROOT}/footprint" \
    --output "${RESULT_ROOT}/counterbalanced-paired-controls.json" \
    --summary-output "${RESULT_ROOT}/m1-summary.json" \
    --minimum-effect-percent "${M1_MINIMUM_EFFECT_PERCENT}"
  touch "${RESULT_ROOT}/m1-complete"
fi

cleanup
trap - EXIT INT TERM
