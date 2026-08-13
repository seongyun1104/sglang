#!/usr/bin/env bash
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"
RESULT_ROOT="${RESULT_ROOT:-results/radix-kv-sharing-m0}"
PORT="${PORT:-30000}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-900}"
M0_BATCH_SIZES="${M0_BATCH_SIZES:-8 16}"
M0_CONTEXT_LENGTHS="${M0_CONTEXT_LENGTHS:-8192 16384}"
M0_STEPS="${M0_STEPS:-0 2 4}"
M0_SEEDS="${M0_SEEDS:-17 29 41}"
M0_COMPONENTS="${M0_COMPONENTS:-target_verify_gpu_time kv_footprint}"
M0_MINIMUM_EFFECT_PERCENT="${M0_MINIMUM_EFFECT_PERCENT:-2.0}"

read -r -a batch_sizes <<<"${M0_BATCH_SIZES}"
read -r -a context_lengths <<<"${M0_CONTEXT_LENGTHS}"
read -r -a steps <<<"${M0_STEPS}"
read -r -a seeds <<<"${M0_SEEDS}"
read -r -a components <<<"${M0_COMPONENTS}"

mkdir -p "${RESULT_ROOT}/plan" "${RESULT_ROOT}/logs" "${RESULT_ROOT}/metadata"

python -m sglang.benchmark.radix_kv_sharing_m0 plan \
  --output "${RESULT_ROOT}/plan" \
  --batch-sizes "${batch_sizes[@]}" \
  --context-lengths "${context_lengths[@]}" \
  --steps "${steps[@]}" \
  --seeds "${seeds[@]}"

nvidia-smi -q >"${RESULT_ROOT}/metadata/nvidia-smi-q.txt"
python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}))' \
  >"${RESULT_ROOT}/metadata/runtime.json"
git rev-parse HEAD >"${RESULT_ROOT}/metadata/git-head.txt"

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

for component in "${components[@]}"; do
  case "${component}" in
    target_verify_gpu_time)
      output_dir="${RESULT_ROOT}/timing"
      output_override=()
      ;;
    kv_footprint)
      output_dir="${RESULT_ROOT}/footprint"
      output_override=(--output-length-override 16)
      ;;
    *)
      echo "Unknown component: ${component}" >&2
      exit 2
      ;;
  esac

  for step in "${steps[@]}"; do
    for layout in shared duplicated; do
      log_path="${RESULT_ROOT}/logs/server-${component}-${layout}-k${step}.log"
      layout_args=()
      if [[ "${layout}" == "duplicated" ]]; then
        layout_args=(--disable-radix-cache)
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

      python -m sglang.benchmark.radix_kv_sharing_m0 run-matching \
        --plan-dir "${RESULT_ROOT}/plan" \
        --base-url "http://127.0.0.1:${PORT}" \
        --output-dir "${output_dir}" \
        --skip-existing \
        "${output_override[@]}"

      kill "${server_pid}"
      wait "${server_pid}" || true
      server_pid=""
    done
  done
done

python -m sglang.benchmark.radix_kv_sharing_m0 analyze \
  --captures "${RESULT_ROOT}/timing" "${RESULT_ROOT}/footprint" \
  --output "${RESULT_ROOT}/paired-summary.csv" \
  --aggregate-output "${RESULT_ROOT}/m0-gate.csv" \
  --minimum-effect-percent "${M0_MINIMUM_EFFECT_PERCENT}"

cleanup
trap - EXIT INT TERM
