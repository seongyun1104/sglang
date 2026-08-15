#!/usr/bin/env bash
set -euo pipefail

# M0.5 screens coherent shared-prefix workloads for speculative acceptance only.
# It does not compare physical layouts or measure latency.

TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"
WORKLOADS="${WORKLOADS:-benchmark/radix_kv_sharing_m0/m05_workloads.json}"
RESULT_ROOT="${RESULT_ROOT:-results/radix-kv-sharing-m05-acceptance}"
PORT="${PORT:-30000}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-900}"
M05_STEPS="${M05_STEPS:-2 4}"
M05_SEEDS="${M05_SEEDS:-17 29 41}"
M05_BATCH_SIZE="${M05_BATCH_SIZE:-8}"
M05_OUTPUT_LENGTH="${M05_OUTPUT_LENGTH:-128}"

read -r -a steps <<<"${M05_STEPS}"
read -r -a seeds <<<"${M05_SEEDS}"
mapfile -t workload_ids < <(
  python -c 'import json,sys; print("\n".join(item["id"] for item in json.load(open(sys.argv[1]))["workloads"]))' "${WORKLOADS}"
)

if [[ -e "${RESULT_ROOT}/screen-started" ]]; then
  echo "Refusing to reuse acceptance-screen root: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}/captures" "${RESULT_ROOT}/logs" "${RESULT_ROOT}/metadata"
touch "${RESULT_ROOT}/screen-started"

nvidia-smi -q >"${RESULT_ROOT}/metadata/nvidia-smi-q.txt"
python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}))' \
  >"${RESULT_ROOT}/metadata/runtime.json"
git rev-parse HEAD >"${RESULT_ROOT}/metadata/git-head.txt"
cp "${WORKLOADS}" "${RESULT_ROOT}/metadata/workloads.json"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" || true
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT INT TERM

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

for step in "${steps[@]}"; do
  log_path="${RESULT_ROOT}/logs/server-k${step}.log"
  SGLANG_RADIX_KV_M0_RECORD=acceptance \
    python -m sglang.launch_server \
      --model-path "${TARGET_MODEL}" \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path "${DRAFT_MODEL}" \
      --speculative-num-steps "${step}" \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens "$((step + 1))" \
      --attention-backend fa3 \
      --speculative-draft-attention-backend fa3 \
      --context-length 2048 \
      --dtype bfloat16 \
      --host 127.0.0.1 \
      --port "${PORT}" >"${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${log_path}"

  for workload_id in "${workload_ids[@]}"; do
    for seed in "${seeds[@]}"; do
      python -m sglang.benchmark.radix_kv_sharing_m05 run \
        --workloads "${WORKLOADS}" \
        --workload-id "${workload_id}" \
        --batch-size "${M05_BATCH_SIZE}" \
        --seed "${seed}" \
        --output-length "${M05_OUTPUT_LENGTH}" \
        --base-url "http://127.0.0.1:${PORT}" \
        --output-dir "${RESULT_ROOT}/captures"
    done
  done

  kill "${server_pid}"
  wait "${server_pid}" || true
  server_pid=""
done

python -m sglang.benchmark.radix_kv_sharing_m05 analyze \
  --captures "${RESULT_ROOT}/captures" \
  --output "${RESULT_ROOT}/acceptance-by-seed.csv" \
  --summary-output "${RESULT_ROOT}/workload-screen.csv"

touch "${RESULT_ROOT}/screen-complete"
cleanup
trap - EXIT INT TERM
