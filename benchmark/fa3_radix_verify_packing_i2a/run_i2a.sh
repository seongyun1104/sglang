#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-/tmp/sglang-fa3-radix-verify-packing-i2a}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-/tmp/sglang-fa3-kv-aliasing-i1-preflight}"
: "${I1_GATE_B_REVIEW_RECEIPT:?Set I1_GATE_B_REVIEW_RECEIPT after reviewing I1 Gate B}"
: "${I2A_SM_CLOCK_MHZ:?Set I2A_SM_CLOCK_MHZ to the preflighted fixed SM clock}"
I2A_CONTEXTS="${I2A_CONTEXTS:-8192 16384}"
I2A_SPECULATIVE_STEPS="${I2A_SPECULATIVE_STEPS:-1 2 4}"
I2A_SEEDS="${I2A_SEEDS:-17 29 41}"
I2A_REPETITIONS="${I2A_REPETITIONS:-50}"
I2A_WARMUP="${I2A_WARMUP:-10}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run I2a from a dirty worktree." >&2
  exit 2
fi
if [[ ! -s "${I1_GATE_B_REVIEW_RECEIPT}" ]] || \
  ! grep -Fxq "I1_GATE_B_REVIEWED" "${I1_GATE_B_REVIEW_RECEIPT}"; then
  echo "I2a requires an explicit I1 Gate B review receipt." >&2
  exit 2
fi
experiment_commit="$(git rev-parse HEAD)"
if ! grep -Fxq "experiment_commit=${experiment_commit}" \
  "${I1_GATE_B_REVIEW_RECEIPT}"; then
  echo "I1 Gate B review receipt does not match the current experiment commit." >&2
  exit 2
fi
if [[ ! -f "${PREFLIGHT_ROOT}/PRECHECK_PASS" ]]; then
  echo "I2a requires the counter/clock permission preflight." >&2
  exit 2
fi
for receipt in visible-gpu-uuid.txt requested-sm-clock-mhz.txt; do
  if [[ ! -s "${PREFLIGHT_ROOT}/${receipt}" ]]; then
    echo "Missing preflight receipt: ${receipt}" >&2
    exit 2
  fi
done
current_gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
if [[ "$(printf '%s\n' "${current_gpu_uuid}" | wc -l)" -ne 1 ]] || \
  [[ "${current_gpu_uuid}" != "$(<"${PREFLIGHT_ROOT}/visible-gpu-uuid.txt")" ]]; then
  echo "Current GPU does not match the preflighted GPU." >&2
  exit 2
fi
if [[ "${I2A_SM_CLOCK_MHZ}" != "$(<"${PREFLIGHT_ROOT}/requested-sm-clock-mhz.txt")" ]]; then
  echo "I2a clock does not match the preflighted clock." >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}/i2a-started" ]]; then
  echo "Refusing to reuse result root: ${RESULT_ROOT}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}/metadata"
touch "${RESULT_ROOT}/i2a-started"
git rev-parse HEAD >"${RESULT_ROOT}/metadata/git-head.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${RESULT_ROOT}/metadata/started-at-utc.txt"
nvidia-smi -q >"${RESULT_ROOT}/metadata/nvidia-smi-q.txt"
python -m pip freeze >"${RESULT_ROOT}/metadata/pip-freeze.txt"
python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}))' \
  >"${RESULT_ROOT}/metadata/runtime.json"

reset_clock() {
  nvidia-smi -rgc >"${RESULT_ROOT}/metadata/clock-reset.txt" 2>&1 || true
}
TELEMETRY_PID=""
stop_telemetry() {
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
}
cleanup() {
  stop_telemetry
  reset_clock
}
trap cleanup EXIT

nvidia-smi -lgc "${I2A_SM_CLOCK_MHZ},${I2A_SM_CLOCK_MHZ}" \
  >"${RESULT_ROOT}/metadata/clock-lock.txt" 2>&1

nvidia-smi \
  --query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used \
  --format=csv \
  --loop-ms=500 \
  --filename="${RESULT_ROOT}/metadata/gpu-telemetry.csv" &
TELEMETRY_PID=$!

python -m sglang.benchmark.fa3_radix_verify_packing_i2a run \
  --output-dir "${RESULT_ROOT}" \
  --contexts ${I2A_CONTEXTS} \
  --speculative-steps ${I2A_SPECULATIVE_STEPS} \
  --seeds ${I2A_SEEDS} \
  --repetitions "${I2A_REPETITIONS}" \
  --warmup "${I2A_WARMUP}"

stop_telemetry
reset_clock
trap - EXIT
date -u +%Y-%m-%dT%H:%M:%SZ >"${RESULT_ROOT}/metadata/completed-at-utc.txt"
touch "${RESULT_ROOT}/i2a-complete"
