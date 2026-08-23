#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-/tmp/sglang-fa3-kv-aliasing-i1-preflight}"
: "${I1_SM_CLOCK_MHZ:?Set I1_SM_CLOCK_MHZ to a supported fixed SM clock}"

if [[ -e "${RESULT_ROOT}/preflight-started" ]]; then
  echo "Refusing to reuse preflight root: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"
touch "${RESULT_ROOT}/preflight-started"

reset_clock() {
  nvidia-smi -rgc >"${RESULT_ROOT}/clock-reset.txt" 2>&1 || true
}
trap reset_clock EXIT

if ! ncu --query-metrics >"${RESULT_ROOT}/ncu-query-metrics.txt" 2>&1; then
  touch "${RESULT_ROOT}/PRECHECK_COUNTER_QUERY_FAILED"
  exit 10
fi
if ! nvidia-smi -lgc "${I1_SM_CLOCK_MHZ},${I1_SM_CLOCK_MHZ}" \
  >"${RESULT_ROOT}/clock-lock.txt" 2>&1; then
  touch "${RESULT_ROOT}/PRECHECK_CLOCK_LOCK_FAILED"
  exit 11
fi

ncu \
  --profile-from-start off \
  --target-processes all \
  --set basic \
  --export "${RESULT_ROOT}/counter-preflight" \
  --force-overwrite \
  --log-file "${RESULT_ROOT}/counter-preflight.log" \
  python -m sglang.benchmark.fa3_kv_aliasing_i1 counter-preflight \
  --output "${RESULT_ROOT}/counter-preflight.json"

test -s "${RESULT_ROOT}/counter-preflight.ncu-rep"
touch "${RESULT_ROOT}/PRECHECK_PASS"
reset_clock
trap - EXIT
