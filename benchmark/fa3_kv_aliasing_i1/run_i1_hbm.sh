#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-/tmp/sglang-fa3-kv-aliasing-i1-hbm}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-/tmp/sglang-fa3-kv-aliasing-i1-preflight}"
I1_HBM_METRICS="${I1_HBM_METRICS:-dram__bytes_read.sum,dram__sectors_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed}"
: "${I1_SM_CLOCK_MHZ:?Set I1_SM_CLOCK_MHZ to the preflighted fixed SM clock}"

if [[ ! -f "${PREFLIGHT_ROOT}/PRECHECK_PASS" ]]; then
  echo "I1 counter/clock preflight did not pass." >&2
  exit 2
fi
for receipt in visible-gpu-uuid.txt requested-sm-clock-mhz.txt requested-hbm-metrics.txt; do
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
if [[ "${I1_SM_CLOCK_MHZ}" != "$(<"${PREFLIGHT_ROOT}/requested-sm-clock-mhz.txt")" ]]; then
  echo "I1 clock does not match the preflighted clock." >&2
  exit 2
fi
if [[ "${I1_HBM_METRICS}" != "$(<"${PREFLIGHT_ROOT}/requested-hbm-metrics.txt")" ]]; then
  echo "I1 metrics do not match the preflighted counter set." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run I1 from a dirty worktree." >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}/i1-started" ]]; then
  echo "Refusing to reuse I1 root: ${RESULT_ROOT}" >&2
  exit 2
fi
reset_clock() {
  mkdir -p "${RESULT_ROOT}/metadata"
  nvidia-smi -rgc >"${RESULT_ROOT}/metadata/clock-reset.txt" 2>&1 || true
}
trap reset_clock EXIT
nvidia-smi -lgc "${I1_SM_CLOCK_MHZ},${I1_SM_CLOCK_MHZ}" >/tmp/i1-clock-lock.txt 2>&1

# Run the clean-worktree latency gate before creating any I1-owned artifact.
RESULT_ROOT="${RESULT_ROOT}/latency" \
I0_REPETITIONS=30 \
I0_WARMUP=10 \
benchmark/fa3_kv_aliasing_i0/run_i0.sh

python -m sglang.benchmark.fa3_kv_aliasing_i1 validate-latency-gate \
  --summary "${RESULT_ROOT}/latency/i0-summary.json" \
  --output "${RESULT_ROOT}/latency/i1-latency-gate.json"

mkdir -p "${RESULT_ROOT}/metadata" "${RESULT_ROOT}/profiles"
touch "${RESULT_ROOT}/i1-started"
git rev-parse HEAD >"${RESULT_ROOT}/metadata/git-head.txt"
nvidia-smi -q >"${RESULT_ROOT}/metadata/nvidia-smi-q.txt"
python -m pip freeze >"${RESULT_ROOT}/metadata/pip-freeze.txt"
ncu --version >"${RESULT_ROOT}/metadata/ncu-version.txt" 2>&1
cp /tmp/i1-clock-lock.txt "${RESULT_ROOT}/metadata/clock-lock.txt"
nvidia-smi -lgc "${I1_SM_CLOCK_MHZ},${I1_SM_CLOCK_MHZ}" \
  >"${RESULT_ROOT}/metadata/clock-lock.txt" 2>&1

profiles=(shared_aliased duplicated_contiguous duplicated_contiguous shared_aliased)
for index in "${!profiles[@]}"; do
  layout="${profiles[$index]}"
  ordinal=$(printf '%02d' "$((index + 1))")
  ncu \
    --profile-from-start off \
    --target-processes all \
    --kernel-name 'regex:.*FlashAttnFwdSm90.*' \
    --launch-count 1 \
    --metrics "${I1_HBM_METRICS}" \
    --csv \
    --export "${RESULT_ROOT}/profiles/${ordinal}-${layout}" \
    --force-overwrite \
    --log-file "${RESULT_ROOT}/profiles/${ordinal}-${layout}.csv" \
    python -m sglang.benchmark.fa3_kv_aliasing_i1 profile-layout \
    --layout "${layout}" \
    --output "${RESULT_ROOT}/profiles/${ordinal}-${layout}.json" \
    --seed 17 \
    --warmup 10
done

python -m sglang.benchmark.fa3_kv_aliasing_i1 validate-pair \
  --a "${RESULT_ROOT}/profiles/01-shared_aliased.json" \
  --b "${RESULT_ROOT}/profiles/02-duplicated_contiguous.json" \
  --output "${RESULT_ROOT}/profiles/pair-validation-forward.json"
python -m sglang.benchmark.fa3_kv_aliasing_i1 validate-pair \
  --a "${RESULT_ROOT}/profiles/04-shared_aliased.json" \
  --b "${RESULT_ROOT}/profiles/03-duplicated_contiguous.json" \
  --output "${RESULT_ROOT}/profiles/pair-validation-reverse.json"

touch "${RESULT_ROOT}/GATE_B_CAPTURE_COMPLETE_REVIEW_REQUIRED"
reset_clock
trap - EXIT
