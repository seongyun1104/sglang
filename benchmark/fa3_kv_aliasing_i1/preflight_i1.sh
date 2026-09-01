#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-/tmp/sglang-fa3-kv-aliasing-i1-preflight}"
I1_HBM_METRICS="${I1_HBM_METRICS:-dram__bytes_read.sum,dram__sectors_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed}"
I1_GPU_INDEX="${I1_GPU_INDEX:-0}"
: "${I1_SM_CLOCK_MHZ:?Set I1_SM_CLOCK_MHZ to a supported fixed SM clock}"

if [[ ! "${I1_GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "I1_GPU_INDEX must be a non-negative integer." >&2
  exit 2
fi

if [[ -e "${RESULT_ROOT}/preflight-started" ]]; then
  echo "Refusing to reuse preflight root: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"
touch "${RESULT_ROOT}/preflight-started"
printf '%s\n' "${I1_SM_CLOCK_MHZ}" >"${RESULT_ROOT}/requested-sm-clock-mhz.txt"
printf '%s\n' "${I1_HBM_METRICS}" >"${RESULT_ROOT}/requested-hbm-metrics.txt"
printf '%s\n' "${I1_GPU_INDEX}" >"${RESULT_ROOT}/selected-gpu-index.txt"

reset_clock() {
  nvidia-smi -i "${I1_GPU_INDEX}" -rgc >"${RESULT_ROOT}/clock-reset.txt" 2>&1 || true
}
trap reset_clock EXIT

failed=0
if ! nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader \
  >"${RESULT_ROOT}/all-visible-gpus.txt" 2>"${RESULT_ROOT}/visible-gpu-query.err"; then
  touch "${RESULT_ROOT}/PRECHECK_GPU_IDENTITY_FAILED"
  failed=1
elif ! nvidia-smi -i "${I1_GPU_INDEX}" --query-gpu=uuid,name --format=csv,noheader \
  >"${RESULT_ROOT}/selected-gpu.txt" 2>"${RESULT_ROOT}/selected-gpu-query.err"; then
  touch "${RESULT_ROOT}/PRECHECK_SELECTED_GPU_FAILED"
  failed=1
elif [[ "$(wc -l <"${RESULT_ROOT}/selected-gpu.txt")" -ne 1 ]] || \
  ! grep -q 'H100' "${RESULT_ROOT}/selected-gpu.txt"; then
  touch "${RESULT_ROOT}/PRECHECK_REQUIRES_SELECTED_H100"
  failed=1
else
  cut -d, -f1 "${RESULT_ROOT}/selected-gpu.txt" | tr -d ' ' \
    >"${RESULT_ROOT}/selected-gpu-uuid.txt"
fi
if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader >"${RESULT_ROOT}/preexisting-compute-apps.txt" \
  2>"${RESULT_ROOT}/preexisting-compute-apps.err"; then
  touch "${RESULT_ROOT}/PRECHECK_COMPUTE_APP_QUERY_FAILED"
  failed=1
elif [[ -s "${RESULT_ROOT}/preexisting-compute-apps.txt" ]]; then
  touch "${RESULT_ROOT}/PRECHECK_HOST_NOT_IDLE"
  failed=1
fi
if ! ncu --query-metrics >"${RESULT_ROOT}/ncu-query-metrics.txt" 2>&1; then
  touch "${RESULT_ROOT}/PRECHECK_COUNTER_QUERY_FAILED"
  failed=1
fi
if ! nvidia-smi -i "${I1_GPU_INDEX}" -lgc "${I1_SM_CLOCK_MHZ},${I1_SM_CLOCK_MHZ}" \
  >"${RESULT_ROOT}/clock-lock.txt" 2>&1; then
  touch "${RESULT_ROOT}/PRECHECK_CLOCK_LOCK_FAILED"
  failed=1
fi

export I1_COUNTER_PREFLIGHT_OUTPUT="${RESULT_ROOT}/counter-preflight.json"
export CUDA_VISIBLE_DEVICES="${I1_GPU_INDEX}"
if ! ncu \
  --profile-from-start off \
  --target-processes all \
  --metrics "${I1_HBM_METRICS}" \
  --export "${RESULT_ROOT}/counter-preflight" \
  --force-overwrite \
  --log-file "${RESULT_ROOT}/counter-preflight.log" \
  python -c 'import json, os, torch; assert torch.cuda.is_available(); assert "H100" in torch.cuda.get_device_name(); assert torch.cuda.get_device_capability()[0] == 9; x=torch.arange(1<<20,dtype=torch.float32,device="cuda"); y=torch.arange(1<<20,dtype=torch.float32,device="cuda"); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart(); z=x+y; torch.cuda.cudart().cudaProfilerStop(); torch.cuda.synchronize(); result={"device":torch.cuda.get_device_name(),"compute_capability":list(torch.cuda.get_device_capability()),"torch":torch.__version__,"cuda":torch.version.cuda,"kernel":"torch.add","elements":x.numel(),"checksum":float(z[:1024].sum())}; open(os.environ["I1_COUNTER_PREFLIGHT_OUTPUT"],"w").write(json.dumps(result,indent=2,sort_keys=True)+"\n")'; then
  touch "${RESULT_ROOT}/PRECHECK_COUNTER_CAPTURE_FAILED"
  failed=1
fi

if [[ ! -s "${RESULT_ROOT}/counter-preflight.ncu-rep" ]]; then
  touch "${RESULT_ROOT}/PRECHECK_COUNTER_REPORT_MISSING"
  failed=1
fi

if (( failed )); then
  touch "${RESULT_ROOT}/PRECHECK_FAILED"
  exit 12
fi
touch "${RESULT_ROOT}/PRECHECK_PASS"
reset_clock
trap - EXIT
