# I1/I2a rental runbook

This runbook separates provider capability screening from environment setup. Do
not install SGLang or run a benchmark before the capability gate passes.

## 1. Provider requirements

Require one selected H100 on an exclusive rental plus explicit permission for both:

```text
Nsight Compute performance counters
nvidia-smi application-clock control
```

The initial container only needs CUDA, Nsight Compute, NVIDIA utilities, and
PyTorch. Model weights are never downloaded.

A provider may expose more than one H100 only when the entire host is rented to this
experiment. Set `I1_GPU_INDEX` to one device, require every GPU to be idle at
preflight, and keep that index fixed through I1 and I2a. Shared or split multi-tenant
hosts remain forbidden.

## 2. First paid action

Copy only `preflight_i1.sh` to the new host and run it. The script has no SGLang
import and uses a small Torch add kernel for the real counter capture.

```bash
I1_GPU_INDEX=0 \
I1_SM_CLOCK_MHZ=<supported-clock> \
RESULT_ROOT=/workspace/radix-i1/preflight \
bash preflight_i1.sh
```

Proceed only when the exit code is zero and `PRECHECK_PASS` exists. On any other
result, collect the small preflight directory and terminate the instance.

## 3. Exact experiment checkout

Only after the capability gate passes:

```bash
git clone https://github.com/seongyun1104/sglang.git /workspace/radix-i1/sglang
cd /workspace/radix-i1/sglang
EXPERIMENT_COMMIT=<reviewed-SHA-from-local-git-rev-parse-HEAD>
git checkout "${EXPERIMENT_COMMIT}"
test "$(git rev-parse HEAD)" = "${EXPERIMENT_COMMIT}"
python -m pip install -e python
python -c 'import sglang, sgl_kernel, torch; print(torch.__version__, sglang.__file__, sgl_kernel.__file__)'
```

Do not substitute a different `sglang-kernel`, Torch, CUDA, FA version, or source
commit after the checkout has been reviewed. The runners record `pip freeze`, the
commit, GPU state, and NCU version in their artifact directories.

## 4. Execution order

```text
I1 latency reproduction
I1 latency gate validation
I1 Gate B HBM capture
manual Gate B review
I1 Gate C only if Gate B requires it
I2a only after the I1 environment remains valid
```

After inspecting both profile-pair validations and the captured HBM counters,
write a review receipt bound to the exact experiment commit:

```bash
I1_GATE_B_REVIEW_RECEIPT=/workspace/radix-i1/i1-hbm/GATE_B_REVIEW_RECEIPT
{
  echo I1_GATE_B_REVIEWED
  echo "experiment_commit=$(git rev-parse HEAD)"
} >"${I1_GATE_B_REVIEW_RECEIPT}"
```

Creating this receipt is a deliberate human review action, not part of the I1
runner. I2a fails closed if the receipt is absent or names another commit.

Commands:

```bash
I1_GPU_INDEX=0 \
I1_SM_CLOCK_MHZ=<same-clock> \
PREFLIGHT_ROOT=/workspace/radix-i1/preflight \
RESULT_ROOT=/workspace/radix-i1/i1-hbm \
benchmark/fa3_kv_aliasing_i1/run_i1_hbm.sh

I2A_GPU_INDEX=0 \
I2A_SM_CLOCK_MHZ=<same-clock> \
PREFLIGHT_ROOT=/workspace/radix-i1/preflight \
I1_GATE_B_REVIEW_RECEIPT=/workspace/radix-i1/i1-hbm/GATE_B_REVIEW_RECEIPT \
RESULT_ROOT=/workspace/radix-i1/i2a \
benchmark/fa3_radix_verify_packing_i2a/run_i2a.sh
```

I2a must not be interpreted until I1 Gate B artifacts have been copied and
reviewed. Destroy the instance after copying artifacts and verifying checksums.
