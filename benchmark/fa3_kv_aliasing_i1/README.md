# FA3 KV aliasing I1

Run the provider permission preflight before any benchmark:

```bash
I1_GPU_INDEX=0 \
I1_SM_CLOCK_MHZ=1410 \
RESULT_ROOT=/workspace/i1/preflight \
benchmark/fa3_kv_aliasing_i1/preflight_i1.sh
```

The preflight is standalone and requires only PyTorch, CUDA/NVIDIA tools, and
Nsight Compute. Read `RENTAL_RUNBOOK.md` before provisioning a paid host.

Only a completed preflight permits the HBM stage:

```bash
I1_GPU_INDEX=0 \
I1_SM_CLOCK_MHZ=1410 \
PREFLIGHT_ROOT=/workspace/i1/preflight \
RESULT_ROOT=/workspace/i1/hbm \
benchmark/fa3_kv_aliasing_i1/run_i1_hbm.sh
```

Gate C is intentionally not automated until Gate B has been reviewed.
