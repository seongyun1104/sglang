# FA3 Radix-local verify packing I2a

Read `I2A_CONTRACT.md` before execution. Generate the CPU-only plan with:

```bash
python -m sglang.benchmark.fa3_radix_verify_packing_i2a plan \
  --output /tmp/i2a-plan.json
```

The H100 runner refuses a dirty worktree and retains raw timing, output checks,
runtime metadata, and GPU telemetry:

```bash
I2A_GPU_INDEX=0 \
I2A_SM_CLOCK_MHZ=1410 \
PREFLIGHT_ROOT=/workspace/i1/preflight \
I1_GATE_B_REVIEW_RECEIPT=/workspace/i1/hbm/GATE_B_REVIEW_RECEIPT \
RESULT_ROOT=/workspace/i2a/results \
benchmark/fa3_radix_verify_packing_i2a/run_i2a.sh
```

The review receipt must contain `I1_GATE_B_REVIEWED` and the current experiment
commit. Do not interpret a CPU plan or reduced smoke as hardware evidence.
