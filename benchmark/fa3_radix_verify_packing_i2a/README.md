# FA3 Radix-local verify packing I2a

Read `I2A_CONTRACT.md` before execution. Generate the CPU-only plan with:

```bash
python -m sglang.benchmark.fa3_radix_verify_packing_i2a plan \
  --output /tmp/i2a-plan.json
```

The H100 runner refuses a dirty worktree and retains raw timing, output checks,
runtime metadata, and GPU telemetry:

```bash
I2A_SM_CLOCK_MHZ=1410 \
PREFLIGHT_ROOT=/workspace/i1/preflight \
RESULT_ROOT=/workspace/i2a/results \
benchmark/fa3_radix_verify_packing_i2a/run_i2a.sh
```

Do not interpret a CPU plan or reduced smoke as hardware evidence.
