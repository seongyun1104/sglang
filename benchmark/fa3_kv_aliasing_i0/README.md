# FA3 KV-page aliasing I0

This directory contains a minimal H100 mechanism-localization experiment for the
physical-KV-sharing latency signal observed by the Radix M0 study.

The default shape is the full-attention component of `Qwen/Qwen3.8-27B`. Model
weights are not downloaded; the checkpoint is provenance for the Q/KV head geometry.

Read `INVESTIGATION_CONTRACT.md` before running anything. The first milestone is
latency reproduction across three page layouts. It does not run Nsight, implement
an optimization, or change speculative-depth selection.

```bash
RESULT_ROOT=results/fa3-kv-aliasing-i0 \
benchmark/fa3_kv_aliasing_i0/run_i0.sh
```

A small GPU smoke can override the anchor without changing the full-run contract:

```bash
RESULT_ROOT=results/fa3-kv-aliasing-i0-smoke \
I0_BATCH_SIZE=2 \
I0_CONTEXT_LENGTH=1024 \
I0_SEEDS=17 \
I0_REPETITIONS=5 \
I0_WARMUP=2 \
benchmark/fa3_kv_aliasing_i0/run_i0.sh
```

Do not use smoke output as I0 evidence.
