# Radix physical KV-sharing M0

This experiment asks one narrow question:

> At fixed batch size, logical context, prompts, acceptance trajectory, and
> speculative depth, does physical KV sharing change target-verification latency?

It is a falsification experiment based on PR #31716. It does not add a Radix-aware
controller and must not be folded into that PR.

## Fixed matrix

```text
batch size:       8, 16
logical context:  8192, 16384
physical layout:  shared, duplicated
fixed K:          0, 2, 4
seed:             17, 29, 41
```

`shared` uses the normal Radix cache. `duplicated` launches the same model and
workload with `--disable-radix-cache`. The primary interval is the target verify
path only; prefill and cache construction are outside the CUDA-event interval.

The target and draft models match the live-server validation used by PR #31716:

```text
target: meta-llama/Llama-3.1-8B-Instruct
draft:  lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
dtype:  BF16
```

## Required two-pass measurement

Timing and physical-page accounting run in separate server processes. Page-table
inspection materializes GPU scalars and is intentionally forbidden in a timing
process.

Timing process:

```bash
SGLANG_RADIX_KV_M0_RECORD=target_verify_gpu_time \
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-num-steps ${K} \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens $((K + 1)) \
  --context-length 20000 \
  --dtype bfloat16
```

Add `--disable-radix-cache` only for the duplicated-layout server. Relaunch the
server for each `(layout, K)` pair. The accounting process uses the same command
with:

```bash
SGLANG_RADIX_KV_M0_RECORD=kv_footprint
```

## Matrix and cell execution

Create the preregistered cells:

```bash
python -m sglang.benchmark.radix_kv_sharing_m0 plan \
  --output results/radix-kv-sharing-m0/plan
```

For each launched server, run only cells whose filename matches its layout and K:

```bash
python -m sglang.benchmark.radix_kv_sharing_m0 run-cell \
  --cell results/radix-kv-sharing-m0/plan/bs8-ctx8192-shared-k2-seed17.json \
  --output-dir results/radix-kv-sharing-m0/timing
```

Repeat against the accounting server, writing to a distinct directory:

```bash
python -m sglang.benchmark.radix_kv_sharing_m0 run-cell \
  --cell results/radix-kv-sharing-m0/plan/bs8-ctx8192-shared-k2-seed17.json \
  --output-dir results/radix-kv-sharing-m0/footprint
```

Analyze both passes together:

```bash
python -m sglang.benchmark.radix_kv_sharing_m0 analyze \
  --captures results/radix-kv-sharing-m0/timing \
             results/radix-kv-sharing-m0/footprint \
  --output results/radix-kv-sharing-m0/paired-summary.csv \
  --aggregate-output results/radix-kv-sharing-m0/m0-gate.csv \
  --minimum-effect-percent 2.0
```

## Hard controls

The analyzer suppresses speedup values unless all of the following hold:

1. tokenized prompt SHA-256 fingerprints match across physical layouts and passes;
2. the number of measured verify records matches;
3. every per-step acceptance vector matches;
4. every per-step logical-context vector matches.

`unique_physical_pages` and `page_reuse_ratio` are accounting quantities, not HBM
traffic measurements. They are candidate predictors only.

## Decision rule

M0 passes only if a paired verify-time change is reproducible at fixed
`BS × context × K`, after the hard controls pass. If the change is below measurement
resolution or is not reproducible across seeds, stop. Do not implement a controller.
The minimum effect is an explicit analyzer input, not a hidden constant. `2.0` above
is only the preregistered value for the first H100 run and must be replaced before
execution if the timing preflight demonstrates a larger measurement floor.

If M0 passes, M1 may expand the K sweep and test whether the optimal K changes. A
feature PR is considered only after physical KV state adds meaningful oracle value
beyond the BS × context policy from PR #31716 and after feature-computation overhead.
