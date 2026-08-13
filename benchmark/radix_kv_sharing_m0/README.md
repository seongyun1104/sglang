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

The released draft checkpoint reports a derived 2K context limit. The M0
runner explicitly sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, matching
SGLang's long-context speculative test setup, so the 8K/16K cells can start.
This override is recorded as part of the experiment contract and must be held
constant across shared and duplicated layouts.

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

For each launched server, run every cell matching its layout and K:

```bash
python -m sglang.benchmark.radix_kv_sharing_m0 run-matching \
  --plan-dir results/radix-kv-sharing-m0/plan \
  --output-dir results/radix-kv-sharing-m0/timing
```

Repeat against the accounting server, writing to a distinct directory. The footprint
pass uses 16 output tokens because it establishes physical-layout separation rather
than estimating latency; prompt fingerprints still have to match the timing pass, and
the two footprint layouts must have identical acceptance trajectories. The analyzer
excludes the first footprint decode record as warmup, then requires exact acceptance
and logical-context equality for every retained record:

```bash
python -m sglang.benchmark.radix_kv_sharing_m0 run-matching \
  --plan-dir results/radix-kv-sharing-m0/plan \
  --output-dir results/radix-kv-sharing-m0/footprint \
  --output-length-override 16
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
2. the number of measured verify records matches within each pass;
3. every per-step acceptance vector matches between physical layouts;
4. every per-step logical-context vector matches between physical layouts;
5. every retained runtime batch equals the preregistered batch size; paired tail
   records where both layouts underfill identically after an early EOS are excluded
   and counted, while asymmetric underfill invalidates the cell;
6. measured physical page reuse is higher in the shared layout.

`unique_physical_pages` and `page_reuse_ratio` are accounting quantities, not HBM
traffic measurements. They are candidate predictors only.

## Decision rule

M0 passes only if a paired verify-time change is reproducible at fixed
`BS × context × K`, after the hard controls pass. If the change is below measurement
resolution or is not reproducible across seeds, stop. Do not implement a controller.
The minimum effect is an explicit analyzer input, not a hidden constant. `2.0` above
is only the preregistered value for the first H100 run and must be replaced before
execution if the timing preflight demonstrates a larger measurement floor.

The first matrix is a screen, not a publishable percent-level claim. If it emits
`M0_SIGNAL`, rerun only the positive cells with counterbalanced server restart order
(`shared → duplicated → duplicated → shared`) and clock/temperature/power telemetry.
The signal must survive that confirmation before starting M1.

The current screen emitted a signal in every cell. The confirmation therefore keeps
the four `BS × context` corners but restricts fixed K to `0` and `4`. It runs timing
only; the full screen already established physical-layout separation with a distinct
footprint pass. Both restart orders must independently pass prompt, acceptance,
logical-context, provenance, and runtime-batch controls, and every seed must retain
the same resolved direction:

```bash
RESULT_ROOT=results/radix-kv-sharing-m0-confirmation \
benchmark/radix_kv_sharing_m0/run_confirmation.sh
```

The output status is `M0_CONFIRMED` only when all three seeds exceed the configured
measurement floor in both execution orders. A confirmation result does not imply an
optimal-K shift. M1 remains blocked until a semantically coherent shared-prefix
workload produces non-zero draft acceptance while retaining the paired controls.

The 2026-08-13 H100 NVL run confirmed all eight `BS × context × K` cells across both
restart orders and all three seeds. The counterbalanced shared-layout speedup ranged
from 12.55% to 27.61%. This closes M0, but not M1: the K=4 captures contained only 20
non-zero correct-draft decisions among 73,708 request decisions. See
`M0_CONFIRMATION_20260813.md` for the complete result and interpretation boundary.

## Screen runner

`run_screen.sh` automates the twelve required server configurations, retains server
logs and 500 ms GPU telemetry, and stops on the first failed cell. A one-cell smoke is:

```bash
M0_BATCH_SIZES=8 \
M0_CONTEXT_LENGTHS=8192 \
M0_STEPS=2 \
M0_SEEDS=17 \
RESULT_ROOT=results/radix-kv-sharing-m0-smoke \
benchmark/radix_kv_sharing_m0/run_screen.sh
```

Run the default matrix only after this smoke proves that the shared and duplicated
page footprints are distinct and that all hard controls pass.

If M0 passes, M1 may expand the K sweep and test whether the optimal K changes. A
feature PR is considered only after physical KV state adds meaningful oracle value
beyond the BS × context policy from PR #31716 and after feature-computation overhead.
