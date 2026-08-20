# M1 counterbalanced paired-layout K sweep

## Research question

> Does physical KV sharing move the optimal fixed speculative depth for an
> acceptance-bearing shared-prefix workload?

M0 established that physical sharing changes target-verification cost. M0.5
qualified two natural workloads with material multi-draft acceptance. M1 tests the
interaction; it does not modify the adaptive controller.

## Provenance

```text
base commit: 9472c72a8f7d1fa0ed5b49b49afcae6528872c7b
branch: exp/radix-kv-sharing-m0
workload source: benchmark/radix_kv_sharing_m0/m05_workloads.json
prompt replay: one JSON prompt list per workload seed
experiment revision: exact git HEAD is captured by run_m1.sh
```

PR #31716 remains unchanged. The vLLM cost-model branch is outside this experiment.

## Fixed matrix

```text
workloads: code_completion, structured_json
layouts: shared, duplicated
K: 0, 1, 2, 3, 4, 5
batch size: 8
seeds/query subsets: 17, 29, 41
temperature: 0
output length: 128
ignore_eos: false
```

Each timing order therefore requires exactly 72 unique cells:

```text
2 workloads × 2 layouts × 6 K × 3 seeds = 72
```

The footprint pass requires the same 72 logical cells. A missing, duplicate, or
unexpected cell produces `M1_INCOMPLETE` before K* is computed.

Natural EOS termination is valid only when `ignore_eos=false`; the benchmark client
accepts `stop` and `length` finish reasons in that mode and continues to reject aborts.

## Timing definition

The timing process uses:

```text
SGLANG_RADIX_KV_M0_RECORD=spec_cycle_gpu_time
```

The primary per-cycle metric is:

```text
useful_tokens_per_spec_cycle_ms
  = (1 + mean accepted drafts per request)
    / (draft_gpu_ms + target_verify_gpu_ms)
```

K=0 records `draft_gpu_ms=0`. The recorder also retains `spec_cycle_gpu_ms`,
`draft_extend_gpu_ms`, and unattributed cycle time. Full-cycle efficiency is a
sensitivity result, not the primary K* definition.

Physical footprint is collected in separate server processes. Timing and footprint
collection must never be enabled together.

## Pair validity

For every `(workload, K, seed)` and execution order, the shared and duplicated arms
must have:

1. identical tokenized prompt fingerprints;
2. identical client seed and server provenance;
3. equal retained record counts;
4. exact per-step acceptance trajectories;
5. exact logical-context trajectories;
6. symmetric runtime batch sizes, with only symmetric underfilled tails excluded;
7. higher physical page reuse in the shared footprint arm.

Any failure produces `M1_INVALID` and suppresses the K* conclusion.

## Counterbalancing

`run_m1.sh` automates both K and layout order:

```text
forward: ascending K, shared -> duplicated
reverse: descending K, duplicated -> shared
footprint: ascending K, shared -> duplicated
```

Per-cell efficiency is the median of the forward and reverse measurements. The
runner records GPU telemetry, server logs, exact git revision, execution order, and
raw captures.

One-cell smoke:

```bash
M1_SMOKE_ONLY=1 \
RESULT_ROOT=results/radix-kv-sharing-m1-smoke \
benchmark/radix_kv_sharing_m0/run_m1.sh
```

Full run:

```bash
RESULT_ROOT=results/radix-kv-sharing-m1 \
benchmark/radix_kv_sharing_m0/run_m1.sh
```

## Decision rule

K* is computed independently for each seed and from the across-seed aggregate.
When shared and duplicated K* differ, both layouts must prefer their own K* over the
other layout's K* by at least `minimum_effect_percent`. At least two of three seeds
must reproduce the aggregate shift direction.

```text
M1_INCOMPLETE
  required cells are missing, duplicated, or unexpected

M1_INVALID
  paired prompt, acceptance, context, provenance, batch, or footprint control fails

M1_NO_INTERACTION
  aggregate shared and duplicated K* are equal

M1_K_STAR_SHIFT_UNPOWERED
  argmax differs but the effect floor or 2/3-seed reproduction is not met

M1_K_STAR_SHIFT
  aggregate shift, bidirectional effect floor, and seed reproduction all pass
```

The analyzer additionally reports:

```text
Delta_sharing(K) = efficiency_shared(K) / efficiency_duplicated(K) - 1
```

Only `M1_K_STAR_SHIFT` permits evaluation of incremental oracle value beyond the
existing BS × context policy. It still does not directly authorize a controller or
an upstream PR.

## Current state

The full H100 NVL run completed on 2026-08-21 at experiment commit
`d05123e75c65fc940932042ad667b7b6ba941e91`. All 72 forward timing cells, 72 reverse
timing cells, and 72 footprint cells were present; all paired controls passed.

The outcome was `M1_NO_INTERACTION`. Code completion selected K=2 in both layouts,
and structured JSON selected K=3 in both layouts. Seed-level K* values matched across
layouts, and the largest absolute interaction was 0.628%, below the 2% effect floor.
See `M1_RESULT_20260821.md`. This closes the track without a controller or upstream
feature PR.
