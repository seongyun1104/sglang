# M0.5 acceptance gate

## Question

> Can the target/draft pair produce material speculative acceptance on a coherent
> shared-prefix workload before physical layout is added as an experimental axis?

M0.5 uses one layout, fixed K, and no latency comparison. It screens four natural
workload classes: short QA, instruction following, Python completion, and structured
JSON generation. Every prompt has the same system/few-shot prefix within a workload
and a distinct query suffix.

## Matrix

```text
K: 2, 4
batch size: 8
workloads: short_qa, instruction, code_completion, structured_json
query subsets: seeds 17, 29, 41
output length: 128
temperature: 0
layout: normal Radix sharing only
```

Run:

```bash
RESULT_ROOT=results/radix-kv-sharing-m05-acceptance \
benchmark/radix_kv_sharing_m0/run_acceptance_screen.sh
```

Outputs:

```text
acceptance-by-seed.csv
workload-screen.csv
raw acceptance captures
server and environment metadata
```

The analyzer reports non-zero acceptance ratio, mean accepted drafts per request and
per target verify, the accepted-draft histogram, multi-draft ratio, full-depth ratio,
and P50/P75/P95/max.

## Decision rule

No arbitrary acceptance percentage is embedded in the code.

- `REJECT_NO_K4_MULTI_DRAFT_SUPPORT`: K=4 never accepts two or more drafts. Stop for
  that workload.
- `CANDIDATE_REQUIRES_MATERIALITY_REVIEW`: multi-draft acceptance exists. Inspect the
  full distribution and determine whether it gives M1 enough effective-work
  separation relative to the timing MDE.
- `INCOMPLETE_OR_INVALID`: required K or capture controls are missing.

The analyzer always emits `m1_authorized=false`. M0.5 data must be reviewed before a
paired physical-layout K sweep is authorized. This prevents a rare one-token accept
from being mistaken for meaningful K=4 operation.

## Scope boundary

M0.5 does not compare shared and duplicated layouts, measure a K optimum, modify the
adaptive controller, or justify an upstream PR. If a workload qualifies, M1 reuses
the exact workload file and prompt selection under paired layouts for K=0..5.
