# M0 counterbalanced confirmation — 2026-08-13

## Status

```text
M0_CONFIRMED
M1_NOT_STARTED
NO_CONTROLLER_OR_UPSTREAM_FEATURE_PR
```

Physical KV sharing changed target-verification CUDA time at fixed batch size,
logical context, prompt set, acceptance trajectory, and speculative depth. The
effect survived both server restart orders and all three seeds. This completes M0
only; it does not establish an optimal speculative-depth shift.

## Experiment contract

```text
GPU: NVIDIA H100 NVL 96 GB
Driver: 580.82.09
PyTorch: 2.13.0+cu130
CUDA runtime: 13.0
FlashInfer: 0.6.17
Target/draft attention backend: FA3
Experiment commit: 2ce83da8b027b0023763f9c1e7bcc3e32e20a2ea

BS: 8, 16
Logical context: 8192, 16384
Fixed K: 0, 4
Seeds: 17, 29, 41
Per-K restart order: shared -> duplicated -> duplicated -> shared
Primary interval: target_verify_gpu_time
Minimum resolved effect: 2.0%
```

The target was `meta-llama/Llama-3.1-8B-Instruct`; the draft model was
`lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`.

## Result

Every aggregate cell was `M0_CONFIRMED`.

| BS | Logical context | K | Counterbalanced shared-layout speedup | Across-seed range |
|---:|---:|---:|---:|---:|
| 8 | 8192 | 0 | 12.77% | 12.68–12.83% |
| 8 | 8192 | 4 | 12.71% | 12.55–12.77% |
| 8 | 16384 | 0 | 19.52% | 19.31–19.61% |
| 8 | 16384 | 4 | 19.51% | 19.48–19.67% |
| 16 | 8192 | 0 | 18.04% | 17.74–18.43% |
| 16 | 8192 | 4 | 19.28% | 18.77–19.36% |
| 16 | 16384 | 0 | 27.45% | 27.03–27.61% |
| 16 | 16384 | 4 | 26.65% | 26.44–26.90% |

Audit summary:

```text
aggregate cells: 8/8 M0_CONFIRMED
seed-level rows: 24/24 controls valid
same direction in both orders: 24/24
both orders above 2%: 24/24
counterbalanced speedup: 12.55% minimum, 19.34% median, 27.61% maximum
maximum forward/reverse order gap: 0.99 percentage points
timing captures: 96
raw recorder rows: 12,288
```

The prior full screen separately established physical-layout separation:
shared-page reuse was approximately 78.6–84.3%, while the duplicated layout was
0%. The confirmation intentionally reused that proof and isolated restart-order
bias with timing-only server pairs.

## Interpretation boundary

The allowed conclusion is:

> Radix physical KV sharing materially changes target-verification CUDA time for
> this controlled shared-prefix workload, beyond fixed server restart order.

The following conclusions are not supported:

- physical KV state changes the optimal speculative depth;
- a Radix-aware controller improves TPOT or end-to-end throughput;
- unique physical page count is a direct HBM-traffic measurement;
- this experiment should be added to PR #31716;
- a production SGLang feature PR is ready.

The effect is broadly similar at K=0 and K=4. More importantly, the K=4 captures
contained only 20 non-zero correct-draft decisions among 73,708 request decisions
(0.027%). The shared and duplicated trajectories matched, so M0 remains valid, but
this workload has insufficient acceptance to identify an optimal K.

## Measurement provenance

The instance ran for approximately 53.48 minutes at $2.5184/hour, for an estimated
$2.24 instance cost. GPU telemetry retained 5,647 samples; among samples with nonzero
GPU utilization, SM clock ranged from 780 to 1785 MHz, temperature from 35 to 61 C,
and peak power was 426.91 W. Counterbalancing, rather than a clock-lock claim, is the
primary restart-order control.

All 258 artifacts were copied locally. The compressed transfer checksum was:

```text
f7bfe34eb312b069b70f0fdcdfce4810a98fe858d47dffe4a688e4a7f9d0e2fd
```

Vast.ai reported zero active instances after teardown.

## Next gate

M1 requires a semantically coherent shared-prefix workload that produces material
draft acceptance while preserving identical paired prompts and acceptance
trajectories. Only then should K be swept to test whether physical sharing changes
the optimal speculative depth. A controller and an upstream feature PR remain
blocked until M1 and the incremental-oracle comparison both pass.

Raw artifacts are stored outside the SGLang source tree at:

```text
results/radix_kv_sharing_m0/h100_nvl_confirmation_20260813
```
