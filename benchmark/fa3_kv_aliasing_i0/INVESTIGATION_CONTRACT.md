# I0 investigation contract: FA3 physical KV-page aliasing

## Research question

> Does physical KV-page aliasing itself cause the 12–28% FA3
> target-verification speedup on H100, and if so, through which memory-system
> mechanism?

This is a mechanism-localization investigation derived from the confirmed M0
observation. It is not an adaptive-speculative-decoding feature project.

## Prior evidence and boundary

M0 measured a 12.55–27.61% reduction in isolated target-verification CUDA time
for shared versus duplicated Radix layouts at 8K/16K logical contexts. M1 found
no useful physical-sharing-by-K interaction in the natural acceptance-bearing
workloads and closed the Radix-aware K-controller hypothesis.

I0 asks whether the remaining M0 latency signal survives below the SGLang
scheduler, allocator, and Radix-tree orchestration layers.

## Fixed anchor

The first run preserves the strongest and simplest confirmed M0 workload corner
while adopting the latest official Qwen dense checkpoint's full-attention shape:

```text
GPU: H100
model-shape provenance: Qwen/Qwen3.8-27B
attention: SGLang FA3 paged KV
batch size: 16
logical KV length: 16,384 tokens
query length: 1 token
shared-prefix fraction: 90%
dtype: BF16
query heads: 24
KV heads: 4
head dimension: 256
page size: 1
causal: true
num_splits: 0
CUDA graph: disabled
```

The query length of one corresponds to the K=0 anchor. Speculative acceptance,
draft execution, and controller behavior are intentionally absent.

Qwen3.8-27B uses a hybrid stack with three Gated DeltaNet layers followed by one
full-attention layer. I0 covers only the full-attention FA3 path. It does not load
model weights and makes no whole-model latency claim.

## A/B/C layouts

All three layouts use the same allocated KV-pool shape, the same query tensor,
and the same logical K/V contents.

```text
A. shared / aliased
   Every request maps the shared prefix to the same physical KV pages.
   Request-specific suffix pages remain distinct.

B. duplicated-contiguous
   Every request owns a distinct contiguous physical-page range.
   Shared-prefix contents are copied into each request's range.

C. duplicated-scattered
   Every request owns distinct pages selected by a deterministic full-pool
   permutation. Consecutive logical pages are physically non-contiguous.
```

Only the page table and the locations receiving identical logical K/V contents
may differ. The backing allocation size and base tensor are identical across
layouts within a process.

## Required invariants

Before a latency conclusion is allowed, the artifact must show equality of:

```text
batch size
query length
logical KV length
dtype
Q/KV head counts
head dimension
page size
logical pages per request
FA implementation and callable
FA version
causal/window configuration
num_splits
softmax scale
varlen-query metadata
return-softmax-LSE setting
stream
kernel-call count per timed interval
query tensor identity
logical K/V contents
output within the preregistered BF16 tolerance
```

The initial output gate is `torch.allclose(rtol=2e-2, atol=2e-2)`. Maximum and
mean absolute differences are retained. This tolerance is a validity guard, not
a numerical-accuracy claim.

The Python call signature and all tensor shapes are recorded in Gate A. Kernel
name plus grid/block launch geometry must be captured in a separate, untimed
Nsight Systems or equivalent trace before the result advances beyond a latency
signal. A kernel-name match without launch-dimension evidence is insufficient.

## Measurement protocol

### Gate A: latency reproduction

No profiler is attached to primary timing.

```text
layout orders: all six permutations of A/B/C
seeds: 17, 29, 41
cache states: cold-ish, warm steady-state
warmup: excluded
primary timer: synchronized CUDA events around one FA3 call
raw samples: retained
GPU clocks, temperature, power, utilization: sampled every 500 ms
```

`cold-ish` means a preregistered buffer larger than H100 L2 is touched and
synchronized before every measured call. It is not called a guaranteed cold
cache. `warm` means repeated calls after excluded warmup calls.

Every seed runs every A/B/C permutation. This separates layout from execution
position and allocation/access order. Layout materialization, page-table copies,
L2-thrash work, output checks, and result serialization are outside the timed
interval.

The first H100 run uses a 2% minimum resolved effect. Replace it before execution
if a same-layout timing preflight demonstrates a larger floor.

### Gate B: HBM traffic

Run only if Gate A reproduces a counterbalanced A-versus-B latency signal.
Profiler collection is a separate process and must not be mixed with primary
timing. Record at minimum device-memory read bytes/sectors and throughput.

### Gate C: L2 behavior

Run only after Gate B. Record requested sectors, hit rate, misses reaching device
memory, and L2 throughput.

### Stall analysis

Run only when HBM/L2 measurements do not explain the latency delta. Do not begin
with a broad counter sweep.

## Gate A interpretation

Using the warm-state counterbalanced medians:

```text
A faster than B; B approximately C
  -> aliasing-dominant candidate

A approximately B; B faster than C
  -> physical-locality candidate

A faster than B; B faster than C
  -> mixed aliasing and locality candidate

A approximately B approximately C
  -> no isolated FA3 signal; return to the SGLang orchestration boundary
```

`approximately` means the absolute difference is below the configured minimum
effect. A latency signal is not a confirmed mechanism. Confirmation additionally
requires identical empirical kernel launch geometry and a memory-system metric
that moves consistently with the latency result.

## Result vocabulary

```text
I0_INCOMPLETE
  required orders, layouts, states, seeds, or samples are missing

I0_INVALID
  logical-input, output, signature, provenance, or timing controls fail

I0_ORDER_SENSITIVE
  aggregate signal exceeds the floor but fewer than two thirds of
  seed/order strata reproduce it

I0_UNEXPECTED_DIRECTION
  A is materially slower than B or B is materially slower than C; do not map
  the result onto the preregistered aliasing/locality interpretations

I0_NO_ISOLATED_SIGNAL
  A/B/C differences remain below the resolved-effect floor

I0_ALIASING_CANDIDATE
  A is faster than B with order support; B and C are unresolved

I0_LOCALITY_CANDIDATE
  B is faster than C with order support; A and B are unresolved

I0_MIXED_CANDIDATE
  both A-versus-B and B-versus-C effects pass
```

`CONFIRMED`, `PARTIAL`, or `REJECTED` are reserved for the completed
latency-plus-profiler investigation, not Gate A alone.

## Explicitly out of scope

```text
NO scheduler
NO adaptive K
NO new controller
NO cache-aware batching
NO production SGLang feature PR
NO assumption that unique physical pages equal HBM bytes
NO Nsight profiling before Gate A passes
```

If isolated FA3 does not reproduce the signal, stop this branch and localize the
effect upward through SGLang metadata construction, allocation, CUDA-graph, and
request-ordering boundaries. Do not tune the reproducer until a positive result
appears.

## Provenance

```text
source experiment branch: exp/radix-kv-sharing-m0
source result commit: a35ad33d
investigation branch: investigate/fa3-kv-aliasing-i0
M0 result: benchmark/radix_kv_sharing_m0/M0_CONFIRMATION_20260813.md
M1 result: benchmark/radix_kv_sharing_m0/M1_RESULT_20260821.md
```

The exact execution commit, runtime versions, GPU identity, configuration, raw
samples, and page-table fingerprints must be stored with every hardware artifact.
