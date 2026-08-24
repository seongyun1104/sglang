# I1 contract: FA3 physical-aliasing memory mechanism

## Question

> Does the isolated A-versus-B latency difference correspond to reduced DRAM
> traffic and increased L2 reuse when physical KV pages are aliased?

I0 localized the causal input. I1 identifies the memory-system mechanism. It uses
the exact Qwen3.8 full-attention shape and compares only:

```text
A shared/aliased
B duplicated-contiguous
```

## Rental hard preflight

The first commands after SSH must establish all three capabilities:

```text
ncu --query-metrics succeeds
a small CUDA kernel produces a readable .ncu-rep
nvidia-smi application-clock lock succeeds
```

Any failure means immediate teardown. Container root without host counter access is
not sufficient. Exactly one H100 must be visible. The preflight records its UUID,
the requested clock, and the exact HBM counter set; paid runners reject a different
GPU, clock, or counter set. No anchor measurement is allowed after a failed
preflight.

## Sequence

```text
1. fixed-clock A/B latency reproduction using the I0 counterbalanced harness
2. fail closed unless the I0 status, effect floor, stratum support, output, sample,
   and exact anchor-shape gates all pass
3. untimed A and B output-digest equality
4. Gate B NCU capture: DRAM read bytes/sectors and throughput
5. analyze Gate B
6. only then Gate C: L2 requested sectors, hits/misses, throughput
7. stalls only if DRAM/L2 do not explain latency
```

The NCU A and B captures use separate processes with the same seed, tensor shapes,
logical values, backing allocation size, warmup count, FA3 specialization, and one
profiled main-kernel call. Output SHA-256 values must match.

## Interpretation

```text
latency down + DRAM reads down + L2 reuse up
  -> memory-working-set mechanism supported

latency down + DRAM reads approximately equal
  -> simple traffic model rejected; continue to translation/partition/stall analysis

latency not reproduced under fixed clocks
  -> stop; I0 provider-state sensitivity remains unresolved
```

No optimization or feature PR is authorized by I1 alone.
