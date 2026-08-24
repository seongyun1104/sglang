# I2a contract: Radix-local target-verify row order

## Question

> With physical KV pages, request membership, speculative depth, and logical
> attention held fixed, does Radix-local request-row order reduce FA3
> target-verification latency?

I2a is a falsifier for execution order. It does not change batch composition or
increase physical aliasing.

## Semantic anchor

The target shape follows the full-attention layers of `Qwen/Qwen3.8-27B`:

```text
GPU: H100
batch size: 16
Radix groups: 4 groups x 4 requests
committed logical context: 8K and 16K
speculative depth K: 1, 2, 4
target-verify query width: K + 1
Q/KV heads: 24/4
head dimension: 256
dtype: BF16
page size: 1
shared-prefix ratio within each group: 90%
FA: flash_attn_with_kvcache, ver=3, causal
```

For top-k=1 EAGLE, SGLang requires
`speculative_num_draft_tokens == speculative_num_steps + 1`. The candidate K/V
pages are materialized in the paged cache before the timed call. Causal FA3 then
reads a target KV length of `context + K + 1`. The 8K/16K labels never include
the candidate window. Merely increasing query length without candidate K/V is
prohibited.

## Arms

```text
clustered:
  G0-R0 G0-R1 G0-R2 G0-R3 G1-R0 ...

interleaved:
  G0-R0 G1-R0 G2-R0 G3-R0 G0-R1 ...

random:
  deterministic full permutation per seed
```

Every arm uses the same query rows, physical page IDs, backing K/V tensors,
unique-page count, alias ratio, cache lengths, and kernel call. Query and page-table
rows are permuted together. Outputs are restored to canonical request order before
comparison and must be bit-identical.

## Measurement

```text
seeds: 17, 29, 41
arm order: all six permutations
warmup: excluded per arm
timer: CUDA events around one FA3 call
primary anchor: context 16K, K=4
preregistered minimum effect: 2%
provider-specific noise: P95 pairwise relative difference among identical-arm block medians
resolved effect floor: max(2%, provider-specific noise)
required support: at least 12/18 seed/order strata
full default matrix: 16,200 timed samples and 36 output-equivalence checks
```

The 8K and K=1/2 cells are mechanism diagnostics. They cannot rescue a failed
primary anchor.

## Result vocabulary

```text
I2A_INCOMPLETE
I2A_INVALID
I2A_NO_ROW_ORDER_SIGNAL
I2A_UNPOWERED
I2A_ORDER_SENSITIVE
I2A_ROW_ORDER_SIGNAL
```

`I2A_UNPOWERED` means the complete, valid run had a provider-specific same-arm
noise floor above 2%, and the primary effect did not clear that measured floor.
It is not evidence that row order has no effect.

Only `I2A_ROW_ORDER_SIGNAL` permits an actual EAGLE target-verify replay. It does
not permit a scheduler, production implementation, or upstream PR.

## Explicit exclusions

```text
NO request admission changes
NO K controller
NO acceptance changes
NO cache placement changes
NO production permutation code
NO end-to-end speedup claim
```
