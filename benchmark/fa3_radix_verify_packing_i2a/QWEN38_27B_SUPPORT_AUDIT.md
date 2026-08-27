# Qwen3.8-27B SGLang support audit

Audit date: 2026-08-28

Upstream reference: `sgl-project/sglang@20a491d1d`

Official checkpoint: `Qwen/Qwen3.8-27B`

## Decision

Do not open a new model-support PR. Core Qwen3.8-27B support and the deployment
cookbook are already merged upstream. Any contribution from this branch must be
one of the following:

1. a regression test for an uncovered, demonstrated failure mode;
2. a focused fix for an open issue after checking ownership and linked PRs; or
3. a measured FA3/Radix target-verification improvement whose mechanism and
   end-to-end benefit have passed the investigation gates.

The current FA3 aliasing and verify-packing work is not a replacement model
implementation. It uses the checkpoint's full-attention geometry while keeping
the upstream Qwen3.5-family model path unchanged.

## Official model anchor

The official model card and checkpoint establish the following serving-relevant
properties:

```text
model: Qwen/Qwen3.8-27B
type: dense hybrid GDN vision-language model
parameters: 27B
layers: 64
layout: 16 x (3 x Gated DeltaNet + 1 x gated full attention)
full-attention Q heads: 24
full-attention KV heads: 4
full-attention head dimension: 256
hidden size: 5120
FFN intermediate size: 17408
native context: 262144
MTP: in-checkpoint, multi-step trained
```

This confirms that the investigation's `24 Q / 4 KV / head_dim 256` shape is
correctly attributed to Qwen3.8-27B.

## Work already merged upstream

### Core implementation

- PR #34859, `Qwen3.8-27B Model Support`, merged 2026-08-19.
- Reuses the Qwen3.5-family model implementation and adds the dense 27B-specific
  non-power-of-two GDN V/K head ratio of 3.
- Adds Hopper BF16 GEMV support and SM120 FP8/NVFP4 paths used by the official
  deployment recipes.
- Updates GDN prefill, quantization, DSpark draft logits, and speculative draft
  sampling paths.
- Adds exact CUDA coverage for fused GDN head ratios, including ratio 3, and a
  Hopper BF16 GEMV test.

### Documentation and deployment recipes

- PR #34860 adds the Qwen3.8-27B cookbook and BF16/FP8/NVFP4 deployment grid.
- PRs #35064 and #35065 fix speculative Mamba-state sizing and rework the grid.
- PR #35663 adds DFlash2 recipes.
- PR #35825 remeasures RTX 5090, RTX PRO 6000, and DGX Spark cells.
- PR #36020 separates NVFP4 variants by LM-head precision.

The cookbook already covers H200, GB300, RTX PRO 6000, RTX 5090, and DGX
Spark, plus native MTP/EAGLE, DSpark, and DFlash2 recipes. Do not duplicate
these launch recipes or documentation grids.

## Overlap audit for current work

No open or merged PR was found that performs the exact current investigation:

```text
H100 + FA3
same logical KV and kernel geometry
physical KV-page aliasing A/B
memory-counter localization
then Radix-local target-verify row packing without changing batch membership
```

The following work is adjacent and must be rechecked before any upstream PR:

| Area | Existing work | Boundary for this branch |
| --- | --- | --- |
| Qwen3.8 model enablement | #34859 merged | Do not implement another model class or registry entry. |
| GDN target-verify parity | #35541 and #36014 open | Do not modify recurrent-state arithmetic without coordinating with these authors. |
| GDN target-verify materialization | #33778 open | A performance PR must show it does not duplicate this projection/materialization work. |
| FA3 speculative context bound | #35985 open | Do not submit another page-table headroom fix. Rebase and include this invariant in tests. |
| Multi-request extend warmup | #36050 open | Distinguish warmup/capture effects from physical aliasing and row order. |
| Native MTP pool allocation | #35567 open | Do not change MTP embed/head binding or pool sizing here. |
| Speculative quantized graph paths | #36038, #36045, #36077 open | Initial investigation remains BF16 on H100; quantized support is a separate follow-up. |
| Radix/spec cache correctness | #35694 and #32170 open | Row packing must preserve committed lengths and cache keys exactly. |

## Open issue ownership audit

The following issues are not free tasks merely because they remain open:

- #35242, Qwen3.8 tool-call schema error, is linked to PR #35631.
- #35148, Rust gateway reasoning parsing, is claimed and linked to PRs #35249
  and #35346.
- #35150, DSpark forced-reject recurrent-state drift, is claimed and linked to
  PRs #35541 and #36014.
- #35797, compressed-tensors MTP weights being dropped, is linked to PR #35887.
- #35985 is itself an active FA3 speculative page-table fix.

Do not start parallel fixes for these without first confirming that the linked
work is abandoned or asking the author/maintainer.

Potentially unowned gaps found during this audit:

- #35772: Qwen3.8/Qwen3-VL vision features diverge from Transformers and vLLM
  on fine-grained grounding. No linked fix was present at audit time. This is a
  high-value correctness issue but requires a separate multimodal reference
  harness and should not be mixed into the FA3 investigation.
- #35822: native MTP/EAGLE target verification hangs on Ampere TP=2. No linked
  fix was present at audit time. This is hardware-specific and cannot be closed
  with an H100-only run.

## Proposed non-duplicating work list

### P0. Keep the current investigation honest

- Keep the official model provenance as `Qwen/Qwen3.8-27B`.
- Record the exact upstream commit and checkpoint revision in every hardware
  receipt.
- Keep model weights optional for isolated FA3 I1/I2a; do not describe those
  microbenchmarks as whole-model validation.
- Rebase the eventual upstream change on current `origin/main` and rerun the
  overlap audit on the day the PR is opened.

### P1. Complete the H100 FA3 mechanism gates

- Run I1 only on a provider that exposes Nsight Compute performance counters.
- Require identical FA3 kernel name, grid, block, logical KV, dtype, and output
  between aliased and duplicated layouts.
- Measure latency first, then DRAM and L2 counters only if the aliasing signal
  reproduces.
- Run I2a only after I1 review authorizes it.
- Treat Radix-local row packing as a new optimization only if the same batch,
  same KV pages, same K, same outputs, and same acceptance decisions are
  preserved and the counterbalanced gain clears the preregistered floor.

### P2. Add official-checkpoint validation before claiming Qwen support impact

- Load `Qwen/Qwen3.8-27B` BF16 from current upstream main on H100.
- Record checkpoint revision, Transformers version, SGLang commit, CUDA,
  FlashInfer, FA3, and kernel provenance.
- Run a text-only greedy smoke test and one image-input smoke test.
- Compare fixed greedy outputs or logits against a trusted reference for a
  small frozen input set; model startup alone is not correctness evidence.
- Exercise no-speculation and in-checkpoint MTP/EAGLE separately.
- Exercise FA3 eager and CUDA-graph target verification separately.
- Include Radix cache miss and long shared-prefix hit cases without changing
  prompts or sampling.
- Confirm page-table capacity near the context boundary after #35985 lands or
  against that PR branch.

### P3. Decide whether a dedicated NVIDIA regression test is acceptable

Current upstream has kernel tests for Qwen3.8-specific shapes and extensive
cookbook measurements, but this audit found no registered NVIDIA integration
test named for the official Qwen3.8-27B checkpoint. Before writing one:

1. ask maintainers whether the checkpoint size and download cost fit an
   existing nightly suite;
2. prefer a focused frozen-input accuracy/smoke test over another benchmark
   table;
3. reuse an existing Qwen3.5-family test harness where architecture semantics
   are identical; and
4. keep BF16 baseline coverage separate from FP8/NVFP4 and speculative modes.

If maintainers do not want a 27B checkpoint in CI, keep the reproducible manual
receipt and add only a smaller unit-level regression for the demonstrated bug.

### P4. Issue-specific contribution path

- Select one unowned issue only after reproducing it on current main.
- Comment with the reproduction and claim the issue before implementing.
- Add the smallest failing test first.
- Submit one focused fix per failure mode.
- Do not bundle model support, parser behavior, multimodal correctness,
  speculative state parity, quantization, and FA3 performance into one PR.

## PR gates

An upstream PR is allowed only when all applicable checks pass:

```text
[ ] Current origin/main used for reproduction
[ ] No active overlapping PR or claimed issue
[ ] Official checkpoint revision pinned
[ ] Correctness reference and failure reproduced
[ ] Minimal regression test exists
[ ] Performance result is counterbalanced and includes raw artifacts
[ ] No model semantics, request membership, K, or acceptance change is hidden
[ ] Scope is one reviewable mechanism
```

Until those gates pass, this document is a backlog and overlap ledger, not a
promise to open a Qwen3.8-27B support PR.
