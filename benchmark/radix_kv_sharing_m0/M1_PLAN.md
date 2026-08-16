# M1 paired-layout K sweep — build plan (desk, 2026-08-17)

Grounded in a live read of the M0/M0.5 code. **No code committed yet** — this is
the build spec + the single integration seam M1 needs. M0.5 authorized exactly two
workloads (`code_completion`, `structured_json`); everything else here reuses the
existing, already-tested machinery.

## Question (unchanged from README)

> Does physical KV sharing move the **optimal fixed K** for a workload that actually
> accepts multi-draft speculation?

M0 answered "physical sharing changes verify-time" on a *non-accepting* synthetic
workload (K=4 had 20 non-zero decisions / 73,708). M0.5 found two workloads that DO
accept to full K=4 in every seed. M1 = M0's paired-layout timing machine driven by
M0.5's real prompts, swept over K=0..5.

## What already exists (reuse verbatim)

| Piece | File | Role in M1 |
|---|---|---|
| `load_workloads` / `select_prompts` | `radix_kv_sharing_m05.py` | deterministic real prompts (shared prefix + seeded distinct queries) |
| `Cell`, `build_cells` | `radix_kv_sharing_m0.py` | cell identity + matrix (parameterized: `DEFAULT_STEPS`) |
| `run_cell` | `radix_kv_sharing_m0.py` | launches client, reads recorder, writes capture |
| `analyze_capture_pairs` | `radix_kv_sharing_m0.py` | **the 5 hard controls — unchanged** |
| observability recorder | `radix_kv_m0_observability.py` | `target_verify_gpu_time` / `kv_footprint` records |

The analyzer's capture contract (verified): each capture needs
`cell{batch_size,context_length,speculative_num_steps,seed,layout}`,
`recorder{components:[one], records:[{batch_size, correct_drafts_per_req,
logical_context_lengths, target_verify_gpu_ms}]}`, `input_fingerprints`,
`client_seed`, `harness_git_head`, `server{...provenance...}`. `run_cell` already
emits all of this — **only the prompt source differs for M1.**

## The single integration seam

`run_cell` drives `one_batch_server` with `--dataset-name generated-shared-prefix`
(synthetic). `one_batch_server.dataset_name` choices are
`{mmmu, random, random-ids, generated-shared-prefix}` — no real-workload path, and
`fixed_prompt_file` is a *single* prompt (M1 needs `batch_size` distinct queries over
one shared prefix). So M1 needs exactly one new prompt source that reuses
`select_prompts`.

**Chosen approach (minimal, keeps `run_cell`/analyzer untouched):** add a
`workload-file` dataset to `one_batch_server` that, given `--workload-file` +
`--workload-id` + `--seed` + `--batch-size`, calls the *same* `select_prompts`
logic M0.5 used, tokenizes each rendered prompt, and reports `input_fingerprints`
identically to the existing path. Then M1's runner is a thin wrapper over
`build_cells` + `run_cell` with `dataset_name="workload-file"`.

- Rejected alt: a bespoke M1 client posting to `/generate` (like M0.5's `_post_json`).
  It would have to re-derive `input_fingerprints` + client_result shape that
  `run_cell`/analyzer expect → more surface, more drift risk.
- `select_prompts` is the shared truth for "same prompts as the M0.5 acceptance
  screen," which is what makes M1's acceptance trajectory match M0.5 by construction.

## M1 cell matrix

```text
workloads:  code_completion (primary), structured_json (secondary)   # M0.5-authorized only
layouts:    shared, duplicated                                        # duplicated = --disable-radix-cache
K:          0, 1, 2, 3, 4, 5
batch size: 8            (M0.5 established acceptance at bs=8; hold constant for the K curve)
context:    the workload's natural rendered length (not synthetic 8192/16384)
seeds:      17, 29, 41
temperature/output length: identical to M0.5 (0 / 128) across layouts
```
`instruction` = reserve; `short_qa` excluded (M0.5: never full K=4, 77–80 decisions).
Two-pass as in M0: `target_verify_gpu_time` (timing) + `kv_footprint` (accounting,
16-token pass) so the footprint hard control still applies.

## Hard controls (inherited, no change)

M1 reuses `analyze_capture_pairs` as-is. The five controls (prompt-SHA fingerprint
match across layouts, verify-record count match, per-step `correct_drafts_per_req`
match, per-step `logical_context_lengths` match, runtime-batch equality with
symmetric-underfill exclusion) + footprint-reuse ordering all still gate the speedup.
`m1_authorized`-style gating stays: **M1 emits a K-curve, it does not by itself
authorize a controller or a feature PR** (README boundary).

## Decision rule

M1 answers "does K\* (argmax useful-progress-per-verify-time) differ between shared
and duplicated?" Output = per-(workload, layout) curve of verify-time vs K and the
implied K\*. A *shift* in K\* between layouts (reproduced across all 3 seeds, past the
measurement floor, all hard controls green) is the only thing that would motivate the
next milestone. No shift / below floor / not reproduced → stop, record, do not build
the controller.

## Build checklist (desk, then one rental)

1. **Desk — DONE (2026-08-17):** implemented.
   - `one_batch_server`: generic `--prompt-list-file` (JSON list of exactly
     `batch_size` prompt strings, tokenized as-is). Decoupled — no experiment import.
   - `radix_kv_sharing_m0.run_cell`: optional `prompt_list_file` param; when set the
     client command replaces the gsp dataset args with `--prompt-list-file`.
     Backward-compatible (default `""` preserves M0 behavior; M0 tests untouched).
   - `radix_kv_sharing_m1.py`: `build_m1_cells` (K0..5 × layouts × seeds, bs=8),
     `write_plan` (rejects non-authorized workloads, renders one prompt-list per seed
     via `select_prompts`), `run_matching_m1_cells` (selects by server layout/K, replays
     the seed's prompt-list through `run_cell`), `summarize_k_curve` (per-layout
     `useful_progress/verify_ms` → K\* + shift flag), `analyze_m1` (reuses
     `analyze_capture_pairs` hard controls + the K\* curve). CLI `plan/run-matching/analyze`.
   - Tests: `test/registered/unit/spec/test_radix_kv_sharing_m1.py` — cell-matrix
     completeness, unauthorized-workload rejection, K\*-shift derivation (+ no-shift
     negative), prompt-list wiring. `py_compile` clean; `summarize_k_curve` logic
     verified standalone. **Full pytest needs the SGLang env (Linux) — unrun locally.**
2. **Style:** new containers = `msgspec.Struct` (repo rule `no-dataclasses`), kw-only
   args, small functions. Do **not** touch `python/sglang/srt/speculative/*` (would
   trigger the `speculative-naming` skill) — M1 is benchmark-only.
3. **Rental (user go):** single H100 NVL, ~$1 (M0.5 was $0.72 / 17 min). Smoke one
   cell first (shared vs duplicated footprints distinct + hard controls pass), then
   the full 2×6×2×3 timing matrix + footprint pass. Teardown, archive with SHA-256.

## Custody / provenance

Branch `exp/radix-kv-sharing-m0` (`9472c72a`). PR #31716 SHA untouched. Controller
and upstream PR remain blocked until M1 shows a real K\* shift.
