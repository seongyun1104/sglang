"""Acceptance-only M0.5 gate for the Radix KV-sharing experiment.

This module does not compare physical layouts or choose speculative depth. It screens
coherent shared-prefix workloads for enough multi-draft acceptance to justify M1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import urllib.request
from pathlib import Path
from typing import Any, Sequence


LLAMA3_PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{shared_user_prefix}"
)
LLAMA3_SUFFIX = (
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _get_json(url: str, *, timeout: int = 60) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def _post_json(
    url: str, payload: dict[str, Any] | None = None, *, timeout: int = 1800
) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST"  # noqa: S310
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode()


def _find_recorder(server_info: dict[str, Any]) -> dict[str, Any]:
    for state in server_info.get("internal_states", []):
        recorder = state.get("radix_kv_m0_record")
        if recorder is not None:
            return recorder
    raise RuntimeError(
        "server has no acceptance recorder; launch with "
        "SGLANG_RADIX_KV_M0_RECORD=acceptance"
    )


def load_workloads(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("workload schema_version must be 1")
    if document.get("chat_template") != "llama3":
        raise ValueError("M0.5 currently requires chat_template=llama3")
    workloads: dict[str, dict[str, Any]] = {}
    for workload in document.get("workloads", []):
        workload_id = str(workload.get("id", "")).strip()
        queries = workload.get("queries")
        if not workload_id or workload_id in workloads:
            raise ValueError(f"invalid or duplicate workload id: {workload_id!r}")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"workload {workload_id} has no queries")
        if len(set(queries)) != len(queries):
            raise ValueError(f"workload {workload_id} has duplicate queries")
        for key in ("system_prompt", "shared_user_prefix"):
            if not str(workload.get(key, "")).strip():
                raise ValueError(f"workload {workload_id} has no {key}")
        workloads[workload_id] = workload
    if not workloads:
        raise ValueError("workload file has no workloads")
    return workloads


def select_prompts(
    workload: dict[str, Any], *, batch_size: int, seed: int
) -> tuple[list[str], list[int]]:
    queries = workload["queries"]
    if batch_size <= 0 or batch_size > len(queries):
        raise ValueError(
            f"batch_size must be in [1, {len(queries)}], got {batch_size}"
        )
    indices = random.Random(seed).sample(range(len(queries)), batch_size)
    shared_prefix = LLAMA3_PREFIX.format(
        system_prompt=workload["system_prompt"],
        shared_user_prefix=workload["shared_user_prefix"],
    )
    prompts = [shared_prefix + queries[index] + LLAMA3_SUFFIX for index in indices]
    if any(not prompt.startswith(shared_prefix) for prompt in prompts):
        raise AssertionError("rendered prompts lost their exact shared prefix")
    return prompts, indices


def _validate_server(server_info: dict[str, Any]) -> tuple[dict[str, Any], int]:
    states = server_info.get("internal_states", [])
    if not states:
        raise RuntimeError("server_info has no internal_states")
    state = states[0]
    if bool(state.get("speculative_adaptive")):
        raise RuntimeError("M0.5 requires fixed speculative depth")
    if int(state.get("speculative_eagle_topk", -1)) != 1:
        raise RuntimeError("M0.5 requires speculative_eagle_topk=1")
    steps = int(state.get("speculative_num_steps", -1))
    if steps not in (2, 4):
        raise RuntimeError(f"M0.5 requires K=2 or K=4, server reports K={steps}")
    if int(state.get("speculative_num_draft_tokens", -1)) != steps + 1:
        raise RuntimeError("server draft-token slots do not match K + 1")
    recorder = _find_recorder(server_info)
    if recorder.get("components") != ["acceptance"]:
        raise RuntimeError(
            f"M0.5 requires acceptance-only recorder, got {recorder.get('components')}"
        )
    return state, steps


def run_workload(
    *,
    workload_file: Path,
    workload_id: str,
    batch_size: int,
    seed: int,
    output_length: int,
    base_url: str,
    output_dir: Path,
) -> Path:
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    workloads = load_workloads(workload_file)
    if workload_id not in workloads:
        raise ValueError(f"unknown workload: {workload_id}")
    prompts, query_indices = select_prompts(
        workloads[workload_id], batch_size=batch_size, seed=seed
    )

    base_url = base_url.rstrip("/")
    _post_json(base_url + "/flush_cache")
    before = _get_json(base_url + "/server_info")
    state, steps = _validate_server(before)
    if _find_recorder(before).get("records"):
        raise RuntimeError("acceptance recorder was not empty after cache flush")

    _post_json(
        base_url + "/generate",
        {
            "text": prompts,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_length,
                "ignore_eos": False,
            },
            "stream": False,
        },
    )
    after = _get_json(base_url + "/server_info")
    after_state, after_steps = _validate_server(after)
    if after_steps != steps:
        raise RuntimeError("server K changed during the workload")
    recorder = _find_recorder(after)
    if not recorder.get("records"):
        raise RuntimeError("acceptance recorder captured no verify records")

    prompt_hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
    file_hash = hashlib.sha256(workload_file.read_bytes()).hexdigest()
    capture = {
        "schema_version": 1,
        "gate": "M0.5_ACCEPTANCE_ONLY",
        "workload_file": str(workload_file),
        "workload_file_sha256": file_hash,
        "workload_id": workload_id,
        "query_indices": query_indices,
        "prompt_sha256": prompt_hashes,
        "batch_size": batch_size,
        "seed": seed,
        "output_length": output_length,
        "speculative_num_steps": steps,
        "server": {
            "version": after.get("version", after_state.get("version")),
            "model_path": after.get("model_path", after_state.get("model_path")),
            "speculative_draft_model_path": after.get(
                "speculative_draft_model_path",
                after_state.get("speculative_draft_model_path"),
            ),
            "attention_backend": after.get(
                "attention_backend", after_state.get("attention_backend")
            ),
            "dtype": after.get("dtype", after_state.get("dtype")),
            "disable_radix_cache": after_state.get("disable_radix_cache"),
        },
        "recorder": recorder,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{workload_id}-bs{batch_size}-k{steps}-seed{seed}-acceptance.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    return output


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze_acceptance_capture(capture: dict[str, Any]) -> dict[str, Any]:
    steps = int(capture["speculative_num_steps"])
    batch_size = int(capture["batch_size"])
    records = capture["recorder"].get("records", [])
    values: list[int] = []
    accepted_per_verify: list[int] = []
    runtime_batches: list[int] = []
    invalid_reasons: list[str] = []
    for record in records:
        acceptance = record.get("correct_drafts_per_req")
        if not isinstance(acceptance, list):
            invalid_reasons.append("missing_acceptance")
            continue
        parsed = [int(value) for value in acceptance]
        if any(value < 0 or value > steps for value in parsed):
            invalid_reasons.append("acceptance_out_of_range")
            continue
        if int(record.get("batch_size", -1)) != len(parsed):
            invalid_reasons.append("runtime_batch_acceptance_length_mismatch")
            continue
        values.extend(parsed)
        accepted_per_verify.append(sum(parsed))
        runtime_batches.append(len(parsed))

    if not records:
        invalid_reasons.append("no_verify_records")
    if not values:
        invalid_reasons.append("no_acceptance_decisions")
    invalid_reasons = sorted(set(invalid_reasons))
    histogram = {value: values.count(value) for value in range(steps + 1)}
    nonzero = sum(value > 0 for value in values)
    multidraft = sum(value >= 2 for value in values)
    full_depth = sum(value == steps for value in values)
    controls_valid = not invalid_reasons
    if not controls_valid:
        status = "INVALID_CONTROLS"
    elif multidraft == 0:
        status = "NO_MULTI_DRAFT_SUPPORT"
    else:
        status = "MULTI_DRAFT_OBSERVED_NEEDS_MATERIALITY_REVIEW"

    return {
        "workload_id": capture["workload_id"],
        "speculative_num_steps": steps,
        "seed": int(capture["seed"]),
        "configured_batch_size": batch_size,
        "verify_records": len(records),
        "acceptance_decisions": len(values),
        "controls_valid": controls_valid,
        "invalid_reason": ";".join(invalid_reasons),
        "runtime_batch_min": min(runtime_batches) if runtime_batches else 0,
        "runtime_batch_max": max(runtime_batches) if runtime_batches else 0,
        "nonzero_acceptance_decisions": nonzero,
        "nonzero_acceptance_ratio": nonzero / len(values) if values else float("nan"),
        "multi_draft_decisions": multidraft,
        "multi_draft_ratio": multidraft / len(values) if values else float("nan"),
        "full_depth_decisions": full_depth,
        "full_depth_ratio": full_depth / len(values) if values else float("nan"),
        "mean_accepted_drafts_per_request_verify": (
            statistics.mean(values) if values else float("nan")
        ),
        "mean_accepted_drafts_per_target_verify": (
            statistics.mean(accepted_per_verify)
            if accepted_per_verify
            else float("nan")
        ),
        "accepted_drafts_p50": _percentile(values, 0.5),
        "accepted_drafts_p75": _percentile(values, 0.75),
        "accepted_drafts_p95": _percentile(values, 0.95),
        "accepted_drafts_max": max(values) if values else -1,
        "accepted_drafts_histogram": json.dumps(histogram, sort_keys=True),
        "m0_5_screen_status": status,
        "m1_authorized": False,
    }


def summarize_workloads(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["workload_id"]), {}).setdefault(
            int(row["speculative_num_steps"]), []
        ).append(row)
    summaries: list[dict[str, Any]] = []
    for workload_id, by_k in sorted(grouped.items()):
        k2 = by_k.get(2, [])
        k4 = by_k.get(4, [])
        valid = bool(k2 and k4) and all(
            bool(row["controls_valid"]) for row in k2 + k4
        )
        k4_multidraft = sum(int(row["multi_draft_decisions"]) for row in k4)
        if not valid:
            status = "INCOMPLETE_OR_INVALID"
        elif k4_multidraft == 0:
            status = "REJECT_NO_K4_MULTI_DRAFT_SUPPORT"
        else:
            status = "CANDIDATE_REQUIRES_MATERIALITY_REVIEW"
        summaries.append(
            {
                "workload_id": workload_id,
                "k2_seed_rows": len(k2),
                "k4_seed_rows": len(k4),
                "controls_valid": valid,
                "k2_nonzero_ratio_median": (
                    statistics.median(
                        float(row["nonzero_acceptance_ratio"]) for row in k2
                    )
                    if k2
                    else float("nan")
                ),
                "k2_multi_draft_ratio_median": (
                    statistics.median(float(row["multi_draft_ratio"]) for row in k2)
                    if k2
                    else float("nan")
                ),
                "k4_nonzero_ratio_median": (
                    statistics.median(
                        float(row["nonzero_acceptance_ratio"]) for row in k4
                    )
                    if k4
                    else float("nan")
                ),
                "k4_multi_draft_ratio_median": (
                    statistics.median(float(row["multi_draft_ratio"]) for row in k4)
                    if k4
                    else float("nan")
                ),
                "k4_accepted_drafts_max": (
                    max(int(row["accepted_drafts_max"]) for row in k4) if k4 else -1
                ),
                "m0_5_workload_status": status,
                "m1_authorized": False,
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one coherent workload against fixed K")
    run.add_argument("--workloads", type=Path, required=True)
    run.add_argument("--workload-id", required=True)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output-length", type=int, default=128)
    run.add_argument("--base-url", default="http://127.0.0.1:30000")
    run.add_argument("--output-dir", type=Path, required=True)
    analyze = commands.add_parser("analyze", help="summarize acceptance captures")
    analyze.add_argument("--captures", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        print(
            run_workload(
                workload_file=args.workloads,
                workload_id=args.workload_id,
                batch_size=args.batch_size,
                seed=args.seed,
                output_length=args.output_length,
                base_url=args.base_url,
                output_dir=args.output_dir,
            )
        )
        return

    captures = [
        json.loads(path.read_text())
        for directory in args.captures
        for path in sorted(directory.glob("*-acceptance.json"))
    ]
    rows = [analyze_acceptance_capture(capture) for capture in captures]
    _write_csv(args.output, rows)
    _write_csv(args.summary_output, summarize_workloads(rows))
    print(args.output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
