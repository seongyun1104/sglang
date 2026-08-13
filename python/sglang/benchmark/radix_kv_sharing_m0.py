"""Harness for the Radix physical-KV-sharing M0 falsification experiment.

The harness intentionally does not implement an adaptive controller.  It creates the
fixed-K matrix, runs one cell against a pre-launched server, captures the opt-in
server-side recorder, and compares shared versus duplicated physical layouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_BATCH_SIZES = (8, 16)
DEFAULT_CONTEXT_LENGTHS = (8192, 16384)
DEFAULT_LAYOUTS = ("shared", "duplicated")
DEFAULT_STEPS = (0, 2, 4)
DEFAULT_SEEDS = (17, 29, 41)
RESEARCH_QUESTION = (
    "Does physical KV sharing in RadixAttention change speculative verification "
    "cost enough to shift the optimal speculative depth beyond what batch size "
    "and logical context length explain?"
)


@dataclass(frozen=True)
class Cell:
    batch_size: int
    context_length: int
    layout: str
    speculative_num_steps: int
    seed: int
    system_prompt_length: int
    question_length: int
    output_length: int

    @property
    def cell_id(self) -> str:
        return (
            f"bs{self.batch_size}-ctx{self.context_length}-"
            f"{self.layout}-k{self.speculative_num_steps}-seed{self.seed}"
        )


def build_cells(
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    layouts: Sequence[str] = DEFAULT_LAYOUTS,
    steps: Sequence[int] = DEFAULT_STEPS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    shared_prefix_ratio: float = 0.9,
    output_length: int = 128,
) -> list[Cell]:
    if not 0.0 < shared_prefix_ratio < 1.0:
        raise ValueError("shared_prefix_ratio must be between 0 and 1")
    unknown_layouts = set(layouts) - set(DEFAULT_LAYOUTS)
    if unknown_layouts:
        raise ValueError(f"unknown layouts: {sorted(unknown_layouts)}")
    cells: list[Cell] = []
    for batch_size in batch_sizes:
        for context_length in context_lengths:
            system_len = int(context_length * shared_prefix_ratio)
            question_len = context_length - system_len
            for layout in layouts:
                for speculative_num_steps in steps:
                    for seed in seeds:
                        cells.append(
                            Cell(
                                batch_size=int(batch_size),
                                context_length=int(context_length),
                                layout=layout,
                                speculative_num_steps=int(speculative_num_steps),
                                seed=int(seed),
                                system_prompt_length=system_len,
                                question_length=question_len,
                                output_length=int(output_length),
                            )
                        )
    return cells


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return json.load(response)


def _find_recorder(server_info: dict[str, Any]) -> dict[str, Any]:
    for state in server_info.get("internal_states", []):
        record = state.get("radix_kv_m0_record")
        if record is not None:
            return record
    raise RuntimeError(
        "server_info has no radix_kv_m0_record; launch with "
        "SGLANG_RADIX_KV_M0_RECORD=target_verify_gpu_time or kv_footprint"
    )


def _git_head() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_server(cell: Cell, server_info: dict[str, Any]) -> None:
    states = server_info.get("internal_states", [])
    if not states:
        raise RuntimeError("server_info has no internal_states")
    state = states[0]
    if bool(state.get("speculative_adaptive")):
        raise RuntimeError("M0 requires a fixed K; disable speculative_adaptive")
    if int(state.get("speculative_eagle_topk", -1)) != 1:
        raise RuntimeError("M0 requires speculative_eagle_topk=1")
    expected_disabled = cell.layout == "duplicated"
    if bool(state.get("disable_radix_cache")) != expected_disabled:
        raise RuntimeError(
            f"layout={cell.layout} requires disable_radix_cache={expected_disabled}, "
            f"server reports {state.get('disable_radix_cache')}"
        )
    actual_steps = int(state.get("speculative_num_steps", -1))
    if actual_steps != cell.speculative_num_steps:
        raise RuntimeError(
            f"cell K={cell.speculative_num_steps}, server K={actual_steps}"
        )
    actual_draft_tokens = int(state.get("speculative_num_draft_tokens", -1))
    expected_draft_tokens = cell.speculative_num_steps + 1
    if actual_draft_tokens != expected_draft_tokens:
        raise RuntimeError(
            f"cell expects {expected_draft_tokens} draft-token slots, "
            f"server reports {actual_draft_tokens}"
        )


def run_cell(
    *,
    cell: Cell,
    base_url: str,
    output_dir: Path,
    local_tokenizer_path: str = "",
    output_length_override: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    server_info_url = base_url.rstrip("/") + "/server_info"
    before = _get_json(server_info_url)
    _validate_server(cell, before)
    recorder_before = _find_recorder(before)
    internal_state_before = before["internal_states"][0]
    model_path = before.get("model_path", internal_state_before.get("model_path"))
    if not model_path:
        raise RuntimeError("server_info does not expose the required model_path")
    component_tag = "-".join(recorder_before["components"])
    output_length = (
        cell.output_length
        if output_length_override is None
        else int(output_length_override)
    )
    if output_length <= 0:
        raise ValueError("output length must be positive")

    result_path = output_dir / f"{cell.cell_id}-{component_tag}-client.jsonl"
    if result_path.exists():
        raise FileExistsError(
            f"refusing to append to existing cell artifact: {result_path}"
        )
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.one_batch_server",
        "--base-url",
        base_url,
        "--model-path",
        str(model_path),
        "--batch-size",
        str(cell.batch_size),
        "--input-len",
        str(cell.context_length),
        "--output-len",
        str(output_length),
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        "1",
        "--gsp-system-prompt-len",
        str(cell.system_prompt_length),
        "--gsp-question-len",
        str(cell.question_length),
        "--gsp-output-len",
        str(output_length),
        "--seed",
        str(cell.seed),
        "--skip-warmup",
        "--result-filename",
        str(result_path),
        "--no-append-to-github-summary",
    ]
    if local_tokenizer_path:
        command.extend(["--local-tokenizer-path", local_tokenizer_path])
    subprocess.run(command, check=True)

    client_lines = [
        json.loads(line)
        for line in result_path.read_text().splitlines()
        if line.strip()
    ]
    if len(client_lines) != 1:
        raise RuntimeError(
            f"expected one client result in {result_path}, got {len(client_lines)}"
        )
    client_result = client_lines[0]

    after = _get_json(server_info_url)
    _validate_server(cell, after)
    recorder = _find_recorder(after)
    internal_state = after["internal_states"][0]

    def server_value(name: str) -> Any:
        return after.get(name, internal_state.get(name))

    capture = {
        "schema_version": 1,
        "research_question": RESEARCH_QUESTION,
        "harness_git_head": _git_head(),
        "cell": asdict(cell),
        "cell_id": cell.cell_id,
        "client_result": str(result_path),
        "input_fingerprints": client_result.get("input_fingerprints"),
        "client_seed": client_result.get("seed"),
        "actual_output_length": output_length,
        "server": {
            "version": server_value("version"),
            "model_path": server_value("model_path"),
            "speculative_draft_model_path": server_value(
                "speculative_draft_model_path"
            ),
            "attention_backend": server_value("attention_backend"),
            "dtype": server_value("dtype"),
            "device": server_value("device"),
            "tp_size": server_value("tp_size"),
            "dp_size": server_value("dp_size"),
            "page_size": server_value("page_size"),
            "disable_cuda_graph": server_value("disable_cuda_graph"),
            "disable_radix_cache": server_value("disable_radix_cache"),
            "speculative_num_steps": server_value("speculative_num_steps"),
            "speculative_num_draft_tokens": server_value(
                "speculative_num_draft_tokens"
            ),
        },
        "recorder": recorder,
    }
    capture_path = output_dir / f"{cell.cell_id}-{component_tag}-capture.json"
    capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    return capture_path


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze_capture_pairs(
    captures: Iterable[dict[str, Any]],
    *,
    discard_first: int = 5,
    footprint_discard_first: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, int], dict[str, dict[str, dict[str, Any]]]] = {}
    for capture in captures:
        cell = capture["cell"]
        key = (
            int(cell["batch_size"]),
            int(cell["context_length"]),
            int(cell["speculative_num_steps"]),
            int(cell["seed"]),
        )
        components = capture["recorder"].get("components", [])
        if len(components) != 1:
            raise RuntimeError(
                f"capture {capture.get('cell_id')} must contain exactly one component"
            )
        component = components[0]
        by_component = grouped.setdefault(key, {}).setdefault(cell["layout"], {})
        if component in by_component:
            raise RuntimeError(
                f"duplicate capture for {key}, layout={cell['layout']}, "
                f"component={component}"
            )
        by_component[component] = capture

    rows: list[dict[str, Any]] = []
    for (batch_size, context_length, steps, seed), arms in sorted(grouped.items()):
        if set(arms) != set(DEFAULT_LAYOUTS):
            continue
        timing_arms = {
            layout: captures_by_component.get("target_verify_gpu_time")
            for layout, captures_by_component in arms.items()
        }
        if any(capture is None for capture in timing_arms.values()):
            continue
        shared_capture = timing_arms["shared"]
        duplicated_capture = timing_arms["duplicated"]
        assert shared_capture is not None and duplicated_capture is not None
        shared_records = shared_capture["recorder"]["records"][discard_first:]
        duplicated_records = duplicated_capture["recorder"]["records"][discard_first:]
        raw_pair_count = min(len(shared_records), len(duplicated_records))
        timing_speedups: list[float] = []
        acceptance_matches = 0
        context_matches = 0
        timing_underfilled_pairs_excluded = 0
        invalid_reasons: list[str] = []
        if any(
            "kv_footprint" not in captures_by_component
            for captures_by_component in arms.values()
        ):
            invalid_reasons.append("missing_kv_footprint_pass")
        all_captures = [
            capture
            for captures_by_component in arms.values()
            for capture in captures_by_component.values()
        ]
        prompt_fingerprints = [
            capture.get("input_fingerprints") for capture in all_captures
        ]
        if not prompt_fingerprints or any(
            value is None or value != prompt_fingerprints[0]
            for value in prompt_fingerprints
        ):
            invalid_reasons.append("prompt_fingerprint_mismatch")
        if any(int(capture.get("client_seed", -1)) != seed for capture in all_captures):
            invalid_reasons.append("client_seed_mismatch")
        git_heads = [capture.get("harness_git_head") for capture in all_captures]
        if not git_heads or any(
            head is None or head != git_heads[0] for head in git_heads
        ):
            invalid_reasons.append("harness_git_head_mismatch")
        server_keys = (
            "version",
            "model_path",
            "speculative_draft_model_path",
            "attention_backend",
            "dtype",
            "device",
            "tp_size",
            "dp_size",
            "page_size",
            "disable_cuda_graph",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
        )
        server_signatures = [
            tuple(capture["server"].get(key) for key in server_keys)
            for capture in all_captures
        ]
        if any(signature != server_signatures[0] for signature in server_signatures):
            invalid_reasons.append("server_provenance_mismatch")
        if len(shared_records) != len(duplicated_records):
            invalid_reasons.append("timing_record_count_mismatch")
        timing_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for shared, duplicated in zip(
            shared_records[:raw_pair_count],
            duplicated_records[:raw_pair_count],
            strict=True,
        ):
            shared_full = int(shared.get("batch_size", -1)) == batch_size
            duplicated_full = int(duplicated.get("batch_size", -1)) == batch_size
            if shared_full != duplicated_full:
                invalid_reasons.append("asymmetric_runtime_batch_size")
                continue
            if not shared_full:
                timing_underfilled_pairs_excluded += 1
                continue
            timing_pairs.append((shared, duplicated))
        pair_count = len(timing_pairs)
        for shared, duplicated in timing_pairs:
            if shared.get("correct_drafts_per_req") == duplicated.get(
                "correct_drafts_per_req"
            ):
                acceptance_matches += 1
            if shared.get("logical_context_lengths") == duplicated.get(
                "logical_context_lengths"
            ):
                context_matches += 1
            if (
                "target_verify_gpu_ms" in shared
                and "target_verify_gpu_ms" in duplicated
            ):
                baseline = float(duplicated["target_verify_gpu_ms"])
                if baseline > 0:
                    timing_speedups.append(
                        (baseline - float(shared["target_verify_gpu_ms"]))
                        / baseline
                        * 100.0
                    )
        if acceptance_matches != pair_count:
            invalid_reasons.append("acceptance_trajectory_mismatch")
        if context_matches != pair_count:
            invalid_reasons.append("logical_context_trajectory_mismatch")
        footprint_arms = {
            layout: captures_by_component.get("kv_footprint")
            for layout, captures_by_component in arms.items()
        }
        shared_page_reuse: list[float] = []
        duplicated_page_reuse: list[float] = []
        footprint_underfilled_pairs_excluded = 0
        if all(capture is not None for capture in footprint_arms.values()):
            shared_footprint = footprint_arms["shared"]
            duplicated_footprint = footprint_arms["duplicated"]
            assert shared_footprint is not None and duplicated_footprint is not None
            shared_footprint_records = shared_footprint["recorder"]["records"][
                footprint_discard_first:
            ]
            duplicated_footprint_records = duplicated_footprint["recorder"]["records"][
                footprint_discard_first:
            ]
            if len(shared_footprint_records) != len(duplicated_footprint_records):
                invalid_reasons.append("footprint_record_count_mismatch")
            footprint_pair_count = min(
                len(shared_footprint_records), len(duplicated_footprint_records)
            )
            footprint_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for shared, duplicated in zip(
                shared_footprint_records[:footprint_pair_count],
                duplicated_footprint_records[:footprint_pair_count],
                strict=True,
            ):
                shared_full = int(shared.get("batch_size", -1)) == batch_size
                duplicated_full = int(duplicated.get("batch_size", -1)) == batch_size
                if shared_full != duplicated_full:
                    invalid_reasons.append("asymmetric_footprint_runtime_batch_size")
                    continue
                if not shared_full:
                    footprint_underfilled_pairs_excluded += 1
                    continue
                footprint_pairs.append((shared, duplicated))
            if [
                shared.get("correct_drafts_per_req") for shared, _ in footprint_pairs
            ] != [
                duplicated.get("correct_drafts_per_req")
                for _, duplicated in footprint_pairs
            ]:
                invalid_reasons.append("footprint_acceptance_trajectory_mismatch")
            if [
                shared.get("logical_context_lengths") for shared, _ in footprint_pairs
            ] != [
                duplicated.get("logical_context_lengths")
                for _, duplicated in footprint_pairs
            ]:
                invalid_reasons.append("footprint_context_trajectory_mismatch")
            shared_page_reuse = [
                float(shared["page_reuse_ratio"]) for shared, _ in footprint_pairs
            ]
            duplicated_page_reuse = [
                float(duplicated["page_reuse_ratio"])
                for _, duplicated in footprint_pairs
            ]
        if shared_page_reuse and duplicated_page_reuse:
            if statistics.median(shared_page_reuse) <= statistics.median(
                duplicated_page_reuse
            ):
                invalid_reasons.append("physical_layout_not_distinct")
        controls_valid = pair_count > 0 and not invalid_reasons
        if not controls_valid:
            timing_speedups.clear()

        rows.append(
            {
                "batch_size": batch_size,
                "context_length": context_length,
                "speculative_num_steps": steps,
                "seed": seed,
                "paired_records": pair_count,
                "timing_underfilled_pairs_excluded": timing_underfilled_pairs_excluded,
                "footprint_underfilled_pairs_excluded": (
                    footprint_underfilled_pairs_excluded
                ),
                "controls_valid": controls_valid,
                "invalid_reason": ";".join(invalid_reasons),
                "acceptance_match_ratio": (
                    float("nan") if pair_count == 0 else acceptance_matches / pair_count
                ),
                "context_match_ratio": (
                    float("nan") if pair_count == 0 else context_matches / pair_count
                ),
                "runtime_batch_match_ratio": float("nan") if pair_count == 0 else 1.0,
                "sharing_speedup_p25_percent": _percentile(timing_speedups, 0.25),
                "sharing_speedup_median_percent": (
                    statistics.median(timing_speedups)
                    if timing_speedups
                    else float("nan")
                ),
                "sharing_speedup_p75_percent": _percentile(timing_speedups, 0.75),
                "shared_page_reuse_median": (
                    statistics.median(shared_page_reuse)
                    if shared_page_reuse
                    else float("nan")
                ),
                "duplicated_page_reuse_median": (
                    statistics.median(duplicated_page_reuse)
                    if duplicated_page_reuse
                    else float("nan")
                ),
            }
        )
    return rows


def aggregate_m0_rows(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_effect_percent: float,
    required_seeds: int = 3,
) -> list[dict[str, Any]]:
    if minimum_effect_percent <= 0:
        raise ValueError("minimum_effect_percent must be positive")
    if required_seeds <= 0:
        raise ValueError("required_seeds must be positive")
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["batch_size"]),
            int(row["context_length"]),
            int(row["speculative_num_steps"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (batch_size, context_length, steps), seed_rows in sorted(grouped.items()):
        valid = [row for row in seed_rows if bool(row["controls_valid"])]
        effects = [float(row["sharing_speedup_median_percent"]) for row in valid]
        positive = sum(effect > 0 for effect in effects)
        negative = sum(effect < 0 for effect in effects)
        same_direction = bool(effects) and (
            positive == len(effects) or negative == len(effects)
        )
        all_above_resolution = bool(effects) and all(
            abs(effect) >= minimum_effect_percent for effect in effects
        )
        if len(valid) < required_seeds:
            status = "INVALID_CONTROLS"
        elif not all_above_resolution:
            status = "BELOW_RESOLUTION"
        elif not same_direction:
            status = "MIXED_DIRECTION"
        else:
            status = "M0_SIGNAL"

        aggregates.append(
            {
                "batch_size": batch_size,
                "context_length": context_length,
                "speculative_num_steps": steps,
                "seed_rows": len(seed_rows),
                "valid_seed_rows": len(valid),
                "minimum_effect_percent": minimum_effect_percent,
                "sharing_speedup_across_seed_median_percent": (
                    statistics.median(effects) if effects else float("nan")
                ),
                "sharing_speedup_across_seed_min_percent": (
                    min(effects) if effects else float("nan")
                ),
                "sharing_speedup_across_seed_max_percent": (
                    max(effects) if effects else float("nan")
                ),
                "same_direction": same_direction,
                "all_seeds_above_resolution": all_above_resolution,
                "m0_status": status,
            }
        )
    return aggregates


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no complete shared/duplicated capture pairs found")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_captures(directories: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for directory in directories
        for path in sorted(directory.glob("*-capture.json"))
    ]


def _parse_cell(path: Path) -> Cell:
    data = json.loads(path.read_text())
    return Cell(**data)


def run_matching_cells(
    *,
    plan_dir: Path,
    base_url: str,
    output_dir: Path,
    local_tokenizer_path: str = "",
    output_length_override: int | None = None,
    skip_existing: bool = False,
) -> list[Path]:
    server_info = _get_json(base_url.rstrip("/") + "/server_info")
    state = server_info["internal_states"][0]
    layout = "duplicated" if bool(state.get("disable_radix_cache")) else "shared"
    steps = int(state.get("speculative_num_steps", -1))
    component_tag = "-".join(_find_recorder(server_info)["components"])
    selected = []
    for path in sorted(plan_dir.glob("bs*.json")):
        cell = _parse_cell(path)
        if cell.layout == layout and cell.speculative_num_steps == steps:
            selected.append(cell)
    if not selected:
        raise RuntimeError(
            f"no plan cells match server layout={layout}, K={steps} in {plan_dir}"
        )

    captures = []
    for cell in selected:
        capture_path = output_dir / f"{cell.cell_id}-{component_tag}-capture.json"
        if skip_existing and capture_path.exists():
            captures.append(capture_path)
            continue
        captures.append(
            run_cell(
                cell=cell,
                base_url=base_url,
                output_dir=output_dir,
                local_tokenizer_path=local_tokenizer_path,
                output_length_override=output_length_override,
            )
        )
    return captures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write the fixed M0 matrix")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    plan.add_argument(
        "--context-lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS
    )
    plan.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    plan.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    plan.add_argument("--shared-prefix-ratio", type=float, default=0.9)
    plan.add_argument("--output-length", type=int, default=128)

    run = subparsers.add_parser("run-cell", help="run and capture one matrix cell")
    run.add_argument("--cell", type=Path, required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:30000")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--local-tokenizer-path", default="")
    run.add_argument("--output-length-override", type=int)

    run_matching = subparsers.add_parser(
        "run-matching", help="run every plan cell matching the current server"
    )
    run_matching.add_argument("--plan-dir", type=Path, required=True)
    run_matching.add_argument("--base-url", default="http://127.0.0.1:30000")
    run_matching.add_argument("--output-dir", type=Path, required=True)
    run_matching.add_argument("--local-tokenizer-path", default="")
    run_matching.add_argument("--output-length-override", type=int)
    run_matching.add_argument("--skip-existing", action="store_true")

    analyze = subparsers.add_parser("analyze", help="compare paired layout captures")
    analyze.add_argument("--captures", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--aggregate-output", type=Path, required=True)
    analyze.add_argument("--discard-first", type=int, default=5)
    analyze.add_argument("--footprint-discard-first", type=int, default=1)
    analyze.add_argument("--minimum-effect-percent", type=float, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        cells = build_cells(
            batch_sizes=args.batch_sizes,
            context_lengths=args.context_lengths,
            steps=args.steps,
            seeds=args.seeds,
            shared_prefix_ratio=args.shared_prefix_ratio,
            output_length=args.output_length,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            (args.output / f"{cell.cell_id}.json").write_text(
                json.dumps(asdict(cell), indent=2, sort_keys=True) + "\n"
            )
        manifest = {
            "schema_version": 1,
            "research_question": RESEARCH_QUESTION,
            "harness_git_head": _git_head(),
            "num_cells": len(cells),
            "timing_env": "SGLANG_RADIX_KV_M0_RECORD=target_verify_gpu_time",
            "footprint_env": "SGLANG_RADIX_KV_M0_RECORD=kv_footprint",
            "cells": [cell.cell_id for cell in cells],
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    elif args.command == "run-cell":
        path = run_cell(
            cell=_parse_cell(args.cell),
            base_url=args.base_url,
            output_dir=args.output_dir,
            local_tokenizer_path=args.local_tokenizer_path,
            output_length_override=args.output_length_override,
        )
        print(path)
    elif args.command == "run-matching":
        paths = run_matching_cells(
            plan_dir=args.plan_dir,
            base_url=args.base_url,
            output_dir=args.output_dir,
            local_tokenizer_path=args.local_tokenizer_path,
            output_length_override=args.output_length_override,
            skip_existing=args.skip_existing,
        )
        for path in paths:
            print(path)
    else:
        rows = analyze_capture_pairs(
            _load_captures(args.captures),
            discard_first=args.discard_first,
            footprint_discard_first=args.footprint_discard_first,
        )
        _write_csv(args.output, rows)
        _write_csv(
            args.aggregate_output,
            aggregate_m0_rows(rows, minimum_effect_percent=args.minimum_effect_percent),
        )
        print(args.output)
        print(args.aggregate_output)


if __name__ == "__main__":
    main()
