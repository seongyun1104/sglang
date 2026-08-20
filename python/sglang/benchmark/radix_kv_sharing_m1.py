"""M1 paired-layout K sweep over the M0.5-authorized accepting workloads.

M1 answers whether physical KV sharing moves the *optimal fixed K* for a workload
that actually accepts multi-draft speculation. It reuses M0's paired controls and
M0.5's prompt rendering, but its primary efficiency includes both draft and target
verification GPU time. Real workload prompts are rendered once and replayed through
``one_batch_server --prompt-list-file``; measured controls, rather than construction,
must prove that paired layouts follow identical acceptance and context trajectories.

Plans and captures remain separated by workload. The final analyzer handles both
authorized workloads without mixing their M0 grouping keys, requires the complete
counterbalanced matrix, and applies effect-size and seed-reproduction gates. It emits
a K curve; it does not authorize a controller or an upstream feature PR.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from sglang.benchmark.radix_kv_sharing_m0 import (
    DEFAULT_LAYOUTS,
    DEFAULT_SEEDS,
    Cell,
    _find_recorder,
    _get_json,
    _parse_cell,
    analyze_capture_pairs,
    run_cell,
)
from sglang.benchmark.radix_kv_sharing_m05 import load_workloads, select_prompts

DEFAULT_M1_STEPS = (0, 1, 2, 3, 4, 5)
M1_AUTHORIZED_WORKLOADS = ("code_completion", "structured_json")
M1_BATCH_SIZE = 8
M1_TIMING_COMPONENT = "spec_cycle_gpu_time"
# Per-workload-constant context label. The real length is defined by the rendered
# prompts; the analyzer only needs it identical across the shared/duplicated pair
# (guaranteed — same prompt-list file), which its context control already checks.
M1_NOMINAL_CONTEXT = 4096


def render_prompt_list(
    *, workload: dict[str, Any], batch_size: int, seed: int
) -> list[str]:
    """Rendered M0.5 prompts (shared prefix + seeded distinct queries)."""
    prompts, _ = select_prompts(workload, batch_size=batch_size, seed=seed)
    return prompts


def build_m1_cells(
    *,
    layouts: Sequence[str] = DEFAULT_LAYOUTS,
    steps: Sequence[int] = DEFAULT_M1_STEPS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    batch_size: int = M1_BATCH_SIZE,
    output_length: int = 128,
) -> list[Cell]:
    unknown = set(layouts) - set(DEFAULT_LAYOUTS)
    if unknown:
        raise ValueError(f"unknown layouts: {sorted(unknown)}")
    cells: list[Cell] = []
    for layout in layouts:
        for k in steps:
            for seed in seeds:
                cells.append(
                    Cell(
                        batch_size=int(batch_size),
                        context_length=M1_NOMINAL_CONTEXT,
                        layout=layout,
                        speculative_num_steps=int(k),
                        seed=int(seed),
                        system_prompt_length=0,
                        question_length=0,
                        output_length=int(output_length),
                    )
                )
    return cells


def write_plan(
    *,
    workload_file: Path,
    workload_id: str,
    plan_dir: Path,
    steps: Sequence[int] = DEFAULT_M1_STEPS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    batch_size: int = M1_BATCH_SIZE,
    output_length: int = 128,
) -> None:
    if workload_id not in M1_AUTHORIZED_WORKLOADS:
        raise ValueError(
            f"{workload_id!r} is not an M0.5-authorized workload; "
            f"authorized: {M1_AUTHORIZED_WORKLOADS}"
        )
    workloads = load_workloads(workload_file)
    if workload_id not in workloads:
        raise ValueError(f"workload file has no workload {workload_id!r}")
    if plan_dir.exists() and any(plan_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty plan dir: {plan_dir}")
    plan_dir.mkdir(parents=True, exist_ok=True)

    (plan_dir / "m1_workload.json").write_text(
        json.dumps(
            {"workload_file": str(workload_file), "workload_id": workload_id},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for cell in build_m1_cells(
        steps=steps, seeds=seeds, batch_size=batch_size, output_length=output_length
    ):
        (plan_dir / f"{cell.cell_id}.json").write_text(json.dumps(asdict(cell)))
    # Render each seed's prompts once; both layouts and every K replay the same file.
    # The paired control, rather than this construction, validates acceptance equality.
    for seed in seeds:
        prompts = render_prompt_list(
            workload=workloads[workload_id], batch_size=batch_size, seed=int(seed)
        )
        (plan_dir / f"prompts-seed{seed}.json").write_text(json.dumps(prompts))


def run_matching_m1_cells(
    *,
    plan_dir: Path,
    base_url: str,
    output_dir: Path,
    output_length_override: int | None = None,
    skip_existing: bool = False,
) -> list[Path]:
    server_info = _get_json(base_url.rstrip("/") + "/server_info")
    state = server_info["internal_states"][0]
    layout = "duplicated" if bool(state.get("disable_radix_cache")) else "shared"
    steps = int(state.get("speculative_num_steps", -1))
    component_tag = "-".join(_find_recorder(server_info)["components"])
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_info = json.loads((plan_dir / "m1_workload.json").read_text())
    workload_id = workload_info["workload_id"]

    captures: list[Path] = []
    for path in sorted(plan_dir.glob("bs*.json")):
        cell = _parse_cell(path)
        if cell.layout != layout or cell.speculative_num_steps != steps:
            continue
        prompt_file = plan_dir / f"prompts-seed{cell.seed}.json"
        if not prompt_file.exists():
            raise RuntimeError(
                f"missing rendered prompts for seed {cell.seed}: {prompt_file}"
            )
        capture_path = output_dir / f"{cell.cell_id}-{component_tag}-capture.json"
        if skip_existing and capture_path.exists():
            existing = json.loads(capture_path.read_text())
            if existing.get("m1_workload_id") != workload_id:
                raise RuntimeError(f"stale or mismatched M1 capture: {capture_path}")
            captures.append(capture_path)
            continue
        capture_path = run_cell(
            cell=cell,
            base_url=base_url,
            output_dir=output_dir,
            output_length_override=output_length_override,
            prompt_list_file=str(prompt_file),
            respect_eos=True,
        )
        capture = json.loads(capture_path.read_text())
        capture["m1_workload_id"] = workload_id
        capture["m1_ignore_eos"] = False
        capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
        captures.append(capture_path)
    if not captures:
        raise RuntimeError(
            f"no plan cells match server layout={layout}, K={steps} in {plan_dir}"
        )
    return captures


def _capture_key(capture: dict[str, Any]) -> tuple[str, str, int, int]:
    cell = capture["cell"]
    return (
        str(capture.get("m1_workload_id", "")),
        str(cell["layout"]),
        int(cell["speculative_num_steps"]),
        int(cell["seed"]),
    )


def _expected_keys(
    *,
    workloads: Sequence[str],
    layouts: Sequence[str],
    steps: Sequence[int],
    seeds: Sequence[int],
) -> set[tuple[str, str, int, int]]:
    return {
        (workload, layout, int(k), int(seed))
        for workload in workloads
        for layout in layouts
        for k in steps
        for seed in seeds
    }


def _index_component(
    captures: Sequence[dict[str, Any]], component: str
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], list[list[Any]]]:
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    duplicates: list[list[Any]] = []
    for capture in captures:
        if capture["recorder"].get("components") != [component]:
            continue
        key = _capture_key(capture)
        if key in indexed:
            duplicates.append(list(key))
        indexed[key] = capture
    return indexed, duplicates


def _load_captures(directories: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for directory in directories
        for path in sorted(Path(directory).rglob("*-capture.json"))
    ]


def _record_efficiency(record: dict[str, Any]) -> dict[str, float] | None:
    accepted = record.get("correct_drafts_per_req")
    if not accepted:
        return None
    draft_ms = float(record.get("draft_gpu_ms", 0.0))
    verify_ms = float(record.get("target_verify_gpu_ms", float("nan")))
    full_cycle_ms = float(record.get("spec_cycle_gpu_ms", float("nan")))
    primary_ms = draft_ms + verify_ms
    useful_progress = 1.0 + sum(float(value) for value in accepted) / len(accepted)
    if primary_ms <= 0 or primary_ms != primary_ms:
        return None
    return {
        "primary_efficiency": useful_progress / primary_ms,
        "full_cycle_efficiency": (
            useful_progress / full_cycle_ms if full_cycle_ms > 0 else float("nan")
        ),
        "useful_progress": useful_progress,
        "draft_gpu_ms": draft_ms,
        "target_verify_gpu_ms": verify_ms,
        "spec_cycle_gpu_ms": full_cycle_ms,
        "draft_extend_gpu_ms": float(record.get("draft_extend_gpu_ms", 0.0)),
        "unattributed_cycle_gpu_ms": float(
            record.get("unattributed_cycle_gpu_ms", float("nan"))
        ),
    }


def _capture_point(capture: dict[str, Any], *, discard_first: int) -> dict[str, float]:
    expected_batch = int(capture["cell"]["batch_size"])
    values = [
        value
        for record in capture["recorder"]["records"][discard_first:]
        if int(record.get("batch_size", -1)) == expected_batch
        if (value := _record_efficiency(record)) is not None
    ]
    if not values:
        raise RuntimeError(
            "capture has no retained full-batch timing records: "
            f"{_capture_key(capture)}"
        )
    return {field: median(value[field] for value in values) for field in values[0]}


def _best_k(efficiency_by_k: dict[int, float]) -> int:
    return max(sorted(efficiency_by_k), key=lambda k: efficiency_by_k[k])


def _relative_gain(best: float, alternative: float) -> float:
    return (
        (best - alternative) / alternative * 100.0 if alternative > 0 else float("nan")
    )


def summarize_k_curve(
    forward: dict[tuple[str, str, int, int], dict[str, Any]],
    reverse: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    workloads: Sequence[str],
    steps: Sequence[int],
    seeds: Sequence[int],
    minimum_effect_percent: float,
    discard_first: int = 5,
) -> dict[str, Any]:
    """Build counterbalanced per-seed K curves and apply the M1 shift gate."""
    points: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for key in forward:
        first = _capture_point(forward[key], discard_first=discard_first)
        second = _capture_point(reverse[key], discard_first=discard_first)
        points[key] = {field: median((first[field], second[field])) for field in first}

    workload_results: dict[str, Any] = {}
    statuses: list[str] = []
    for workload in workloads:
        seed_results: list[dict[str, Any]] = []
        aggregate_efficiency: dict[str, dict[int, float]] = {}
        curve: dict[str, list[dict[str, float]]] = {}
        for layout in DEFAULT_LAYOUTS:
            aggregate_efficiency[layout] = {
                int(k): median(
                    points[(workload, layout, int(k), int(seed))]["primary_efficiency"]
                    for seed in seeds
                )
                for k in steps
            }
            curve[layout] = [
                {
                    "k": int(k),
                    **{
                        field: median(
                            points[(workload, layout, int(k), int(seed))][field]
                            for seed in seeds
                        )
                        for field in next(iter(points.values()))
                    },
                }
                for k in steps
            ]

        aggregate_k_star = {
            layout: _best_k(aggregate_efficiency[layout]) for layout in DEFAULT_LAYOUTS
        }
        aggregate_direction = (
            aggregate_k_star["shared"] - aggregate_k_star["duplicated"]
        )
        for seed in seeds:
            per_layout = {
                layout: {
                    int(k): points[(workload, layout, int(k), int(seed))][
                        "primary_efficiency"
                    ]
                    for k in steps
                }
                for layout in DEFAULT_LAYOUTS
            }
            seed_k_star = {
                layout: _best_k(per_layout[layout]) for layout in DEFAULT_LAYOUTS
            }
            seed_results.append(
                {
                    "seed": int(seed),
                    "k_star": seed_k_star,
                    "shift_direction": seed_k_star["shared"]
                    - seed_k_star["duplicated"],
                }
            )

        shared_k = aggregate_k_star["shared"]
        duplicated_k = aggregate_k_star["duplicated"]
        shared_preference = _relative_gain(
            aggregate_efficiency["shared"][shared_k],
            aggregate_efficiency["shared"][duplicated_k],
        )
        duplicated_preference = _relative_gain(
            aggregate_efficiency["duplicated"][duplicated_k],
            aggregate_efficiency["duplicated"][shared_k],
        )
        supporting_seeds = sum(
            result["shift_direction"] * aggregate_direction > 0
            for result in seed_results
        )
        powered = (
            shared_preference >= minimum_effect_percent
            and duplicated_preference >= minimum_effect_percent
        )
        if aggregate_direction == 0:
            status = "M1_NO_INTERACTION"
        elif supporting_seeds >= 2 and powered:
            status = "M1_K_STAR_SHIFT"
        else:
            status = "M1_K_STAR_SHIFT_UNPOWERED"
        statuses.append(status)

        interaction = [
            {
                "k": int(k),
                "shared_over_duplicated_efficiency_percent": (
                    aggregate_efficiency["shared"][int(k)]
                    / aggregate_efficiency["duplicated"][int(k)]
                    - 1.0
                )
                * 100.0,
            }
            for k in steps
        ]
        workload_results[workload] = {
            "status": status,
            "curve": curve,
            "interaction": interaction,
            "aggregate_k_star": aggregate_k_star,
            "shared_preference_over_duplicated_k_percent": shared_preference,
            "duplicated_preference_over_shared_k_percent": duplicated_preference,
            "supporting_seed_count": supporting_seeds,
            "seed_results": seed_results,
        }

    overall_status = (
        "M1_K_STAR_SHIFT"
        if "M1_K_STAR_SHIFT" in statuses
        else "M1_K_STAR_SHIFT_UNPOWERED"
        if "M1_K_STAR_SHIFT_UNPOWERED" in statuses
        else "M1_NO_INTERACTION"
    )
    return {"m1_status": overall_status, "workloads": workload_results}


def analyze_m1(
    *,
    forward_captures_dirs: Sequence[Path],
    reverse_captures_dirs: Sequence[Path],
    footprint_captures_dirs: Sequence[Path],
    output: Path,
    summary_output: Path,
    minimum_effect_percent: float,
    workloads: Sequence[str] = M1_AUTHORIZED_WORKLOADS,
    steps: Sequence[int] = DEFAULT_M1_STEPS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    if minimum_effect_percent <= 0:
        raise ValueError("minimum_effect_percent must be positive")
    expected = _expected_keys(
        workloads=workloads,
        layouts=DEFAULT_LAYOUTS,
        steps=steps,
        seeds=seeds,
    )
    forward, forward_duplicates = _index_component(
        _load_captures(forward_captures_dirs), M1_TIMING_COMPONENT
    )
    reverse, reverse_duplicates = _index_component(
        _load_captures(reverse_captures_dirs), M1_TIMING_COMPONENT
    )
    footprint, footprint_duplicates = _index_component(
        _load_captures(footprint_captures_dirs), "kv_footprint"
    )
    completeness = {
        "expected_unique_cells": len(expected),
        "forward_missing": [list(key) for key in sorted(expected - set(forward))],
        "reverse_missing": [list(key) for key in sorted(expected - set(reverse))],
        "footprint_missing": [list(key) for key in sorted(expected - set(footprint))],
        "forward_unexpected": [list(key) for key in sorted(set(forward) - expected)],
        "reverse_unexpected": [list(key) for key in sorted(set(reverse) - expected)],
        "footprint_unexpected": [
            list(key) for key in sorted(set(footprint) - expected)
        ],
        "duplicates": forward_duplicates + reverse_duplicates + footprint_duplicates,
    }
    complete = all(
        not value
        for key, value in completeness.items()
        if key != "expected_unique_cells"
    )
    if not complete:
        summary = {
            "m1_status": "M1_INCOMPLETE",
            "minimum_effect_percent": minimum_effect_percent,
            "completeness": completeness,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("[]\n")
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    paired_output: dict[str, Any] = {}
    controls_valid = True
    for workload in workloads:
        footprint_values = [
            capture for key, capture in footprint.items() if key[0] == workload
        ]
        forward_rows = analyze_capture_pairs(
            [capture for key, capture in forward.items() if key[0] == workload]
            + footprint_values,
            timing_component=M1_TIMING_COMPONENT,
            timing_fields=("draft_gpu_ms", "target_verify_gpu_ms"),
        )
        reverse_rows = analyze_capture_pairs(
            [capture for key, capture in reverse.items() if key[0] == workload]
            + footprint_values,
            timing_component=M1_TIMING_COMPONENT,
            timing_fields=("draft_gpu_ms", "target_verify_gpu_ms"),
        )
        paired_output[workload] = {
            "forward": forward_rows,
            "reverse": reverse_rows,
        }
        expected_paired_rows = len(steps) * len(seeds)
        controls_valid = controls_valid and (
            len(forward_rows) == expected_paired_rows
            and len(reverse_rows) == expected_paired_rows
            and all(row["controls_valid"] for row in forward_rows + reverse_rows)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(paired_output, indent=2, sort_keys=True) + "\n")
    if not controls_valid:
        summary = {
            "m1_status": "M1_INVALID",
            "minimum_effect_percent": minimum_effect_percent,
            "completeness": completeness,
        }
    else:
        summary = summarize_k_curve(
            forward,
            reverse,
            workloads=workloads,
            steps=steps,
            seeds=seeds,
            minimum_effect_percent=minimum_effect_percent,
        )
        summary.update(
            {
                "minimum_effect_percent": minimum_effect_percent,
                "completeness": completeness,
                "controls_valid": True,
            }
        )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def analyze_smoke(
    *, timing_captures_dirs: Sequence[Path], footprint_captures_dirs: Sequence[Path]
) -> dict[str, Any]:
    timing = _load_captures(timing_captures_dirs)
    footprint = _load_captures(footprint_captures_dirs)
    workloads = sorted(
        {capture.get("m1_workload_id", "") for capture in timing + footprint}
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for workload in workloads:
        workload_captures = [
            capture
            for capture in timing + footprint
            if capture.get("m1_workload_id") == workload
        ]
        rows[workload] = analyze_capture_pairs(
            workload_captures,
            timing_component=M1_TIMING_COMPONENT,
            timing_fields=("draft_gpu_ms", "target_verify_gpu_ms"),
        )
    valid = bool(rows) and all(
        workload_rows and all(row["controls_valid"] for row in workload_rows)
        for workload_rows in rows.values()
    )
    return {"smoke_valid": valid, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write one workload's M1 K-sweep plan")
    plan.add_argument("--workload-file", type=Path, required=True)
    plan.add_argument("--workload-id", required=True, choices=M1_AUTHORIZED_WORKLOADS)
    plan.add_argument("--plan-dir", type=Path, required=True)

    run = subparsers.add_parser(
        "run-matching", help="run every plan cell matching the current server"
    )
    run.add_argument("--plan-dir", type=Path, required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:30000")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--output-length-override", type=int)
    run.add_argument("--skip-existing", action="store_true")

    analyze = subparsers.add_parser(
        "analyze", help="counterbalanced controls + K* curve"
    )
    analyze.add_argument("--forward-captures", type=Path, nargs="+", required=True)
    analyze.add_argument("--reverse-captures", type=Path, nargs="+", required=True)
    analyze.add_argument("--footprint-captures", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--summary-output", type=Path, required=True)
    analyze.add_argument("--minimum-effect-percent", type=float, default=2.0)

    smoke = subparsers.add_parser("analyze-smoke", help="validate one paired M1 cell")
    smoke.add_argument("--timing-captures", type=Path, nargs="+", required=True)
    smoke.add_argument("--footprint-captures", type=Path, nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "plan":
        write_plan(
            workload_file=args.workload_file,
            workload_id=args.workload_id,
            plan_dir=args.plan_dir,
        )
    elif args.command == "run-matching":
        for path in run_matching_m1_cells(
            plan_dir=args.plan_dir,
            base_url=args.base_url,
            output_dir=args.output_dir,
            output_length_override=args.output_length_override,
            skip_existing=args.skip_existing,
        ):
            print(path)
    elif args.command == "analyze":
        summary = analyze_m1(
            forward_captures_dirs=args.forward_captures,
            reverse_captures_dirs=args.reverse_captures,
            footprint_captures_dirs=args.footprint_captures,
            output=args.output,
            summary_output=args.summary_output,
            minimum_effect_percent=args.minimum_effect_percent,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["m1_status"] in {"M1_INCOMPLETE", "M1_INVALID"}:
            raise SystemExit(2)
    else:
        summary = analyze_smoke(
            timing_captures_dirs=args.timing_captures,
            footprint_captures_dirs=args.footprint_captures,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["smoke_valid"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
