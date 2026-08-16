"""M1 paired-layout K sweep over the M0.5-authorized accepting workloads.

M1 answers whether physical KV sharing moves the *optimal fixed K* for a workload
that actually accepts multi-draft speculation. It reuses the M0 paired-timing
machinery (``run_cell`` + ``analyze_capture_pairs`` + all five hard controls) and
the M0.5 prompt rendering (``select_prompts``). The only difference from M0 is the
prompt source: real workload prompts are rendered once and replayed through
``one_batch_server --prompt-list-file`` so each ``(layout, K)`` cell sees the exact
M0.5 prompts and, at temperature 0, the same acceptance trajectory across layouts.

One workload per plan / analyze pass (separate output directories) so the M0
analyzer's ``(batch_size, context_length, K, seed)`` grouping never mixes the two
workloads. Only ``code_completion`` and ``structured_json`` (the M0.5 materiality
review) are accepted. This module emits a K curve; it does not authorize a
controller or an upstream feature PR.
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
    # Render each seed's prompts once; both layouts and every K replay the same file
    # so acceptance is identical across the paired arms by construction.
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
            captures.append(capture_path)
            continue
        captures.append(
            run_cell(
                cell=cell,
                base_url=base_url,
                output_dir=output_dir,
                output_length_override=output_length_override,
                prompt_list_file=str(prompt_file),
            )
        )
    if not captures:
        raise RuntimeError(
            f"no plan cells match server layout={layout}, K={steps} in {plan_dir}"
        )
    return captures


def _mean_accepted_drafts(records: Sequence[dict[str, Any]]) -> float:
    values = [
        float(count)
        for record in records
        for count in record.get("correct_drafts_per_req", [])
    ]
    return sum(values) / len(values) if values else 0.0


def _median_verify_ms(records: Sequence[dict[str, Any]]) -> float:
    times = [
        float(record["target_verify_gpu_ms"])
        for record in records
        if "target_verify_gpu_ms" in record
    ]
    return median(times) if times else float("nan")


def summarize_k_curve(
    captures: Sequence[dict[str, Any]],
    *,
    discard_first: int = 5,
) -> dict[str, Any]:
    """Per-layout K -> (verify time, useful progress) curve and the implied K*.

    ``useful_progress`` = ``1 + mean accepted drafts`` (tokens emitted per target
    verify). ``K*`` maximizes ``useful_progress / median verify time`` — the
    tokens-per-unit-verify-time proxy. A shift means the argmax K differs between
    the shared and duplicated layouts.
    """
    per_layout_k: dict[str, dict[int, list[dict[str, float]]]] = {}
    for capture in captures:
        if capture["recorder"].get("components") != ["target_verify_gpu_time"]:
            continue
        cell = capture["cell"]
        layout = cell["layout"]
        k = int(cell["speculative_num_steps"])
        records = capture["recorder"]["records"][discard_first:]
        per_layout_k.setdefault(layout, {}).setdefault(k, []).append(
            {
                "verify_ms": _median_verify_ms(records),
                "useful_progress": 1.0 + _mean_accepted_drafts(records),
            }
        )

    curve: dict[str, list[dict[str, float]]] = {}
    k_star: dict[str, int] = {}
    for layout, by_k in per_layout_k.items():
        points: list[dict[str, float]] = []
        for k in sorted(by_k):
            seed_points = by_k[k]
            verify = median(p["verify_ms"] for p in seed_points)
            progress = median(p["useful_progress"] for p in seed_points)
            throughput = progress / verify if verify > 0 else float("nan")
            points.append(
                {
                    "k": k,
                    "median_verify_ms": verify,
                    "median_useful_progress": progress,
                    "useful_progress_per_ms": throughput,
                }
            )
        curve[layout] = points
        # keep finite throughput points (drop NaN from zero/absent verify time)
        finite = [
            p
            for p in points
            if p["useful_progress_per_ms"] == p["useful_progress_per_ms"]
        ]
        if finite:
            best = max(finite, key=lambda p: p["useful_progress_per_ms"])
            k_star[layout] = best["k"]

    shift = (
        len(k_star) == len(DEFAULT_LAYOUTS)
        and len(set(k_star.values())) > 1
    )
    return {"curve": curve, "k_star": k_star, "k_star_shift": shift}


def analyze_m1(
    *,
    captures_dirs: Sequence[Path],
    output: Path,
    summary_output: Path,
    minimum_effect_percent: float,
) -> dict[str, Any]:
    captures: list[dict[str, Any]] = []
    for directory in captures_dirs:
        for path in sorted(Path(directory).glob("*-capture.json")):
            captures.append(json.loads(path.read_text()))
    # Hard controls (prompt fingerprint, acceptance/context equality, runtime batch,
    # footprint reuse) are validated per (bs, ctx, K, seed) by the M0 analyzer.
    paired_rows = analyze_capture_pairs(captures)
    output.write_text(json.dumps(paired_rows, indent=2, sort_keys=True) + "\n")

    controls_ok = bool(paired_rows) and all(
        row["controls_valid"] for row in paired_rows
    )
    k_curve = summarize_k_curve(captures)
    summary = {
        "controls_valid": controls_ok,
        "m1_status": (
            "M1_K_STAR_SHIFT"
            if controls_ok and k_curve["k_star_shift"]
            else "M1_NO_SHIFT"
            if controls_ok
            else "M1_CONTROLS_INVALID"
        ),
        **k_curve,
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


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

    analyze = subparsers.add_parser("analyze", help="paired controls + K* curve")
    analyze.add_argument("--captures", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--summary-output", type=Path, required=True)
    analyze.add_argument("--minimum-effect-percent", type=float, default=2.0)

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
            captures_dirs=args.captures,
            output=args.output,
            summary_output=args.summary_output,
            minimum_effect_percent=args.minimum_effect_percent,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
