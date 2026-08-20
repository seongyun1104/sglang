import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import sglang.benchmark.radix_kv_sharing_m1 as m1
from sglang.benchmark.radix_kv_sharing_m1 import (
    build_m1_cells,
    summarize_k_curve,
    write_plan,
)


def _timing_capture(
    workload,
    layout,
    k,
    seed,
    *,
    verify_ms,
    draft_ms,
    accepted,
):
    primary_ms = verify_ms + draft_ms
    return {
        "m1_workload_id": workload,
        "m1_ignore_eos": False,
        "cell": {
            "batch_size": 8,
            "context_length": m1.M1_NOMINAL_CONTEXT,
            "layout": layout,
            "speculative_num_steps": k,
            "seed": seed,
        },
        "recorder": {
            "components": [m1.M1_TIMING_COMPONENT],
            "records": [
                {
                    "batch_size": 8,
                    "logical_context_lengths": [m1.M1_NOMINAL_CONTEXT] * 8,
                    "correct_drafts_per_req": [accepted] * 8,
                    "draft_gpu_ms": draft_ms,
                    "target_verify_gpu_ms": verify_ms,
                    "spec_cycle_gpu_ms": primary_ms + 0.2,
                    "draft_extend_gpu_ms": 0.1,
                    "unattributed_cycle_gpu_ms": 0.1,
                }
            ],
        },
    }


def _indexed_curves(shared_ms, duplicated_ms):
    forward = {}
    reverse = {}
    accepted = {0: 0, 2: 1, 4: 2}
    for seed in (17, 29, 41):
        for layout, timings in (
            ("shared", shared_ms),
            ("duplicated", duplicated_ms),
        ):
            for k, primary_ms in timings.items():
                capture = _timing_capture(
                    "code_completion",
                    layout,
                    k,
                    seed,
                    verify_ms=primary_ms,
                    draft_ms=0.0,
                    accepted=accepted[k],
                )
                key = ("code_completion", layout, k, seed)
                forward[key] = capture
                reverse[key] = capture
    return forward, reverse


def _with_controls(capture, *, component, page_reuse=0.0):
    cell = capture["cell"]
    record = capture["recorder"]["records"][0]
    if component == "kv_footprint":
        record = {
            "batch_size": 8,
            "logical_context_lengths": record["logical_context_lengths"],
            "correct_drafts_per_req": record["correct_drafts_per_req"],
            "page_reuse_ratio": page_reuse,
        }
        capture["recorder"] = {
            "components": [component],
            "records": [record, record],
        }
    else:
        capture["recorder"]["records"] = [record] * 6
    capture.update(
        {
            "input_fingerprints": [f"prompt-{i}" for i in range(8)],
            "client_seed": cell["seed"],
            "harness_git_head": "test-head",
            "server": {
                "version": "test",
                "model_path": "target",
                "speculative_draft_model_path": "draft",
                "attention_backend": "fa3",
                "dtype": "bfloat16",
                "device": "cuda",
                "tp_size": 1,
                "dp_size": 1,
                "page_size": 1,
                "disable_cuda_graph": False,
                "speculative_num_steps": cell["speculative_num_steps"],
                "speculative_num_draft_tokens": cell["speculative_num_steps"] + 1,
            },
        }
    )
    return capture


class TestRadixKVSharingM1(unittest.TestCase):
    def test_build_cells_sweeps_k0_to_5_for_both_layouts(self):
        cells = build_m1_cells(seeds=(17, 29, 41))
        self.assertEqual(len(cells), 6 * 2 * 3)
        self.assertEqual(len({c.cell_id for c in cells}), len(cells))
        self.assertEqual(
            sorted({c.speculative_num_steps for c in cells}), [0, 1, 2, 3, 4, 5]
        )
        self.assertEqual(sorted({c.layout for c in cells}), ["duplicated", "shared"])

    def test_plan_rejects_unauthorized_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            workload_file = Path(directory) / "w.json"
            workload_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workloads": [
                            {
                                "id": "short_qa",
                                "system_prompt": "s",
                                "shared_user_prefix": "p",
                                "queries": list("abcdefgh"),
                            }
                        ],
                    }
                )
            )
            with self.assertRaises(ValueError):
                write_plan(
                    workload_file=workload_file,
                    workload_id="short_qa",
                    plan_dir=Path(directory) / "plan",
                )

    def test_k_star_shift_requires_seed_support_and_effect_floor(self):
        forward, reverse = _indexed_curves(
            shared_ms={0: 1.0, 2: 1.5, 4: 1.8},
            duplicated_ms={0: 1.0, 2: 1.3, 4: 2.2},
        )
        summary = summarize_k_curve(
            forward,
            reverse,
            workloads=("code_completion",),
            steps=(0, 2, 4),
            seeds=(17, 29, 41),
            minimum_effect_percent=2.0,
            discard_first=0,
        )
        result = summary["workloads"]["code_completion"]
        self.assertEqual(result["aggregate_k_star"], {"shared": 4, "duplicated": 2})
        self.assertEqual(result["supporting_seed_count"], 3)
        self.assertEqual(result["status"], "M1_K_STAR_SHIFT")

    def test_argmax_shift_below_floor_is_unpowered(self):
        forward, reverse = _indexed_curves(
            shared_ms={0: 2.0, 2: 2.0, 4: 2.998},
            duplicated_ms={0: 2.0, 2: 1.999, 4: 3.0},
        )
        summary = summarize_k_curve(
            forward,
            reverse,
            workloads=("code_completion",),
            steps=(0, 2, 4),
            seeds=(17, 29, 41),
            minimum_effect_percent=2.0,
            discard_first=0,
        )
        self.assertEqual(
            summary["workloads"]["code_completion"]["status"],
            "M1_K_STAR_SHIFT_UNPOWERED",
        )

    def test_no_shift_when_both_layouts_share_k_star(self):
        forward, reverse = _indexed_curves(
            shared_ms={0: 1.0, 2: 1.3, 4: 2.2},
            duplicated_ms={0: 1.0, 2: 1.3, 4: 2.2},
        )
        summary = summarize_k_curve(
            forward,
            reverse,
            workloads=("code_completion",),
            steps=(0, 2, 4),
            seeds=(17, 29, 41),
            minimum_effect_percent=2.0,
            discard_first=0,
        )
        self.assertEqual(
            summary["workloads"]["code_completion"]["status"],
            "M1_NO_INTERACTION",
        )

    def test_analyzer_fails_closed_on_incomplete_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("forward", "reverse", "footprint"):
                (root / name).mkdir()
            summary = m1.analyze_m1(
                forward_captures_dirs=(root / "forward",),
                reverse_captures_dirs=(root / "reverse",),
                footprint_captures_dirs=(root / "footprint",),
                output=root / "paired.json",
                summary_output=root / "summary.json",
                minimum_effect_percent=2.0,
            )
            self.assertEqual(summary["m1_status"], "M1_INCOMPLETE")
            self.assertEqual(summary["completeness"]["expected_unique_cells"], 72)

    def test_complete_matrix_runs_controls_and_shift_gate(self):
        shared_ms = {0: 1.0, 2: 1.5, 4: 1.8}
        duplicated_ms = {0: 1.0, 2: 1.3, 4: 2.2}
        accepted = {0: 0, 2: 1, 4: 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("forward", "reverse", "footprint"):
                (root / name).mkdir()
            for layout, timings in (
                ("shared", shared_ms),
                ("duplicated", duplicated_ms),
            ):
                for k, primary_ms in timings.items():
                    for seed in (17, 29, 41):
                        timing = _with_controls(
                            _timing_capture(
                                "code_completion",
                                layout,
                                k,
                                seed,
                                verify_ms=primary_ms,
                                draft_ms=0.0,
                                accepted=accepted[k],
                            ),
                            component=m1.M1_TIMING_COMPONENT,
                        )
                        footprint = _with_controls(
                            _timing_capture(
                                "code_completion",
                                layout,
                                k,
                                seed,
                                verify_ms=primary_ms,
                                draft_ms=0.0,
                                accepted=accepted[k],
                            ),
                            component="kv_footprint",
                            page_reuse=0.8 if layout == "shared" else 0.0,
                        )
                        filename = f"{layout}-k{k}-seed{seed}-capture.json"
                        for order in ("forward", "reverse"):
                            (root / order / filename).write_text(json.dumps(timing))
                        (root / "footprint" / filename).write_text(
                            json.dumps(footprint)
                        )
            summary = m1.analyze_m1(
                forward_captures_dirs=(root / "forward",),
                reverse_captures_dirs=(root / "reverse",),
                footprint_captures_dirs=(root / "footprint",),
                output=root / "paired.json",
                summary_output=root / "summary.json",
                minimum_effect_percent=2.0,
                workloads=("code_completion",),
                steps=(0, 2, 4),
                seeds=(17, 29, 41),
            )
            self.assertEqual(summary["m1_status"], "M1_K_STAR_SHIFT")
            self.assertTrue(summary["controls_valid"])

    def test_run_matching_replays_prompts_and_respects_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory) / "plan"
            output_dir = Path(directory) / "out"
            plan_dir.mkdir()
            (plan_dir / "m1_workload.json").write_text(
                json.dumps({"workload_id": "code_completion"})
            )
            for cell in build_m1_cells(steps=(0, 2), seeds=(17,)):
                (plan_dir / f"{cell.cell_id}.json").write_text(json.dumps(asdict(cell)))
            (plan_dir / "prompts-seed17.json").write_text(json.dumps(["p"] * 8))
            server_info = {
                "internal_states": [
                    {
                        "disable_radix_cache": False,
                        "speculative_num_steps": 2,
                        "radix_kv_m0_record": {"components": [m1.M1_TIMING_COMPONENT]},
                    }
                ]
            }

            def fake_run_cell(**kwargs):
                cell = kwargs["cell"]
                path = output_dir / f"{cell.cell_id}-capture.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"cell": asdict(cell)}))
                return path

            with (
                patch.object(m1, "_get_json", return_value=server_info),
                patch.object(m1, "run_cell", side_effect=fake_run_cell) as run,
            ):
                captures = m1.run_matching_m1_cells(
                    plan_dir=plan_dir,
                    base_url="http://server",
                    output_dir=output_dir,
                )
            self.assertEqual(len(captures), 1)
            self.assertTrue(run.call_args.kwargs["respect_eos"])
            self.assertEqual(
                run.call_args.kwargs["prompt_list_file"],
                str(plan_dir / "prompts-seed17.json"),
            )
            capture = json.loads(captures[0].read_text())
            self.assertEqual(capture["m1_workload_id"], "code_completion")
            self.assertFalse(capture["m1_ignore_eos"])


if __name__ == "__main__":
    unittest.main()
