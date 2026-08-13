import math
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import sglang.benchmark.radix_kv_sharing_m0 as harness
from sglang.benchmark.radix_kv_sharing_m0 import (
    aggregate_m0_rows,
    analyze_capture_pairs,
    build_cells,
)


def _capture(layout, times, acceptances, *, component="target_verify_gpu_time"):
    records = []
    for index, (timing, acceptance) in enumerate(zip(times, acceptances, strict=True)):
        record = {
            "record_id": index + 1,
            "batch_size": 8,
            "logical_context_lengths": [8192] * 8,
            "correct_drafts_per_req": acceptance,
            "target_verify_gpu_ms": timing,
        }
        records.append(record)
    return {
        "cell": {
            "batch_size": 8,
            "context_length": 8192,
            "layout": layout,
            "speculative_num_steps": 2,
            "seed": 17,
        },
        "input_fingerprints": ["prompt-a"] * 8,
        "client_seed": 17,
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
            "speculative_num_steps": 2,
            "speculative_num_draft_tokens": 3,
        },
        "recorder": {"components": [component], "records": records},
    }


def _footprint_capture(layout, acceptances, page_reuse):
    capture = _capture(layout, [], [], component="kv_footprint")
    capture["recorder"]["records"] = [
        {
            "batch_size": 8,
            "logical_context_lengths": [8192] * 8,
            "correct_drafts_per_req": acceptance,
            "page_reuse_ratio": page_reuse,
        }
        for acceptance in acceptances
    ]
    return capture


class TestRadixKVSharingM0Harness(unittest.TestCase):
    def test_default_matrix_has_72_seeded_cells(self):
        cells = build_cells()
        self.assertEqual(len(cells), 72)
        self.assertEqual(len({cell.cell_id for cell in cells}), 72)

    def test_pair_analysis_requires_acceptance_and_context_equivalence(self):
        acceptance = [[2] * 8, [1] * 8]
        rows = analyze_capture_pairs(
            [
                _capture("shared", [8.0, 9.0], acceptance),
                _capture("duplicated", [10.0, 10.0], acceptance),
                _footprint_capture("shared", acceptance, 0.8),
                _footprint_capture("duplicated", acceptance, 0.0),
            ],
            discard_first=0,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["paired_records"], 2)
        self.assertEqual(row["acceptance_match_ratio"], 1.0)
        self.assertEqual(row["context_match_ratio"], 1.0)
        self.assertTrue(row["controls_valid"])
        self.assertAlmostEqual(row["sharing_speedup_median_percent"], 15.0)
        self.assertEqual(row["shared_page_reuse_median"], 0.8)
        self.assertEqual(row["duplicated_page_reuse_median"], 0.0)

    def test_invalid_control_pair_suppresses_speedup(self):
        shared = _capture("shared", [8.0], [[2] * 8])
        duplicated = _capture("duplicated", [10.0], [[1] * 8])
        rows = analyze_capture_pairs([shared, duplicated], discard_first=0)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["controls_valid"])
        self.assertIn("acceptance_trajectory_mismatch", rows[0]["invalid_reason"])
        self.assertTrue(math.isnan(rows[0]["sharing_speedup_median_percent"]))

    def test_aggregate_requires_consistent_resolved_seed_effects(self):
        rows = [
            {
                "batch_size": 8,
                "context_length": 8192,
                "speculative_num_steps": 2,
                "controls_valid": True,
                "sharing_speedup_median_percent": effect,
            }
            for effect in (2.1, 2.5, 3.0)
        ]
        result = aggregate_m0_rows(rows, minimum_effect_percent=2.0)
        self.assertEqual(result[0]["m0_status"], "M0_SIGNAL")

        rows[1]["sharing_speedup_median_percent"] = 1.0
        result = aggregate_m0_rows(rows, minimum_effect_percent=2.0)
        self.assertEqual(result[0]["m0_status"], "BELOW_RESOLUTION")

    def test_run_matching_selects_current_layout_and_k(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory) / "plan"
            output_dir = Path(directory) / "output"
            plan_dir.mkdir()
            for cell in build_cells(
                batch_sizes=(8,),
                context_lengths=(8192,),
                steps=(0, 2),
                seeds=(17,),
            ):
                (plan_dir / f"{cell.cell_id}.json").write_text(json.dumps(asdict(cell)))
            server_info = {
                "internal_states": [
                    {
                        "disable_radix_cache": False,
                        "speculative_num_steps": 2,
                        "radix_kv_m0_record": {
                            "components": ["target_verify_gpu_time"]
                        },
                    }
                ]
            }
            expected = output_dir / "capture.json"
            with (
                patch.object(harness, "_get_json", return_value=server_info),
                patch.object(harness, "run_cell", return_value=expected) as run,
            ):
                captures = harness.run_matching_cells(
                    plan_dir=plan_dir,
                    base_url="http://server",
                    output_dir=output_dir,
                    output_length_override=16,
                )
            self.assertEqual(captures, [expected])
            selected = run.call_args.kwargs["cell"]
            self.assertEqual(selected.layout, "shared")
            self.assertEqual(selected.speculative_num_steps, 2)
            self.assertEqual(run.call_args.kwargs["output_length_override"], 16)

    def test_run_cell_passes_server_model_path_to_client(self):
        cell = build_cells(
            batch_sizes=(8,),
            context_lengths=(8192,),
            steps=(2,),
            seeds=(17,),
        )[0]
        server_info = {
            "internal_states": [
                {
                    "model_path": "target-model",
                    "disable_radix_cache": False,
                    "speculative_adaptive": False,
                    "speculative_eagle_topk": 1,
                    "speculative_num_steps": 2,
                    "speculative_num_draft_tokens": 3,
                    "radix_kv_m0_record": {
                        "components": ["target_verify_gpu_time"],
                        "records": [],
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            def fake_run(command, *, check):
                self.assertTrue(check)
                model_index = command.index("--model-path")
                self.assertEqual(command[model_index + 1], "target-model")
                result_index = command.index("--result-filename")
                Path(command[result_index + 1]).write_text(
                    json.dumps(
                        {
                            "input_fingerprints": ["prompt-a"] * 8,
                            "seed": 17,
                        }
                    )
                    + "\n"
                )

            with (
                patch.object(harness, "_get_json", return_value=server_info),
                patch.object(harness.subprocess, "run", side_effect=fake_run),
                patch.object(harness, "_git_head", return_value="test-head"),
            ):
                capture_path = harness.run_cell(
                    cell=cell,
                    base_url="http://server",
                    output_dir=output_dir,
                )
            capture = json.loads(capture_path.read_text())
            self.assertEqual(capture["server"]["model_path"], "target-model")


if __name__ == "__main__":
    unittest.main()
