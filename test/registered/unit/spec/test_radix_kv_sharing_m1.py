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


def _timing_capture(layout, k, seed, *, verify_ms, accepted):
    # One retained record per capture (discard_first=0 in the test) carrying the
    # verify time and a full-batch accepted-draft vector.
    return {
        "cell": {
            "batch_size": 8,
            "context_length": m1.M1_NOMINAL_CONTEXT,
            "layout": layout,
            "speculative_num_steps": k,
            "seed": seed,
        },
        "recorder": {
            "components": ["target_verify_gpu_time"],
            "records": [
                {
                    "batch_size": 8,
                    "logical_context_lengths": [m1.M1_NOMINAL_CONTEXT] * 8,
                    "correct_drafts_per_req": [accepted] * 8,
                    "target_verify_gpu_ms": verify_ms,
                }
            ],
        },
    }


class TestRadixKVSharingM1(unittest.TestCase):
    def test_build_cells_sweeps_k0_to_5_for_both_layouts(self):
        # Bookkeeping: the K sweep and both layouts must stay complete. A dropped
        # K or layout (or a colliding cell_id) is the failure this guards.
        cells = build_m1_cells(seeds=(17, 29, 41))
        self.assertEqual(len(cells), 6 * 2 * 3)
        self.assertEqual(len({c.cell_id for c in cells}), len(cells))
        self.assertEqual(
            sorted({c.speculative_num_steps for c in cells}), [0, 1, 2, 3, 4, 5]
        )
        self.assertEqual(sorted({c.layout for c in cells}), ["duplicated", "shared"])

    def test_plan_rejects_unauthorized_workload(self):
        # Negative-branch contract: only the two M0.5-authorized workloads may be
        # planned. This guards the materiality gate degrading to always-allow.
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
                                "queries": ["a", "b", "c", "d", "e", "f", "g", "h"],
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

    def test_k_star_shift_detected_when_argmax_differs(self):
        # Derived property: K* is argmax over K of useful_progress / verify_ms.
        # Construct duplicated to peak at K=2 and shared to peak at K=4, so the
        # summary must report a shift. Guards the argmax/shift derivation.
        captures = []
        # useful_progress = 1 + accepted. Choose verify_ms so the ratio peaks where
        # intended for each layout.
        shared = {0: (1.0, 0), 2: (1.2, 2), 4: (1.3, 5)}  # ratios 1.0/2.5/4.6 -> K=4
        duplicated = {0: (1.0, 0), 2: (1.0, 2), 4: (3.0, 5)}  # ratios 1.0/3.0/2.0 -> K=2
        for k, (ms, acc) in shared.items():
            captures.append(_timing_capture("shared", k, 17, verify_ms=ms, accepted=acc))
        for k, (ms, acc) in duplicated.items():
            captures.append(
                _timing_capture("duplicated", k, 17, verify_ms=ms, accepted=acc)
            )
        summary = summarize_k_curve(captures, discard_first=0)
        self.assertEqual(summary["k_star"]["shared"], 4)
        self.assertEqual(summary["k_star"]["duplicated"], 2)
        self.assertTrue(summary["k_star_shift"])

    def test_no_shift_when_both_layouts_share_k_star(self):
        # Negative of the above: identical curves must not report a shift.
        captures = []
        curve = {0: (1.0, 0), 2: (1.0, 2), 4: (2.0, 5)}
        for layout in ("shared", "duplicated"):
            for k, (ms, acc) in curve.items():
                captures.append(
                    _timing_capture(layout, k, 17, verify_ms=ms, accepted=acc)
                )
        summary = summarize_k_curve(captures, discard_first=0)
        self.assertEqual(summary["k_star"]["shared"], summary["k_star"]["duplicated"])
        self.assertFalse(summary["k_star_shift"])

    def test_run_matching_passes_seed_prompt_list_to_run_cell(self):
        # Critical-path wiring: run-matching must select cells by the server's
        # layout/K and replay that seed's rendered prompt list through run_cell.
        # A regression that dropped the prompt-list wiring would make M1 silently
        # re-run synthetic gsp prompts.
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory) / "plan"
            output_dir = Path(directory) / "out"
            plan_dir.mkdir()
            for cell in build_m1_cells(steps=(0, 2), seeds=(17,)):
                (plan_dir / f"{cell.cell_id}.json").write_text(json.dumps(asdict(cell)))
            (plan_dir / "prompts-seed17.json").write_text(json.dumps(["p"] * 8))
            server_info = {
                "internal_states": [
                    {
                        "disable_radix_cache": False,
                        "speculative_num_steps": 2,
                        "radix_kv_m0_record": {"components": ["target_verify_gpu_time"]},
                    }
                ]
            }
            expected = output_dir / "capture.json"
            with (
                patch.object(m1, "_get_json", return_value=server_info),
                patch.object(m1, "run_cell", return_value=expected) as run,
            ):
                captures = m1.run_matching_m1_cells(
                    plan_dir=plan_dir,
                    base_url="http://server",
                    output_dir=output_dir,
                )
            self.assertEqual(captures, [expected])
            selected = run.call_args.kwargs["cell"]
            self.assertEqual(selected.layout, "shared")
            self.assertEqual(selected.speculative_num_steps, 2)
            self.assertEqual(
                run.call_args.kwargs["prompt_list_file"],
                str(plan_dir / "prompts-seed17.json"),
            )


if __name__ == "__main__":
    unittest.main()
