import json
import tempfile
import unittest
from pathlib import Path

from sglang.benchmark.radix_kv_sharing_m05 import (
    analyze_acceptance_capture,
    load_workloads,
    select_prompts,
    summarize_workloads,
)


def _capture(steps, acceptance, *, workload_id="qa", seed=17):
    records = [
        {
            "batch_size": len(values),
            "correct_drafts_per_req": values,
        }
        for values in acceptance
    ]
    return {
        "workload_id": workload_id,
        "speculative_num_steps": steps,
        "seed": seed,
        "batch_size": 4,
        "recorder": {"components": ["acceptance"], "records": records},
    }


class TestRadixKVSharingM05(unittest.TestCase):
    def test_load_and_select_prompts_preserves_exact_prefix(self):
        document = {
            "schema_version": 1,
            "chat_template": "llama3",
            "workloads": [
                {
                    "id": "qa",
                    "system_prompt": "Answer briefly.",
                    "shared_user_prefix": "Question: ",
                    "queries": ["A?", "B?", "C?", "D?"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.json"
            path.write_text(json.dumps(document))
            workload = load_workloads(path)["qa"]
        prompts, indices = select_prompts(workload, batch_size=3, seed=17)
        repeated, repeated_indices = select_prompts(workload, batch_size=3, seed=17)
        self.assertEqual(prompts, repeated)
        self.assertEqual(indices, repeated_indices)
        self.assertEqual(len(set(prompts)), 3)
        shared = prompts[0].split("Question: ")[0]
        self.assertTrue(all(prompt.startswith(shared) for prompt in prompts))

    def test_acceptance_distribution_reports_multi_draft_support(self):
        row = analyze_acceptance_capture(
            _capture(4, [[0, 1, 2, 4], [1, 3]])
        )
        self.assertTrue(row["controls_valid"])
        self.assertEqual(row["acceptance_decisions"], 6)
        self.assertEqual(row["nonzero_acceptance_decisions"], 5)
        self.assertEqual(row["multi_draft_decisions"], 3)
        self.assertAlmostEqual(row["mean_accepted_drafts_per_target_verify"], 5.5)
        self.assertEqual(row["accepted_drafts_max"], 4)
        self.assertEqual(
            row["m0_5_screen_status"],
            "MULTI_DRAFT_OBSERVED_NEEDS_MATERIALITY_REVIEW",
        )
        self.assertFalse(row["m1_authorized"])

    def test_single_draft_only_workload_is_rejected(self):
        row = analyze_acceptance_capture(_capture(4, [[0, 0, 1, 0]]))
        self.assertEqual(row["m0_5_screen_status"], "NO_MULTI_DRAFT_SUPPORT")
        summary = summarize_workloads(
            [
                analyze_acceptance_capture(_capture(2, [[0, 1, 1, 0]])),
                row,
            ]
        )
        self.assertEqual(
            summary[0]["m0_5_workload_status"],
            "REJECT_NO_K4_MULTI_DRAFT_SUPPORT",
        )
        self.assertFalse(summary[0]["m1_authorized"])

    def test_out_of_range_acceptance_invalidates_capture(self):
        row = analyze_acceptance_capture(_capture(2, [[0, 3]]))
        self.assertFalse(row["controls_valid"])
        self.assertIn("acceptance_out_of_range", row["invalid_reason"])


if __name__ == "__main__":
    unittest.main()
