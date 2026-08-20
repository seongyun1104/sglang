import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.radix_kv_m0_observability import (
    ACCEPTANCE,
    KV_FOOTPRINT,
    SPEC_CYCLE_GPU_TIME,
    TARGET_VERIFY_GPU_TIME,
    RadixKVM0Recorder,
    compute_kv_footprint,
    resolve_components,
)


class TestRadixKVM0Observability(unittest.TestCase):
    def test_components_require_separate_timing_and_footprint_runs(self):
        self.assertEqual(resolve_components((ACCEPTANCE,)), {ACCEPTANCE})
        self.assertEqual(
            resolve_components((TARGET_VERIFY_GPU_TIME,)), {TARGET_VERIFY_GPU_TIME}
        )
        self.assertEqual(resolve_components((KV_FOOTPRINT,)), {KV_FOOTPRINT})
        self.assertEqual(
            resolve_components((SPEC_CYCLE_GPU_TIME,)), {SPEC_CYCLE_GPU_TIME}
        )
        with self.assertRaisesRegex(ValueError, "separate runs"):
            resolve_components((TARGET_VERIFY_GPU_TIME, KV_FOOTPRINT))
        with self.assertRaisesRegex(ValueError, "only one"):
            resolve_components((TARGET_VERIFY_GPU_TIME, SPEC_CYCLE_GPU_TIME))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_components(("not-a-component",))

    def test_shared_physical_prefix_footprint(self):
        req_to_token = torch.tensor(
            [
                [0, 0, 0, 0],
                [4, 5, 6, 7],
                [4, 5, 8, 9],
            ],
            dtype=torch.int32,
        )
        result = compute_kv_footprint(
            req_to_token,
            torch.tensor([1, 2]),
            torch.tensor([4, 4]),
            page_size=2,
        )
        self.assertEqual(result.logical_kv_tokens, 8)
        self.assertEqual(result.unique_physical_slots, 6)
        self.assertAlmostEqual(result.slot_reuse_ratio, 0.25)
        self.assertEqual(result.logical_page_references, 4)
        self.assertEqual(result.unique_physical_pages, 3)
        self.assertAlmostEqual(result.page_reuse_ratio, 0.25)

    def test_duplicated_prefix_has_no_physical_reuse(self):
        req_to_token = torch.tensor(
            [
                [0, 0, 0, 0],
                [4, 5, 6, 7],
                [10, 11, 12, 13],
            ],
            dtype=torch.int32,
        )
        result = compute_kv_footprint(
            req_to_token,
            torch.tensor([1, 2]),
            torch.tensor([4, 4]),
            page_size=2,
        )
        self.assertEqual(result.unique_physical_slots, 8)
        self.assertEqual(result.slot_reuse_ratio, 0.0)
        self.assertEqual(result.unique_physical_pages, 4)
        self.assertEqual(result.page_reuse_ratio, 0.0)

    def test_empty_batch(self):
        result = compute_kv_footprint(
            torch.zeros((1, 4), dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            page_size=1,
        )
        self.assertEqual(result.logical_kv_tokens, 0)
        self.assertEqual(result.unique_physical_pages, 0)

    def test_spec_cycle_records_primary_and_sensitivity_intervals(self):
        class FakeEvent:
            def __init__(self, **_):
                pass

            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, _):
                return 1.0

        recorder = RadixKVM0Recorder(
            components=(SPEC_CYCLE_GPU_TIME,),
            device="cuda",
            page_size=1,
            radix_cache_enabled=True,
            gpu_id=0,
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(seqlen=8, rid="r0")],
            forward_iter=3,
        )
        with patch("torch.cuda.Event", FakeEvent):
            with recorder.spec_cycle(batch=batch, speculative_num_steps=2):
                with recorder.draft_stage():
                    pass
                with recorder.target_verify(
                    batch=batch,
                    req_to_token_pool=SimpleNamespace(),
                    speculative_num_steps=2,
                ):
                    pass
                with recorder.draft_extend_stage():
                    pass
            recorder.observe_acceptance([2])
            records = recorder.dump()["records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["draft_gpu_ms"], 1.0)
        self.assertEqual(records[0]["target_verify_gpu_ms"], 1.0)
        self.assertEqual(records[0]["primary_spec_cycle_gpu_ms"], 2.0)
        self.assertEqual(records[0]["spec_cycle_gpu_ms"], 1.0)
        self.assertEqual(records[0]["correct_drafts_per_req"], [2])


if __name__ == "__main__":
    unittest.main()
