import unittest

import torch

from sglang.srt.speculative.radix_kv_m0_observability import (
    KV_FOOTPRINT,
    TARGET_VERIFY_GPU_TIME,
    compute_kv_footprint,
    resolve_components,
)


class TestRadixKVM0Observability(unittest.TestCase):
    def test_components_require_separate_timing_and_footprint_runs(self):
        self.assertEqual(
            resolve_components((TARGET_VERIFY_GPU_TIME,)), {TARGET_VERIFY_GPU_TIME}
        )
        self.assertEqual(resolve_components((KV_FOOTPRINT,)), {KV_FOOTPRINT})
        with self.assertRaisesRegex(ValueError, "separate runs"):
            resolve_components((TARGET_VERIFY_GPU_TIME, KV_FOOTPRINT))
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


if __name__ == "__main__":
    unittest.main()
