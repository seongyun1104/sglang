"""CPU contract tests for the Rubin MLA target-verify bridge.

The real kernel requires SM107, but routing, canonical metadata construction,
plan reuse, and failure boundaries are ordinary Python contracts.
"""

import unittest

import torch

from sglang.srt.layers.attention.rubin_mtp_mla_adapter import (
    RubinMtpMLAAdapter,
    _FlashInferMLAApi,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeMetadata:
    calls = []

    @classmethod
    def dense(cls, q_indptr, block_tables, seq_lens, max_q_len=None):
        metadata = {
            "q_indptr": q_indptr,
            "block_tables": block_tables,
            "seq_lens": seq_lens,
            "max_q_len": max_q_len,
        }
        cls.calls.append(metadata)
        return metadata


class _FakeWrapper:
    instances = []
    plan_error = None

    def __init__(self, workspace, *, backend):
        self.workspace = workspace
        self.backend = backend
        self.plan_calls = []
        self.run_calls = []
        type(self).instances.append(self)

    def plan(self, **kwargs):
        self.plan_calls.append(kwargs)
        if type(self).plan_error is not None:
            raise type(self).plan_error

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        return kwargs["out"]


class TestRubinMtpMLAAdapter(CustomTestCase):
    def setUp(self):
        super().setUp()
        _FakeMetadata.calls.clear()
        _FakeWrapper.instances.clear()
        _FakeWrapper.plan_error = None
        self.adapter = RubinMtpMLAAdapter(
            torch.empty(1024, dtype=torch.uint8),
            page_size=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            api=_FlashInferMLAApi(_FakeWrapper, _FakeMetadata),
            device_capability=(10, 7),
        )

    @staticmethod
    def _inputs(*, batch_size=32, q_len=4, dtype=torch.float8_e4m3fn):
        query = torch.empty((batch_size, q_len, 128, 576), dtype=dtype)
        kv_cache = torch.empty((4, 1, 64, 576), dtype=dtype)
        block_tables = torch.zeros((batch_size, 2), dtype=torch.int32)
        seq_lens = torch.full((batch_size,), 128, dtype=torch.int32)
        return query, kv_cache, block_tables, seq_lens

    def _run(self, inputs=None, **kwargs):
        if inputs is None:
            inputs = self._inputs()
        return self.adapter.run_if_eligible(
            *inputs,
            bmm1_scale=0.125,
            is_target_verify=kwargs.pop("is_target_verify", True),
            is_cuda_graph=kwargs.pop("is_cuda_graph", False),
            **kwargs,
        )

    def test_builds_canonical_packed_plan_and_preserves_output_shape(self):
        query, kv_cache, block_tables, seq_lens = self._inputs()
        result = self._run((query, kv_cache, block_tables, seq_lens))

        self.assertEqual(result.shape, (32, 4, 128, 512))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)
        self.assertEqual(len(_FakeWrapper.instances), 1)
        wrapper = _FakeWrapper.instances[0]
        self.assertEqual(wrapper.backend, "auto")
        plan = wrapper.plan_calls[0]
        self.assertEqual(plan["sm_scale"], 0.125)
        self.assertEqual(plan["query_layout"], "packed")
        self.assertEqual(plan["kv_cache_layout"], "packed")
        self.assertEqual(plan["output_dtype"], torch.float8_e4m3fn)
        torch.testing.assert_close(
            _FakeMetadata.calls[0]["q_indptr"],
            torch.arange(0, 33 * 4, 4, dtype=torch.int32),
        )
        run = wrapper.run_calls[0]
        self.assertEqual(run["query"].shape, (32 * 4, 128, 576))
        self.assertEqual(run["kv_cache"].shape, (4, 64, 576))

    def test_reuses_plan_within_step_and_replans_after_metadata_boundary(self):
        inputs = self._inputs()
        self._run(inputs)
        self._run(inputs)
        self.assertEqual(len(_FakeWrapper.instances), 1)
        self.assertEqual(len(_FakeWrapper.instances[0].run_calls), 2)

        self.adapter.begin_forward()
        self._run(inputs)
        self.assertEqual(len(_FakeWrapper.instances), 2)

    def test_non_target_graph_and_out_of_contract_shapes_fall_back(self):
        inputs = self._inputs()
        self.assertIsNone(self._run(inputs, is_target_verify=False))
        self.assertIsNone(self._run(inputs, is_cuda_graph=True))
        self.assertIsNone(self._run(self._inputs(batch_size=31)))
        self.assertIsNone(self._run(self._inputs(q_len=1)))
        self.assertIsNone(self._run(self._inputs(dtype=torch.bfloat16)))
        self.assertFalse(_FakeWrapper.instances)

    def test_non_rubin_device_falls_back_before_planning(self):
        adapter = RubinMtpMLAAdapter(
            torch.empty(1024, dtype=torch.uint8),
            page_size=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            api=_FlashInferMLAApi(_FakeWrapper, _FakeMetadata),
            device_capability=(10, 0),
        )
        self.assertIsNone(
            adapter.run_if_eligible(
                *self._inputs(),
                bmm1_scale=0.125,
                is_target_verify=True,
                is_cuda_graph=False,
            )
        )
        self.assertFalse(_FakeWrapper.instances)

    def test_admitted_plan_failure_is_not_silently_downgraded(self):
        _FakeWrapper.plan_error = RuntimeError("compiler failed")
        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
            self._run()


if __name__ == "__main__":
    unittest.main()
