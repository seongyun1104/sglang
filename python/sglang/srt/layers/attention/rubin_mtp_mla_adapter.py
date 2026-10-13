"""Bridge SGLang's packed FP8 MLA verify tensors to FlashInfer's MLA wrapper.

The Rubin MTP implementation is exposed through FlashInfer's planned
``BatchMLAPagedAttentionWrapper`` API.  ``cutedsl_mla`` otherwise calls the
legacy monolithic decode entry point directly, so an explicit bridge is needed
for the new backend to be reachable.

This module deliberately owns only the narrow, eager target-verify contract.
FlashInfer remains responsible for the measured-region policy and concrete
backend fallback through ``backend="auto"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class _FlashInferMLAApi:
    wrapper_type: type
    metadata_type: type


def _load_flashinfer_mla_api() -> Optional[_FlashInferMLAApi]:
    """Return the canonical MLA API when the installed FlashInfer has it."""
    try:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
    except (ImportError, AttributeError):
        return None
    return _FlashInferMLAApi(BatchMLAPagedAttentionWrapper, MLAPlanMetadata)


class RubinMtpMLAAdapter:
    """Plan and run eligible packed FP8 target-verify batches.

    A backend instance is shared by every MLA layer.  Plans are therefore
    cached for one forward step and keyed by the metadata buffers and attention
    scale; callers must invoke :meth:`begin_forward` when metadata is rebuilt.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        *,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        api: Optional[_FlashInferMLAApi] = None,
        device_capability: Optional[tuple[int, int]] = None,
    ) -> None:
        self._workspace_buffer = workspace_buffer
        self._page_size = page_size
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._api = _load_flashinfer_mla_api() if api is None else api
        if device_capability is None and workspace_buffer.is_cuda:
            device_capability = torch.cuda.get_device_capability(
                workspace_buffer.device
            )
        self._device_capability = device_capability
        self._plans: dict[tuple, object] = {}

    @property
    def available(self) -> bool:
        return self._api is not None

    def begin_forward(self) -> None:
        """Discard plans whose metadata belonged to the previous step."""
        self._plans.clear()

    def _eligible(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        is_target_verify: bool,
        is_cuda_graph: bool,
    ) -> bool:
        if self._api is None or not is_target_verify or is_cuda_graph:
            return False
        if self._device_capability != (10, 7):
            return False
        if self._page_size not in (64, 128):
            return False
        if self._kv_lora_rank != 512 or self._qk_rope_head_dim != 64:
            return False
        if query.dtype != torch.float8_e4m3fn or kv_cache.dtype != query.dtype:
            return False
        if query.dim() != 4 or kv_cache.dim() != 4:
            return False
        if not query.is_contiguous() or not kv_cache.is_contiguous():
            return False

        batch_size, q_len, num_heads, head_dim = query.shape
        if not 32 <= batch_size <= 128 or q_len not in (2, 4):
            return False
        if num_heads != 128 or head_dim != 576:
            return False
        if kv_cache.shape[1:] != (1, self._page_size, 576):
            return False
        if block_tables.shape[0] != batch_size or seq_lens.shape != (batch_size,):
            return False
        if block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
            return False
        if query.device != kv_cache.device:
            return False
        if self._workspace_buffer.device != query.device:
            return False
        if block_tables.device != query.device or seq_lens.device != query.device:
            return False
        return True

    def run_if_eligible(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        bmm1_scale: float,
        is_target_verify: bool,
        is_cuda_graph: bool,
    ) -> Optional[torch.Tensor]:
        """Return wrapper output, or ``None`` when the narrow route is absent.

        Once the route is eligible, planning and launch errors intentionally
        propagate.  FlashInfer's auto planner handles typed backend rejection;
        hiding compiler or runtime failures here would turn correctness bugs
        into silent backend changes.
        """
        if not self._eligible(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            is_target_verify=is_target_verify,
            is_cuda_graph=is_cuda_graph,
        ):
            return None

        assert self._api is not None
        batch_size, q_len, num_heads, _ = query.shape
        plan_key = (
            query.device,
            batch_size,
            q_len,
            num_heads,
            float(bmm1_scale),
            block_tables.data_ptr(),
            tuple(block_tables.shape),
            seq_lens.data_ptr(),
        )
        wrapper = self._plans.get(plan_key)
        if wrapper is None:
            q_indptr = torch.arange(
                0,
                (batch_size + 1) * q_len,
                q_len,
                dtype=torch.int32,
                device=query.device,
            )
            metadata = self._api.metadata_type.dense(
                q_indptr,
                block_tables,
                seq_lens,
                max_q_len=q_len,
            )
            wrapper = self._api.wrapper_type(
                self._workspace_buffer,
                backend="auto",
            )
            wrapper.plan(
                metadata=metadata,
                num_heads=num_heads,
                head_dim_ckv=self._kv_lora_rank,
                head_dim_kpe=self._qk_rope_head_dim,
                page_size=self._page_size,
                causal=True,
                sm_scale=float(bmm1_scale),
                q_data_type=query.dtype,
                kv_data_type=kv_cache.dtype,
                output_dtype=query.dtype,
                query_layout="packed",
                kv_cache_layout="packed",
                lse_mode="none",
            )
            self._plans[plan_key] = wrapper

        packed_query = query.reshape(batch_size * q_len, num_heads, 576)
        packed_kv = kv_cache[:, 0]
        out = torch.empty(
            (batch_size * q_len, num_heads, self._kv_lora_rank),
            dtype=query.dtype,
            device=query.device,
        )
        result = wrapper.run(query=packed_query, kv_cache=packed_kv, out=out)
        return result.view(batch_size, q_len, num_heads, self._kv_lora_rank)
