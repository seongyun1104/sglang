"""Opt-in observability for the Radix KV-sharing M0 falsification experiment.

This module is deliberately not a controller. It records acceptance only,
target-verify GPU time, a speculative-cycle timing breakdown, or the physical KV
footprint for a decode batch. Footprint collection launches GPU work and is therefore
forbidden in the same process as timing collection.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Iterator, Sequence

import torch


ACCEPTANCE = "acceptance"
TARGET_VERIFY_GPU_TIME = "target_verify_gpu_time"
SPEC_CYCLE_GPU_TIME = "spec_cycle_gpu_time"
KV_FOOTPRINT = "kv_footprint"
_TIMING_COMPONENTS = {TARGET_VERIFY_GPU_TIME, SPEC_CYCLE_GPU_TIME}
_VALID_COMPONENTS = {ACCEPTANCE, *_TIMING_COMPONENTS, KV_FOOTPRINT}
_MAX_RECORDS = 200_000


def resolve_components(raw: Sequence[str]) -> set[str]:
    components = {value.strip() for value in raw if value.strip()}
    unknown = components - _VALID_COMPONENTS
    if unknown:
        raise ValueError(
            f"Unknown Radix KV M0 component(s): {sorted(unknown)}; "
            f"valid: {sorted(_VALID_COMPONENTS)}"
        )
    if len(components & _TIMING_COMPONENTS) > 1:
        raise ValueError("select only one Radix KV timing component")
    if components & _TIMING_COMPONENTS and KV_FOOTPRINT in components:
        raise ValueError(
            "GPU timing and kv_footprint require separate runs: "
            "footprint collection touches the page table and would contaminate timing"
        )
    return components


@dataclass(frozen=True)
class KVFootprint:
    logical_kv_tokens: int
    unique_physical_slots: int
    slot_reuse_ratio: float
    logical_page_references: int
    unique_physical_pages: int
    page_reuse_ratio: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "logical_kv_tokens": self.logical_kv_tokens,
            "unique_physical_slots": self.unique_physical_slots,
            "slot_reuse_ratio": self.slot_reuse_ratio,
            "logical_page_references": self.logical_page_references,
            "unique_physical_pages": self.unique_physical_pages,
            "page_reuse_ratio": self.page_reuse_ratio,
        }


def compute_kv_footprint(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
) -> KVFootprint:
    """Count logical references and unique physical KV slots/pages.

    This helper is intended for an accounting-only run.  It synchronizes when the
    returned scalar values are materialized and must not be enabled in timing runs.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if req_pool_indices.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("req_pool_indices and seq_lens must be one-dimensional")
    if req_pool_indices.numel() != seq_lens.numel():
        raise ValueError("req_pool_indices and seq_lens must have equal length")
    if seq_lens.numel() == 0:
        return KVFootprint(0, 0, 0.0, 0, 0, 0.0)
    if torch.any(seq_lens < 0):
        raise ValueError("seq_lens must be nonnegative")

    max_len = int(seq_lens.max().item())
    if max_len > req_to_token.shape[1]:
        raise ValueError(
            f"max seq_len {max_len} exceeds req_to_token width {req_to_token.shape[1]}"
        )
    if max_len == 0:
        return KVFootprint(0, 0, 0.0, 0, 0, 0.0)

    rows = req_to_token.index_select(0, req_pool_indices.to(torch.long))[:, :max_len]
    valid = torch.arange(max_len, device=seq_lens.device).unsqueeze(
        0
    ) < seq_lens.unsqueeze(1)
    slots = rows[valid].to(torch.int64)
    if torch.any(slots < 0):
        raise ValueError("req_to_token contains an unallocated slot in the live prefix")
    logical_tokens = int(slots.numel())
    unique_slots = int(torch.unique(slots).numel())

    page_rows = torch.div(rows.to(torch.int64), page_size, rounding_mode="floor")
    logical_page_refs = 0
    per_request_pages: list[torch.Tensor] = []
    for row, seq_len in zip(page_rows, seq_lens, strict=True):
        length = int(seq_len.item())
        pages = torch.unique(row[:length])
        logical_page_refs += int(pages.numel())
        per_request_pages.append(pages)
    unique_pages = int(torch.unique(torch.cat(per_request_pages)).numel())

    return KVFootprint(
        logical_kv_tokens=logical_tokens,
        unique_physical_slots=unique_slots,
        slot_reuse_ratio=(
            0.0 if logical_tokens == 0 else 1.0 - unique_slots / logical_tokens
        ),
        logical_page_references=logical_page_refs,
        unique_physical_pages=unique_pages,
        page_reuse_ratio=(
            0.0 if logical_page_refs == 0 else 1.0 - unique_pages / logical_page_refs
        ),
    )


class RadixKVM0Recorder:
    def __init__(
        self,
        *,
        components: Sequence[str],
        device: str,
        page_size: int,
        radix_cache_enabled: bool,
        gpu_id: int,
        max_records: int = _MAX_RECORDS,
    ) -> None:
        self.components = resolve_components(components)
        self.enabled = bool(self.components)
        self.device = device
        self.page_size = int(page_size)
        self.radix_cache_enabled = bool(radix_cache_enabled)
        self.gpu_id = int(gpu_id)
        self.records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._pending_timings: deque[
            tuple[
                dict[str, Any],
                dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
            ]
        ] = deque()
        self._acceptance_queue: deque[dict[str, Any]] = deque()
        self._active_cycle_record: dict[str, Any] | None = None
        self._active_cycle_timings: (
            dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] | None
        ) = None
        self._record_id = 0

    def _new_record(
        self,
        *,
        batch: Any,
        speculative_num_steps: int,
    ) -> dict[str, Any]:
        self._record_id += 1
        seq_lens_cpu = [int(req.seqlen) for req in batch.reqs]
        record: dict[str, Any] = {
            "record_id": self._record_id,
            "forward_iter": int(getattr(batch, "forward_iter", -1)),
            "gpu_id": self.gpu_id,
            "batch_size": len(batch.reqs),
            "speculative_num_steps": int(speculative_num_steps),
            "logical_context_lengths": seq_lens_cpu,
            "logical_kv_tokens": sum(seq_lens_cpu),
            "radix_cache_enabled": self.radix_cache_enabled,
            "page_size": self.page_size,
            "request_ids": [str(req.rid) for req in batch.reqs],
            "correct_drafts_per_req": None,
        }
        self._acceptance_queue.append(record)
        return record

    def spec_cycle(
        self,
        *,
        batch: Any,
        speculative_num_steps: int,
    ) -> ContextManager[None]:
        if SPEC_CYCLE_GPU_TIME not in self.components:
            return nullcontext()
        return self._spec_cycle(
            batch=batch,
            speculative_num_steps=speculative_num_steps,
        )

    @contextmanager
    def _spec_cycle(
        self,
        *,
        batch: Any,
        speculative_num_steps: int,
    ) -> Iterator[None]:
        if self._active_cycle_record is not None:
            raise RuntimeError("nested speculative-cycle timing is not supported")
        record = self._new_record(
            batch=batch,
            speculative_num_steps=speculative_num_steps,
        )
        timings: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self._active_cycle_record = record
        self._active_cycle_timings = timings
        start.record()
        try:
            yield
        finally:
            end.record()
            timings["spec_cycle_gpu_ms"] = (start, end)
            self._pending_timings.append((record, timings))
            self._active_cycle_record = None
            self._active_cycle_timings = None

    def draft_stage(self) -> ContextManager[None]:
        return self._cycle_stage("draft_gpu_ms")

    def draft_extend_stage(self) -> ContextManager[None]:
        return self._cycle_stage("draft_extend_gpu_ms")

    @contextmanager
    def _cycle_stage(self, field: str) -> Iterator[None]:
        if SPEC_CYCLE_GPU_TIME not in self.components:
            yield
            return
        if self._active_cycle_timings is None:
            raise RuntimeError(f"{field} timing requires an active speculative cycle")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._active_cycle_timings[field] = (start, end)

    def target_verify(
        self,
        *,
        batch: Any,
        req_to_token_pool: Any,
        speculative_num_steps: int,
    ) -> ContextManager[None]:
        if not self.enabled:
            return nullcontext()
        return self._target_verify(
            batch=batch,
            req_to_token_pool=req_to_token_pool,
            speculative_num_steps=speculative_num_steps,
        )

    @contextmanager
    def _target_verify(
        self,
        *,
        batch: Any,
        req_to_token_pool: Any,
        speculative_num_steps: int,
    ) -> Iterator[None]:
        cycle_owned = self._active_cycle_record is not None
        record = self._active_cycle_record or self._new_record(
            batch=batch,
            speculative_num_steps=speculative_num_steps,
        )

        if KV_FOOTPRINT in self.components:
            footprint = compute_kv_footprint(
                req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                batch.seq_lens,
                page_size=self.page_size,
            )
            record.update(footprint.as_dict())

        start = end = None
        if self.components & _TIMING_COMPONENTS:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if start is not None and end is not None:
                end.record()
                if cycle_owned:
                    assert self._active_cycle_timings is not None
                    self._active_cycle_timings["target_verify_gpu_ms"] = (start, end)
                else:
                    self._pending_timings.append(
                        (record, {"target_verify_gpu_ms": (start, end)})
                    )
            elif cycle_owned:
                pass
            else:
                self.records.append(record)

    def observe_acceptance(self, correct_drafts_per_req: Sequence[int]) -> None:
        if not self.enabled:
            return
        while self._acceptance_queue:
            record = self._acceptance_queue.popleft()
            if record["correct_drafts_per_req"] is None:
                record["correct_drafts_per_req"] = [
                    int(value) for value in correct_drafts_per_req
                ]
                return

    def dump(self) -> dict[str, Any]:
        self._drain_timings()
        return {
            "schema_version": 1,
            "components": sorted(self.components),
            "radix_cache_enabled": self.radix_cache_enabled,
            "page_size": self.page_size,
            "gpu_id": self.gpu_id,
            "records": list(self.records),
        }

    def clear(self) -> None:
        self._drain_timings()
        self.records.clear()
        self._acceptance_queue.clear()
        self._active_cycle_record = None
        self._active_cycle_timings = None
        self._record_id = 0

    def _drain_timings(self) -> None:
        while self._pending_timings:
            record, timings = self._pending_timings.popleft()
            for field, (start, end) in timings.items():
                end.synchronize()
                record[field] = float(start.elapsed_time(end))
            if "spec_cycle_gpu_ms" in timings:
                record.setdefault("draft_gpu_ms", 0.0)
                record.setdefault("draft_extend_gpu_ms", 0.0)
                primary = record["draft_gpu_ms"] + record["target_verify_gpu_ms"]
                record["primary_spec_cycle_gpu_ms"] = primary
                record["unattributed_cycle_gpu_ms"] = (
                    record["spec_cycle_gpu_ms"]
                    - primary
                    - record["draft_extend_gpu_ms"]
                )
            self.records.append(record)
