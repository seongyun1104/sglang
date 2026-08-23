"""Isolated FA3 A/B/C reproducer for physical KV-page aliasing.

The benchmark holds the query, logical K/V contents, backing allocation shape,
kernel call signature, and stream constant. It changes only the physical page table
and the addresses receiving the same logical K/V pages. Primary timing uses CUDA
events without a profiler. See ``benchmark/fa3_kv_aliasing_i0/INVESTIGATION_CONTRACT.md``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ALIASED = "shared_aliased"
CONTIGUOUS = "duplicated_contiguous"
SCATTERED = "duplicated_scattered"
LAYOUTS = (ALIASED, CONTIGUOUS, SCATTERED)
CACHE_STATES = ("coldish", "warm")
BALANCED_ORDERS = tuple(itertools.permutations(LAYOUTS))
OUTPUT_RTOL = 2e-2
OUTPUT_ATOL = 2e-2


@dataclass(frozen=True)
class I0Config:
    model_id: str = "Qwen/Qwen3.8-27B"
    batch_size: int = 16
    context_length: int = 16384
    query_length: int = 1
    shared_prefix_ratio: float = 0.9
    page_size: int = 1
    num_query_heads: int = 24
    num_kv_heads: int = 4
    head_dim: int = 256
    dtype: str = "bfloat16"
    causal: bool = True
    num_splits: int = 0
    repetitions: int = 50
    warmup: int = 10
    l2_thrash_mib: int = 128
    minimum_effect_percent: float = 2.0

    def validate(self) -> None:
        if self.batch_size < 2:
            raise ValueError(
                "batch_size must be at least 2 to test inter-request aliasing"
            )
        if self.context_length <= 0 or self.context_length % self.page_size:
            raise ValueError("context_length must be positive and page-size aligned")
        if self.query_length <= 0:
            raise ValueError("query_length must be positive")
        if not 0.0 < self.shared_prefix_ratio < 1.0:
            raise ValueError("shared_prefix_ratio must be between zero and one")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if self.dtype != "bfloat16":
            raise ValueError("I0 is preregistered for BF16 only")
        if self.repetitions <= 0 or self.warmup < 0:
            raise ValueError("repetitions must be positive and warmup non-negative")
        if self.l2_thrash_mib <= 0:
            raise ValueError("l2_thrash_mib must be positive")
        if self.minimum_effect_percent <= 0:
            raise ValueError("minimum_effect_percent must be positive")
        if self.shared_pages <= 0 or self.tail_pages <= 0:
            raise ValueError(
                "shared prefix and request-specific tail need at least one page"
            )

    @property
    def pages_per_request(self) -> int:
        return self.context_length // self.page_size

    @property
    def shared_pages(self) -> int:
        return int(self.pages_per_request * self.shared_prefix_ratio)

    @property
    def tail_pages(self) -> int:
        return self.pages_per_request - self.shared_pages

    @property
    def total_backing_pages(self) -> int:
        # Deliberately identical for A/B/C even though A references fewer pages.
        return self.batch_size * self.pages_per_request


def _scattered_permutation(total_pages: int, seed: int) -> torch.Tensor:
    """Return a deterministic affine permutation with large physical jumps."""
    if total_pages < 2:
        raise ValueError("scattered layout requires at least two pages")
    stride = (int(total_pages * 0.61803398875) + 2 * int(seed) + 1) | 1
    stride %= total_pages
    if stride == 0:
        stride = 1
    while math.gcd(stride, total_pages) != 1:
        stride = (stride + 2) % total_pages
        if stride == 0:
            stride = 1
    offset = (int(seed) * 104729) % total_pages
    indices = torch.arange(total_pages, dtype=torch.int64)
    return ((indices * stride + offset) % total_pages).to(torch.int32)


def build_page_tables(
    config: I0Config, *, scatter_seed: int
) -> dict[str, torch.Tensor]:
    """Build A/B/C page tables over one equal-sized backing allocation."""
    config.validate()
    bs = config.batch_size
    pages = config.pages_per_request
    shared = config.shared_pages
    tail = config.tail_pages
    total = config.total_backing_pages

    aliased = torch.empty((bs, pages), dtype=torch.int32)
    aliased[:, :shared] = torch.arange(shared, dtype=torch.int32)
    for request_id in range(bs):
        start = shared + request_id * tail
        aliased[request_id, shared:] = torch.arange(
            start, start + tail, dtype=torch.int32
        )

    contiguous = torch.arange(total, dtype=torch.int32).view(bs, pages)
    scattered = _scattered_permutation(total, scatter_seed).view(bs, pages)
    tables = {
        ALIASED: aliased,
        CONTIGUOUS: contiguous,
        SCATTERED: scattered,
    }
    validate_page_tables(config, tables)
    return tables


def validate_page_tables(config: I0Config, tables: dict[str, torch.Tensor]) -> None:
    if set(tables) != set(LAYOUTS):
        raise ValueError(f"expected layouts {LAYOUTS}, got {sorted(tables)}")
    expected_shape = (config.batch_size, config.pages_per_request)
    total = config.total_backing_pages
    for layout, table in tables.items():
        if tuple(table.shape) != expected_shape or table.dtype != torch.int32:
            raise ValueError(f"invalid {layout} table shape or dtype")
        if int(table.min()) < 0 or int(table.max()) >= total:
            raise ValueError(f"{layout} page id outside backing allocation")

    aliased = tables[ALIASED]
    prefix = aliased[:, : config.shared_pages]
    if not torch.equal(prefix, prefix[0].expand_as(prefix)):
        raise ValueError("aliased prefix rows do not reference identical pages")
    aliased_tails = aliased[:, config.shared_pages :].reshape(-1)
    if torch.unique(aliased_tails).numel() != aliased_tails.numel():
        raise ValueError("aliased request tails must remain physically distinct")
    if bool(torch.isin(aliased_tails, prefix[0]).any().item()):
        raise ValueError("aliased tail overlaps the shared prefix")

    for layout in (CONTIGUOUS, SCATTERED):
        flat = tables[layout].reshape(-1)
        if torch.unique(flat).numel() != flat.numel():
            raise ValueError(f"{layout} must assign one distinct page per reference")
    if not torch.all(torch.diff(tables[CONTIGUOUS], dim=1) == 1):
        raise ValueError("contiguous layout is not contiguous within each request")
    if torch.all(torch.abs(torch.diff(tables[SCATTERED], dim=1)) == 1):
        raise ValueError("scattered layout accidentally remained contiguous")


def page_table_statistics(table: torch.Tensor) -> dict[str, Any]:
    flat = table.reshape(-1).to(torch.int64)
    unique = int(torch.unique(flat).numel())
    refs = int(flat.numel())
    jumps = torch.abs(torch.diff(table.to(torch.int64), dim=1)).reshape(-1)
    return {
        "physical_page_references": refs,
        "unique_physical_pages": unique,
        "page_reuse_ratio": 1.0 - unique / refs,
        "adjacent_step_ratio": float((jumps == 1).double().mean()),
        "mean_absolute_page_jump": float(jumps.double().mean()),
        "minimum_page_id": int(flat.min()),
        "maximum_page_id": int(flat.max()),
        "page_table_sha256": hashlib.sha256(table.numpy().tobytes()).hexdigest(),
    }


def build_plan_metadata(config: I0Config, seeds: Sequence[int]) -> dict[str, Any]:
    seeds = _validated_seeds(seeds)
    return {
        "config": asdict(config),
        "layouts": list(LAYOUTS),
        "cache_states": list(CACHE_STATES),
        "orders": [list(order) for order in BALANCED_ORDERS],
        "seeds": [int(seed) for seed in seeds],
        "page_tables": {
            str(seed): {
                layout: page_table_statistics(table)
                for layout, table in build_page_tables(
                    config, scatter_seed=int(seed)
                ).items()
            }
            for seed in seeds
        },
        "timed_region": "one flash_attn_with_kvcache call",
        "empirical_launch_geometry": "PENDING_SEPARATE_UNTIMED_CAPTURE",
    }


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("at least one seed is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique")
    return normalized


@dataclass
class LogicalKV:
    prefix_k: torch.Tensor
    prefix_v: torch.Tensor
    tail_k: torch.Tensor
    tail_v: torch.Tensor


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _make_logical_kv(config: I0Config, *, seed: int, device: torch.device) -> LogicalKV:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    dtype = _torch_dtype(config.dtype)
    common_shape = (config.page_size, config.num_kv_heads, config.head_dim)
    prefix_shape = (config.shared_pages, *common_shape)
    tail_shape = (config.batch_size, config.tail_pages, *common_shape)
    return LogicalKV(
        prefix_k=torch.randn(
            prefix_shape, generator=generator, device=device, dtype=dtype
        ),
        prefix_v=torch.randn(
            prefix_shape, generator=generator, device=device, dtype=dtype
        ),
        tail_k=torch.randn(tail_shape, generator=generator, device=device, dtype=dtype),
        tail_v=torch.randn(tail_shape, generator=generator, device=device, dtype=dtype),
    )


def _materialize_layout(
    *,
    config: I0Config,
    layout: str,
    page_table: torch.Tensor,
    logical: LogicalKV,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    shared = config.shared_pages
    if layout == ALIASED:
        prefix_ids = page_table[0, :shared].to(torch.int64)
        k_cache.index_copy_(0, prefix_ids, logical.prefix_k)
        v_cache.index_copy_(0, prefix_ids, logical.prefix_v)
        for request_id in range(config.batch_size):
            tail_ids = page_table[request_id, shared:].to(torch.int64)
            k_cache.index_copy_(0, tail_ids, logical.tail_k[request_id])
            v_cache.index_copy_(0, tail_ids, logical.tail_v[request_id])
        return

    for request_id in range(config.batch_size):
        prefix_ids = page_table[request_id, :shared].to(torch.int64)
        tail_ids = page_table[request_id, shared:].to(torch.int64)
        k_cache.index_copy_(0, prefix_ids, logical.prefix_k)
        v_cache.index_copy_(0, prefix_ids, logical.prefix_v)
        k_cache.index_copy_(0, tail_ids, logical.tail_k[request_id])
        v_cache.index_copy_(0, tail_ids, logical.tail_v[request_id])


def _call_fa3(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    config: I0Config,
) -> torch.Tensor:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    result = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=config.query_length,
        softmax_scale=1.0 / math.sqrt(config.head_dim),
        causal=config.causal,
        window_size=(-1, -1),
        num_splits=config.num_splits,
        return_softmax_lse=False,
        ver=3,
    )
    return result[0] if isinstance(result, (tuple, list)) else result


def _time_one_call(callable_: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _effect_percent(faster_ms: float, baseline_ms: float) -> float:
    return (baseline_ms - faster_ms) / baseline_ms * 100.0


def analyze_rows(
    rows: Sequence[dict[str, Any]],
    *,
    config: I0Config,
    seeds: Sequence[int],
    outputs_valid: bool,
) -> dict[str, Any]:
    try:
        seeds = _validated_seeds(seeds)
    except ValueError:
        return {
            "i0_status": "I0_INCOMPLETE",
            "expected_samples": 0,
            "observed_samples": len(rows),
            "outputs_valid": outputs_valid,
        }
    expected = (
        len(seeds)
        * len(CACHE_STATES)
        * len(BALANCED_ORDERS)
        * len(LAYOUTS)
        * config.repetitions
    )
    if len(rows) != expected:
        return {
            "i0_status": "I0_INCOMPLETE",
            "expected_samples": expected,
            "observed_samples": len(rows),
            "outputs_valid": outputs_valid,
        }

    expected_cells = {
        (
            seed,
            cache_state,
            order_index,
            ">".join(order),
            position,
            layout,
            repetition,
        )
        for seed in seeds
        for cache_state in CACHE_STATES
        for order_index, order in enumerate(BALANCED_ORDERS)
        for position, layout in enumerate(order)
        for repetition in range(config.repetitions)
    }
    observed_cells: list[tuple[int, str, int, str, int, str, int]] = []
    malformed_rows = 0
    invalid_timings = 0
    for row in rows:
        try:
            cell = (
                int(row["seed"]),
                str(row["cache_state"]),
                int(row["order_index"]),
                str(row["order"]),
                int(row["position"]),
                str(row["layout"]),
                int(row["repetition"]),
            )
            latency_ms = float(row["latency_ms"])
        except (KeyError, TypeError, ValueError):
            malformed_rows += 1
            continue
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            invalid_timings += 1
        observed_cells.append(cell)
    observed_cell_set = set(observed_cells)
    if (
        malformed_rows
        or len(observed_cell_set) != len(observed_cells)
        or observed_cell_set != expected_cells
    ):
        return {
            "i0_status": "I0_INCOMPLETE",
            "expected_samples": expected,
            "observed_samples": len(rows),
            "unique_observed_cells": len(observed_cell_set),
            "missing_cells": len(expected_cells - observed_cell_set),
            "unexpected_cells": len(observed_cell_set - expected_cells),
            "malformed_rows": malformed_rows,
            "outputs_valid": outputs_valid,
        }
    if invalid_timings:
        return {
            "i0_status": "I0_INVALID",
            "expected_samples": expected,
            "observed_samples": len(rows),
            "invalid_timings": invalid_timings,
            "outputs_valid": outputs_valid,
        }
    if not outputs_valid:
        return {
            "i0_status": "I0_INVALID",
            "expected_samples": expected,
            "observed_samples": len(rows),
            "outputs_valid": False,
        }

    aggregate: dict[str, dict[str, float]] = {}
    for cache_state in CACHE_STATES:
        aggregate[cache_state] = {}
        for layout in LAYOUTS:
            samples = [
                float(row["latency_ms"])
                for row in rows
                if row["cache_state"] == cache_state and row["layout"] == layout
            ]
            aggregate[cache_state][layout] = statistics.median(samples)

    stratum_effects = []
    for cache_state in CACHE_STATES:
        for seed in seeds:
            for order_index in range(len(BALANCED_ORDERS)):
                medians = {}
                for layout in LAYOUTS:
                    samples = [
                        float(row["latency_ms"])
                        for row in rows
                        if row["cache_state"] == cache_state
                        and int(row["seed"]) == int(seed)
                        and int(row["order_index"]) == order_index
                        and row["layout"] == layout
                    ]
                    medians[layout] = statistics.median(samples)
                stratum_effects.append(
                    {
                        "cache_state": cache_state,
                        "seed": int(seed),
                        "order_index": order_index,
                        "alias_speedup_percent": _effect_percent(
                            medians[ALIASED], medians[CONTIGUOUS]
                        ),
                        "scattering_penalty_percent": _effect_percent(
                            medians[CONTIGUOUS], medians[SCATTERED]
                        ),
                    }
                )

    warm = aggregate["warm"]
    alias_effect = _effect_percent(warm[ALIASED], warm[CONTIGUOUS])
    scattering_effect = _effect_percent(warm[CONTIGUOUS], warm[SCATTERED])
    warm_strata = [row for row in stratum_effects if row["cache_state"] == "warm"]
    required_support = math.ceil(len(warm_strata) * 2 / 3)
    alias_support = sum(
        row["alias_speedup_percent"] >= config.minimum_effect_percent
        for row in warm_strata
    )
    scattering_support = sum(
        row["scattering_penalty_percent"] >= config.minimum_effect_percent
        for row in warm_strata
    )
    alias_pass = alias_effect >= config.minimum_effect_percent
    scattering_pass = scattering_effect >= config.minimum_effect_percent
    unexpected_direction = (
        alias_effect <= -config.minimum_effect_percent
        or scattering_effect <= -config.minimum_effect_percent
    )

    if unexpected_direction:
        status = "I0_UNEXPECTED_DIRECTION"
    elif (alias_pass and alias_support < required_support) or (
        scattering_pass and scattering_support < required_support
    ):
        status = "I0_ORDER_SENSITIVE"
    elif alias_pass and scattering_pass:
        status = "I0_MIXED_CANDIDATE"
    elif alias_pass:
        status = "I0_ALIASING_CANDIDATE"
    elif scattering_pass:
        status = "I0_LOCALITY_CANDIDATE"
    else:
        status = "I0_NO_ISOLATED_SIGNAL"

    return {
        "i0_status": status,
        "expected_samples": expected,
        "observed_samples": len(rows),
        "outputs_valid": True,
        "minimum_effect_percent": config.minimum_effect_percent,
        "aggregate_latency_ms": aggregate,
        "warm_alias_speedup_percent": alias_effect,
        "warm_scattering_penalty_percent": scattering_effect,
        "required_stratum_support": required_support,
        "alias_supporting_strata": alias_support,
        "scattering_supporting_strata": scattering_support,
        "stratum_effects": stratum_effects,
        "launch_geometry_status": "PENDING_SEPARATE_UNTIMED_CAPTURE",
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write empty raw-sample CSV")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_i0(
    config: I0Config, *, seeds: Sequence[int], output_dir: Path
) -> dict[str, Any]:
    config.validate()
    seeds = _validated_seeds(seeds)
    if not torch.cuda.is_available():
        raise RuntimeError("I0 requires a CUDA GPU")
    major, _ = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    if major != 9 or "H100" not in device_name:
        raise RuntimeError("I0 anchor requires an NVIDIA H100 GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = _torch_dtype(config.dtype)
    plan = build_plan_metadata(config, seeds)
    (output_dir / "i0-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )

    raw_rows: list[dict[str, Any]] = []
    output_checks: list[dict[str, Any]] = []
    seed_invariants: list[dict[str, Any]] = []
    thrash_elements = config.l2_thrash_mib * 1024 * 1024 // 4
    thrash = torch.ones((thrash_elements,), dtype=torch.float32, device=device)

    for seed in seeds:
        tables_cpu = build_page_tables(config, scatter_seed=int(seed))
        tables = {layout: table.to(device) for layout, table in tables_cpu.items()}
        logical = _make_logical_kv(config, seed=int(seed), device=device)
        generator = torch.Generator(device=device).manual_seed(int(seed) + 1_000_003)
        q = torch.randn(
            (
                config.batch_size,
                config.query_length,
                config.num_query_heads,
                config.head_dim,
            ),
            generator=generator,
            dtype=dtype,
            device=device,
        ).view(
            config.batch_size * config.query_length,
            config.num_query_heads,
            config.head_dim,
        )
        cu_seqlens_q = torch.arange(
            0,
            (config.batch_size + 1) * config.query_length,
            config.query_length,
            dtype=torch.int32,
            device=device,
        )
        cache_seqlens = torch.full(
            (config.batch_size,),
            config.context_length,
            dtype=torch.int32,
            device=device,
        )
        cache_shape = (
            config.total_backing_pages,
            config.page_size,
            config.num_kv_heads,
            config.head_dim,
        )
        k_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        v_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        seed_invariants.append(
            {
                "seed": int(seed),
                "query_data_ptr": q.data_ptr(),
                "logical_prefix_k_data_ptr": logical.prefix_k.data_ptr(),
                "logical_prefix_v_data_ptr": logical.prefix_v.data_ptr(),
                "logical_tail_k_data_ptr": logical.tail_k.data_ptr(),
                "logical_tail_v_data_ptr": logical.tail_v.data_ptr(),
                "backing_k_data_ptr": k_cache.data_ptr(),
                "backing_v_data_ptr": v_cache.data_ptr(),
                "backing_tensor_bytes_each": k_cache.numel() * k_cache.element_size(),
                "reuse_rule": "same tensor objects are reused for every A/B/C arm",
            }
        )

        outputs: dict[str, torch.Tensor] = {}
        for layout in LAYOUTS:
            _materialize_layout(
                config=config,
                layout=layout,
                page_table=tables[layout],
                logical=logical,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            torch.cuda.synchronize()
            outputs[layout] = (
                _call_fa3(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    page_table=tables[layout],
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    config=config,
                )
                .detach()
                .clone()
            )
            torch.cuda.synchronize()

        reference = outputs[ALIASED]
        for layout in (CONTIGUOUS, SCATTERED):
            difference = (reference.float() - outputs[layout].float()).abs()
            output_checks.append(
                {
                    "seed": int(seed),
                    "layout": layout,
                    "allclose": bool(
                        torch.allclose(
                            reference,
                            outputs[layout],
                            rtol=OUTPUT_RTOL,
                            atol=OUTPUT_ATOL,
                        )
                    ),
                    "maximum_absolute_difference": float(difference.max()),
                    "mean_absolute_difference": float(difference.mean()),
                }
            )

        for cache_state in CACHE_STATES:
            for order_index, order in enumerate(BALANCED_ORDERS):
                for position, layout in enumerate(order):
                    _materialize_layout(
                        config=config,
                        layout=layout,
                        page_table=tables[layout],
                        logical=logical,
                        k_cache=k_cache,
                        v_cache=v_cache,
                    )
                    torch.cuda.synchronize()

                    call = functools.partial(
                        _call_fa3,
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        page_table=tables[layout],
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        config=config,
                    )

                    for _ in range(config.warmup):
                        if cache_state == "coldish":
                            thrash.add_(1.0)
                            torch.cuda.synchronize()
                        call()
                    torch.cuda.synchronize()

                    for repetition in range(config.repetitions):
                        if cache_state == "coldish":
                            thrash.add_(1.0)
                            torch.cuda.synchronize()
                        latency_ms = _time_one_call(call)
                        raw_rows.append(
                            {
                                "seed": int(seed),
                                "cache_state": cache_state,
                                "order_index": order_index,
                                "order": ">".join(order),
                                "position": position,
                                "layout": layout,
                                "repetition": repetition,
                                "latency_ms": latency_ms,
                            }
                        )

        del logical, q, cu_seqlens_q, cache_seqlens, k_cache, v_cache, outputs, tables
        torch.cuda.empty_cache()

    outputs_valid = all(row["allclose"] for row in output_checks)
    summary = analyze_rows(
        raw_rows, config=config, seeds=seeds, outputs_valid=outputs_valid
    )
    summary["config"] = asdict(config)
    summary["seeds"] = [int(seed) for seed in seeds]
    summary["output_checks"] = output_checks
    summary["seed_invariants"] = seed_invariants
    summary["device"] = {
        "name": device_name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "current_stream": int(torch.cuda.current_stream().cuda_stream),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    summary["static_call_signature"] = {
        "callable": "sgl_kernel.flash_attn.flash_attn_with_kvcache",
        "q_shape": [
            config.batch_size * config.query_length,
            config.num_query_heads,
            config.head_dim,
        ],
        "kv_cache_shape": [
            config.total_backing_pages,
            config.page_size,
            config.num_kv_heads,
            config.head_dim,
        ],
        "page_table_shape": [config.batch_size, config.pages_per_request],
        "cache_seqlens_shape": [config.batch_size],
        "cu_seqlens_q_shape": [config.batch_size + 1],
        "dtype": config.dtype,
        "causal": config.causal,
        "window_size": [-1, -1],
        "max_seqlen_q": config.query_length,
        "softmax_scale": 1.0 / math.sqrt(config.head_dim),
        "num_splits": config.num_splits,
        "fa_version": 3,
        "return_softmax_lse": False,
        "cuda_graph": False,
        "calls_per_timed_interval": 1,
        "output_rtol": OUTPUT_RTOL,
        "output_atol": OUTPUT_ATOL,
    }
    _write_csv(output_dir / "i0-raw-samples.csv", raw_rows)
    (output_dir / "i0-output-checks.json").write_text(
        json.dumps(output_checks, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "i0-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _config_from_args(args: argparse.Namespace) -> I0Config:
    return I0Config(
        model_id=args.model_id,
        batch_size=args.batch_size,
        context_length=args.context_length,
        query_length=args.query_length,
        shared_prefix_ratio=args.shared_prefix_ratio,
        page_size=args.page_size,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        repetitions=args.repetitions,
        warmup=args.warmup,
        l2_thrash_mib=args.l2_thrash_mib,
        minimum_effect_percent=args.minimum_effect_percent,
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.9)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--l2-thrash-mib", type=int, default=128)
    parser.add_argument("--minimum-effect-percent", type=float, default=2.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    _add_common_arguments(plan_parser)
    plan_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    config.validate()
    if args.command == "plan":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                build_plan_metadata(config, args.seeds), indent=2, sort_keys=True
            )
            + "\n"
        )
        return 0
    summary = run_i0(config, seeds=args.seeds, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
