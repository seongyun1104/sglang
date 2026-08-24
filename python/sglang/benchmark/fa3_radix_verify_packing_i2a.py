"""I2a falsifier for Radix-local row order inside FA3 target verification.

This benchmark does not alter request admission, speculative depth, physical KV
pages, or logical inputs.  It permutes only the request-row order presented to one
FA3 call and restores canonical order before checking outputs.
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


CLUSTERED = "radix_clustered"
INTERLEAVED = "radix_interleaved"
RANDOM = "random"
ROW_ORDERS = (CLUSTERED, INTERLEAVED, RANDOM)
BALANCED_ARM_ORDERS = tuple(itertools.permutations(ROW_ORDERS))
DEFAULT_CONTEXT_LENGTHS = (8192, 16384)
DEFAULT_SPECULATIVE_STEPS = (1, 2, 4)


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


@dataclass(frozen=True)
class I2AConfig:
    model_id: str = "Qwen/Qwen3.8-27B"
    batch_size: int = 16
    group_count: int = 4
    shared_prefix_ratio: float = 0.9
    page_size: int = 1
    num_query_heads: int = 24
    num_kv_heads: int = 4
    head_dim: int = 256
    dtype: str = "bfloat16"
    repetitions: int = 50
    warmup: int = 10
    minimum_effect_percent: float = 2.0

    def validate(self) -> None:
        if self.batch_size <= 0 or self.group_count <= 1:
            raise ValueError("batch_size must be positive and group_count > 1")
        if self.batch_size % self.group_count:
            raise ValueError("batch_size must be divisible by group_count")
        if not 0.0 < self.shared_prefix_ratio < 1.0:
            raise ValueError("shared_prefix_ratio must be between zero and one")
        if self.page_size != 1:
            raise ValueError("I2a currently requires page_size=1")
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError(
                "attention head counts and head dimension must be positive"
            )
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.dtype != "bfloat16":
            raise ValueError("I2a is preregistered for BF16 only")
        if self.repetitions <= 0 or self.warmup < 0:
            raise ValueError("invalid repetition or warmup count")
        if self.minimum_effect_percent <= 0:
            raise ValueError("minimum_effect_percent must be positive")

    @property
    def requests_per_group(self) -> int:
        return self.batch_size // self.group_count


def verify_width(speculative_steps: int) -> int:
    if speculative_steps <= 0:
        raise ValueError("speculative_steps must be positive")
    # SGLang's top-k=1 EAGLE chain pins target-verify width to K + 1.
    return speculative_steps + 1


def shared_pages(config: I2AConfig, context_length: int, speculative_steps: int) -> int:
    verify_width(speculative_steps)
    if context_length <= 1:
        raise ValueError("committed context length must exceed one page")
    pages = int(context_length * config.shared_prefix_ratio)
    return min(max(pages, 1), context_length - 1)


def request_group_ids(config: I2AConfig) -> tuple[int, ...]:
    return tuple(
        request_id // config.requests_per_group
        for request_id in range(config.batch_size)
    )


def row_permutations(config: I2AConfig, *, seed: int) -> dict[str, torch.Tensor]:
    config.validate()
    clustered = torch.arange(config.batch_size, dtype=torch.int64)
    interleaved = torch.tensor(
        [
            group * config.requests_per_group + offset
            for offset in range(config.requests_per_group)
            for group in range(config.group_count)
        ],
        dtype=torch.int64,
    )
    generator = torch.Generator().manual_seed(int(seed) + 7_919)
    random = torch.randperm(config.batch_size, generator=generator)
    if torch.equal(random, clustered) or torch.equal(random, interleaved):
        raise ValueError("random row order collided with a controlled arm")
    result = {CLUSTERED: clustered, INTERLEAVED: interleaved, RANDOM: random}
    validate_row_permutations(config, result)
    return result


def validate_row_permutations(
    config: I2AConfig, permutations: dict[str, torch.Tensor]
) -> None:
    if set(permutations) != set(ROW_ORDERS):
        raise ValueError("missing row-order arm")
    expected = torch.arange(config.batch_size, dtype=torch.int64)
    for name, permutation in permutations.items():
        if permutation.dtype != torch.int64 or permutation.shape != expected.shape:
            raise ValueError(f"invalid permutation shape or dtype for {name}")
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError(f"{name} is not a full request permutation")


def group_adjacency_ratio(config: I2AConfig, permutation: torch.Tensor) -> float:
    groups = torch.tensor(request_group_ids(config), dtype=torch.int64)[permutation]
    return float((groups[1:] == groups[:-1]).double().mean())


def build_canonical_page_table(
    config: I2AConfig, *, context_length: int, speculative_steps: int
) -> torch.Tensor:
    config.validate()
    shared = shared_pages(config, context_length, speculative_steps)
    target_kv_length = context_length + verify_width(speculative_steps)
    tail = target_kv_length - shared
    prefix_base = 0
    tail_base = config.group_count * shared
    rows = []
    for request_id, group_id in enumerate(request_group_ids(config)):
        prefix = torch.arange(
            prefix_base + group_id * shared,
            prefix_base + (group_id + 1) * shared,
            dtype=torch.int32,
        )
        suffix = torch.arange(
            tail_base + request_id * tail,
            tail_base + (request_id + 1) * tail,
            dtype=torch.int32,
        )
        rows.append(torch.cat((prefix, suffix)))
    table = torch.stack(rows)
    validate_page_table(
        config,
        table,
        context_length=context_length,
        speculative_steps=speculative_steps,
    )
    return table


def validate_page_table(
    config: I2AConfig,
    table: torch.Tensor,
    *,
    context_length: int,
    speculative_steps: int,
) -> None:
    target_kv_length = context_length + verify_width(speculative_steps)
    if (
        table.shape != (config.batch_size, target_kv_length)
        or table.dtype != torch.int32
    ):
        raise ValueError("invalid canonical page table")
    shared = shared_pages(config, context_length, speculative_steps)
    groups = request_group_ids(config)
    for left in range(config.batch_size):
        for right in range(left + 1, config.batch_size):
            same_group = groups[left] == groups[right]
            prefix_equal = torch.equal(table[left, :shared], table[right, :shared])
            if prefix_equal != same_group:
                raise ValueError("prefix aliases do not match Radix groups")
            if bool(torch.isin(table[left, shared:], table[right, shared:]).any()):
                raise ValueError("request-specific suffix pages overlap")


def page_table_statistics(table: torch.Tensor) -> dict[str, Any]:
    flat = table.reshape(-1).to(torch.int64)
    unique = int(torch.unique(flat).numel())
    refs = int(flat.numel())
    return {
        "physical_page_references": refs,
        "unique_physical_pages": unique,
        "page_reuse_ratio": 1.0 - unique / refs,
        "page_table_sha256": hashlib.sha256(table.numpy().tobytes()).hexdigest(),
    }


def build_plan(
    config: I2AConfig,
    *,
    contexts: Sequence[int],
    speculative_steps: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    config.validate()
    contexts, steps, seeds = validate_axes(
        contexts=contexts,
        speculative_steps=speculative_steps,
        seeds=seeds,
        require_primary_anchor=False,
    )
    cells: dict[str, Any] = {}
    for context_length in contexts:
        for depth in steps:
            table = build_canonical_page_table(
                config, context_length=context_length, speculative_steps=depth
            )
            key = f"ctx{context_length}-k{depth}"
            cells[key] = {
                "context_length": context_length,
                "target_kv_length": context_length + verify_width(depth),
                "speculative_steps": depth,
                "target_verify_width": verify_width(depth),
                "shared_pages_per_group": shared_pages(config, context_length, depth),
                "canonical_page_table": page_table_statistics(table),
            }
    permutations = {
        str(seed): {
            name: {
                "request_ids": permutation.tolist(),
                "group_adjacency_ratio": group_adjacency_ratio(config, permutation),
            }
            for name, permutation in row_permutations(config, seed=seed).items()
        }
        for seed in seeds
    }
    return {
        "config": asdict(config),
        "contexts": list(contexts),
        "speculative_steps": list(steps),
        "seeds": list(seeds),
        "expected_samples": (
            len(contexts)
            * len(steps)
            * len(seeds)
            * len(BALANCED_ARM_ORDERS)
            * len(ROW_ORDERS)
            * config.repetitions
        ),
        "expected_output_checks": len(contexts) * len(steps) * len(seeds) * 2,
        "row_orders": list(ROW_ORDERS),
        "balanced_arm_orders": [list(order) for order in BALANCED_ARM_ORDERS],
        "primary_anchor": {"context_length": 16384, "speculative_steps": 4},
        "cells": cells,
        "permutations": permutations,
        "semantic_contract": {
            "target_verify_width": "speculative_steps + 1",
            "context_length": "committed prefix before candidate KV",
            "target_kv_length": "context_length + target_verify_width",
            "candidate_kv_present_in_cache": True,
            "only_timed_operation": "one FA3 call",
            "changed_variable": "request row order only",
        },
    }


def validate_axes(
    *,
    contexts: Sequence[int],
    speculative_steps: Sequence[int],
    seeds: Sequence[int],
    require_primary_anchor: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    normalized_contexts = tuple(int(value) for value in contexts)
    normalized_steps = tuple(int(value) for value in speculative_steps)
    normalized_seeds = tuple(int(value) for value in seeds)
    if not normalized_contexts or any(value <= 1 for value in normalized_contexts):
        raise ValueError("positive committed contexts greater than one are required")
    if not normalized_steps or any(value <= 0 for value in normalized_steps):
        raise ValueError("positive speculative steps are required")
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    for name, values in (
        ("contexts", normalized_contexts),
        ("speculative steps", normalized_steps),
        ("seeds", normalized_seeds),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be unique")
    if require_primary_anchor and (
        16384 not in normalized_contexts or 4 not in normalized_steps
    ):
        raise ValueError("hardware evidence requires the preregistered 16K/K=4 anchor")
    return normalized_contexts, normalized_steps, normalized_seeds


def _dtype(name: str) -> torch.dtype:
    if name != "bfloat16":
        raise ValueError(f"unsupported dtype: {name}")
    return torch.bfloat16


def _materialize_cache(
    config: I2AConfig,
    *,
    table: torch.Tensor,
    context_length: int,
    speculative_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shared = shared_pages(config, context_length, speculative_steps)
    tail = context_length + verify_width(speculative_steps) - shared
    total_pages = config.group_count * shared + config.batch_size * tail
    shape = (total_pages, config.page_size, config.num_kv_heads, config.head_dim)
    k_cache = torch.empty(shape, dtype=_dtype(config.dtype), device=device)
    v_cache = torch.empty_like(k_cache)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    group_shape = (
        config.group_count,
        shared,
        config.page_size,
        config.num_kv_heads,
        config.head_dim,
    )
    tail_shape = (
        config.batch_size,
        tail,
        config.page_size,
        config.num_kv_heads,
        config.head_dim,
    )
    prefix_k = torch.randn(
        group_shape, generator=generator, device=device, dtype=k_cache.dtype
    )
    prefix_v = torch.randn(
        group_shape, generator=generator, device=device, dtype=k_cache.dtype
    )
    tail_k = torch.randn(
        tail_shape, generator=generator, device=device, dtype=k_cache.dtype
    )
    tail_v = torch.randn(
        tail_shape, generator=generator, device=device, dtype=k_cache.dtype
    )
    table_device = table.to(device=device, dtype=torch.int64)
    groups = request_group_ids(config)
    for request_id, group_id in enumerate(groups):
        if request_id % config.requests_per_group == 0:
            prefix_ids = table_device[request_id, :shared]
            k_cache.index_copy_(0, prefix_ids, prefix_k[group_id])
            v_cache.index_copy_(0, prefix_ids, prefix_v[group_id])
        suffix_ids = table_device[request_id, shared:]
        k_cache.index_copy_(0, suffix_ids, tail_k[request_id])
        v_cache.index_copy_(0, suffix_ids, tail_v[request_id])
    return k_cache, v_cache, table_device.to(torch.int32)


def _call_fa3(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    config: I2AConfig,
) -> torch.Tensor:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    result = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=1.0 / math.sqrt(config.head_dim),
        causal=True,
        window_size=(-1, -1),
        num_splits=0,
        return_softmax_lse=False,
        ver=3,
    )
    return result[0] if isinstance(result, (tuple, list)) else result


def _time_one(callable_: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write empty I2a CSV")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_rows(
    rows: Sequence[dict[str, Any]],
    *,
    config: I2AConfig,
    contexts: Sequence[int],
    speculative_steps: Sequence[int],
    seeds: Sequence[int],
    outputs_valid: bool,
) -> dict[str, Any]:
    expected = (
        len(contexts)
        * len(speculative_steps)
        * len(seeds)
        * len(BALANCED_ARM_ORDERS)
        * len(ROW_ORDERS)
        * config.repetitions
    )
    if len(rows) != expected:
        return {
            "i2a_status": "I2A_INCOMPLETE",
            "expected": expected,
            "observed": len(rows),
        }
    if not outputs_valid:
        return {
            "i2a_status": "I2A_INVALID",
            "expected": expected,
            "observed": len(rows),
        }

    expected_cells = {
        (
            int(context_length),
            int(depth),
            int(seed),
            order_index,
            ">".join(arm_order),
            position,
            arm,
            repetition,
        )
        for context_length in contexts
        for depth in speculative_steps
        for seed in seeds
        for order_index, arm_order in enumerate(BALANCED_ARM_ORDERS)
        for position, arm in enumerate(arm_order)
        for repetition in range(config.repetitions)
    }
    observed_cells = []
    invalid_timing = False
    for row in rows:
        observed_cells.append(
            (
                int(row["context_length"]),
                int(row["speculative_steps"]),
                int(row["seed"]),
                int(row["order_index"]),
                str(row["arm_order"]),
                int(row["position"]),
                str(row["row_order"]),
                int(row["repetition"]),
            )
        )
        latency = float(row["latency_ms"])
        invalid_timing |= not math.isfinite(latency) or latency <= 0
    observed_set = set(observed_cells)
    if (
        len(observed_set) != len(observed_cells)
        or observed_set != expected_cells
        or invalid_timing
    ):
        return {
            "i2a_status": "I2A_INVALID" if invalid_timing else "I2A_INCOMPLETE",
            "expected": expected,
            "observed": len(rows),
            "missing_cells": len(expected_cells - observed_set),
            "unexpected_cells": len(observed_set - expected_cells),
            "duplicate_cells": len(observed_cells) - len(observed_set),
            "invalid_timing": invalid_timing,
        }

    cell_results = []
    for context_length in contexts:
        for depth in speculative_steps:
            medians = {}
            for arm in ROW_ORDERS:
                values = [
                    float(row["latency_ms"])
                    for row in rows
                    if int(row["context_length"]) == int(context_length)
                    and int(row["speculative_steps"]) == int(depth)
                    and row["row_order"] == arm
                ]
                medians[arm] = statistics.median(values)
            effect = (
                (medians[INTERLEAVED] - medians[CLUSTERED])
                / medians[INTERLEAVED]
                * 100.0
            )
            block_medians = {}
            for seed in seeds:
                for order_index in range(len(BALANCED_ARM_ORDERS)):
                    for arm in ROW_ORDERS:
                        values = [
                            float(row["latency_ms"])
                            for row in rows
                            if int(row["context_length"]) == int(context_length)
                            and int(row["speculative_steps"]) == int(depth)
                            and int(row["seed"]) == int(seed)
                            and int(row["order_index"]) == order_index
                            and row["row_order"] == arm
                        ]
                        block_medians[(int(seed), order_index, arm)] = (
                            statistics.median(values)
                        )
            same_arm_differences = []
            for seed in seeds:
                for arm in ROW_ORDERS:
                    repeated = [
                        block_medians[(int(seed), order_index, arm)]
                        for order_index in range(len(BALANCED_ARM_ORDERS))
                    ]
                    for left, right in itertools.combinations(repeated, 2):
                        denominator = statistics.median((left, right))
                        same_arm_differences.append(
                            abs(left - right) / denominator * 100.0
                        )
            same_arm_p95_noise = _nearest_rank_percentile(same_arm_differences, 0.95)
            resolved_effect_floor = max(
                config.minimum_effect_percent, same_arm_p95_noise
            )
            support = 0
            for seed in seeds:
                for order_index in range(len(BALANCED_ARM_ORDERS)):
                    by_arm = {
                        arm: block_medians[(int(seed), order_index, arm)]
                        for arm in ROW_ORDERS
                    }
                    local_effect = (
                        (by_arm[INTERLEAVED] - by_arm[CLUSTERED])
                        / by_arm[INTERLEAVED]
                        * 100.0
                    )
                    support += local_effect >= resolved_effect_floor
            cell_results.append(
                {
                    "context_length": int(context_length),
                    "speculative_steps": int(depth),
                    "target_verify_width": verify_width(int(depth)),
                    "median_latency_ms": medians,
                    "clustered_vs_interleaved_percent": effect,
                    "preregistered_effect_floor_percent": (
                        config.minimum_effect_percent
                    ),
                    "same_arm_p95_noise_percent": same_arm_p95_noise,
                    "resolved_effect_floor_percent": resolved_effect_floor,
                    "supporting_strata": support,
                    "required_support": math.ceil(
                        len(seeds) * len(BALANCED_ARM_ORDERS) * 2 / 3
                    ),
                }
            )

    anchor = next(
        (
            cell
            for cell in cell_results
            if cell["context_length"] == 16384 and cell["speculative_steps"] == 4
        ),
        None,
    )
    if anchor is None:
        return {
            "i2a_status": "I2A_INCOMPLETE",
            "expected": expected,
            "observed": len(rows),
            "outputs_valid": True,
            "missing_primary_anchor": {
                "context_length": 16384,
                "speculative_steps": 4,
            },
            "cells": cell_results,
        }
    powered = (
        anchor["clustered_vs_interleaved_percent"]
        >= anchor["resolved_effect_floor_percent"]
    )
    supported = anchor["supporting_strata"] >= anchor["required_support"]
    status = (
        "I2A_ROW_ORDER_SIGNAL" if powered and supported else "I2A_NO_ROW_ORDER_SIGNAL"
    )
    if powered and not supported:
        status = "I2A_ORDER_SENSITIVE"
    elif (
        not powered
        and anchor["same_arm_p95_noise_percent"] > config.minimum_effect_percent
    ):
        status = "I2A_UNPOWERED"
    return {
        "i2a_status": status,
        "expected": expected,
        "observed": len(rows),
        "outputs_valid": True,
        "primary_anchor": anchor,
        "cells": cell_results,
    }


def run_i2a(
    config: I2AConfig,
    *,
    contexts: Sequence[int],
    speculative_steps: Sequence[int],
    seeds: Sequence[int],
    output_dir: Path,
) -> dict[str, Any]:
    config.validate()
    contexts, speculative_steps, seeds = validate_axes(
        contexts=contexts,
        speculative_steps=speculative_steps,
        seeds=seeds,
        require_primary_anchor=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("I2a requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 9 or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("I2a anchor requires H100")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        config, contexts=contexts, speculative_steps=speculative_steps, seeds=seeds
    )
    (output_dir / "i2a-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for context_length in contexts:
        for depth in speculative_steps:
            width = verify_width(int(depth))
            canonical_table = build_canonical_page_table(
                config, context_length=int(context_length), speculative_steps=int(depth)
            )
            for seed in seeds:
                k_cache, v_cache, table_device = _materialize_cache(
                    config,
                    table=canonical_table,
                    context_length=int(context_length),
                    speculative_steps=int(depth),
                    seed=int(seed),
                    device=device,
                )
                generator = torch.Generator(device=device).manual_seed(
                    int(seed) + 1_000_003
                )
                q = torch.randn(
                    (config.batch_size, width, config.num_query_heads, config.head_dim),
                    generator=generator,
                    dtype=_dtype(config.dtype),
                    device=device,
                )
                cache_seqlens = torch.full(
                    (config.batch_size,),
                    int(context_length) + width,
                    dtype=torch.int32,
                    device=device,
                )
                cu_seqlens_q = torch.arange(
                    0,
                    (config.batch_size + 1) * width,
                    width,
                    dtype=torch.int32,
                    device=device,
                )
                permutations = {
                    name: value.to(device)
                    for name, value in row_permutations(config, seed=int(seed)).items()
                }
                outputs = {}
                for arm, permutation in permutations.items():
                    output = _call_fa3(
                        q=q[permutation].reshape(
                            -1, config.num_query_heads, config.head_dim
                        ),
                        k_cache=k_cache,
                        v_cache=v_cache,
                        page_table=table_device[permutation],
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=width,
                        config=config,
                    ).view(
                        config.batch_size,
                        width,
                        config.num_query_heads,
                        config.head_dim,
                    )
                    outputs[arm] = output[torch.argsort(permutation)].detach().clone()
                reference = outputs[CLUSTERED]
                for arm in (INTERLEAVED, RANDOM):
                    difference = (reference.float() - outputs[arm].float()).abs()
                    checks.append(
                        {
                            "context_length": int(context_length),
                            "speculative_steps": int(depth),
                            "seed": int(seed),
                            "row_order": arm,
                            "bit_identical": bool(torch.equal(reference, outputs[arm])),
                            "maximum_absolute_difference": float(difference.max()),
                        }
                    )

                for order_index, arm_order in enumerate(BALANCED_ARM_ORDERS):
                    for position, arm in enumerate(arm_order):
                        permutation = permutations[arm]
                        call = functools.partial(
                            _call_fa3,
                            q=q[permutation].reshape(
                                -1, config.num_query_heads, config.head_dim
                            ),
                            k_cache=k_cache,
                            v_cache=v_cache,
                            page_table=table_device[permutation],
                            cache_seqlens=cache_seqlens,
                            cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_q=width,
                            config=config,
                        )
                        for _ in range(config.warmup):
                            call()
                        torch.cuda.synchronize()
                        for repetition in range(config.repetitions):
                            rows.append(
                                {
                                    "context_length": int(context_length),
                                    "speculative_steps": int(depth),
                                    "target_verify_width": width,
                                    "seed": int(seed),
                                    "order_index": order_index,
                                    "arm_order": ">".join(arm_order),
                                    "position": position,
                                    "row_order": arm,
                                    "repetition": repetition,
                                    "latency_ms": _time_one(call),
                                }
                            )
                del k_cache, v_cache, table_device, q, outputs
                torch.cuda.empty_cache()

    outputs_valid = all(check["bit_identical"] for check in checks)
    summary = analyze_rows(
        rows,
        config=config,
        contexts=contexts,
        speculative_steps=speculative_steps,
        seeds=seeds,
        outputs_valid=outputs_valid,
    )
    summary["config"] = asdict(config)
    summary["device"] = {
        "name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    summary["output_checks"] = checks
    _write_csv(output_dir / "i2a-raw-samples.csv", rows)
    (output_dir / "i2a-output-checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "i2a-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _config_from_args(args: argparse.Namespace) -> I2AConfig:
    return I2AConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        group_count=args.group_count,
        shared_prefix_ratio=args.shared_prefix_ratio,
        page_size=args.page_size,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        repetitions=args.repetitions,
        warmup=args.warmup,
        minimum_effect_percent=args.minimum_effect_percent,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-count", type=int, default=4)
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.9)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXT_LENGTHS)
    )
    parser.add_argument(
        "--speculative-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPECULATIVE_STEPS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--minimum-effect-percent", type=float, default=2.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    _add_arguments(plan_parser)
    plan_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    _add_arguments(run_parser)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    if args.command == "plan":
        plan = build_plan(
            config,
            contexts=args.contexts,
            speculative_steps=args.speculative_steps,
            seeds=args.seeds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        return 0
    summary = run_i2a(
        config,
        contexts=args.contexts,
        speculative_steps=args.speculative_steps,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["i2a_status"] not in {"I2A_INCOMPLETE", "I2A_INVALID"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
