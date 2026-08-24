"""Profiler targets for I1 FA3 physical-KV-aliasing mechanism localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch


ALIASED = "shared_aliased"
CONTIGUOUS = "duplicated_contiguous"
PROFILE_LAYOUTS = (ALIASED, CONTIGUOUS)
I1_LATENCY_PASS_STATUSES = ("I0_ALIASING_CANDIDATE", "I0_MIXED_CANDIDATE")
I1_ANCHOR_CONFIG = {
    "model_id": "Qwen/Qwen3.8-27B",
    "batch_size": 16,
    "context_length": 16384,
    "query_length": 1,
    "shared_prefix_ratio": 0.9,
    "page_size": 1,
    "num_query_heads": 24,
    "num_kv_heads": 4,
    "head_dim": 256,
    "dtype": "bfloat16",
    "causal": True,
    "num_splits": 0,
}


def _start_profiler() -> None:
    result = torch.cuda.cudart().cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with code {result}")


def _stop_profiler() -> None:
    result = torch.cuda.cudart().cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with code {result}")


def run_counter_preflight(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("counter preflight requires CUDA")
    x = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
    y = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
    for _ in range(3):
        torch.add(x, y)
    torch.cuda.synchronize()
    _start_profiler()
    z = torch.add(x, y)
    _stop_profiler()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kernel": "torch.add",
        "elements": x.numel(),
        "checksum": float(z[:1024].sum()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _output_digest(output: torch.Tensor) -> str:
    raw = output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def profile_layout(
    *,
    layout: str,
    output: Path,
    seed: int,
    warmup: int,
) -> dict[str, Any]:
    from sglang.benchmark.fa3_kv_aliasing_i0 import (
        I0Config,
        _call_fa3,
        _make_logical_kv,
        _materialize_layout,
        _torch_dtype,
        build_page_tables,
    )

    if layout not in PROFILE_LAYOUTS:
        raise ValueError(f"layout must be one of {PROFILE_LAYOUTS}")
    if not torch.cuda.is_available():
        raise RuntimeError("I1 requires CUDA")
    config = I0Config()
    config.validate()
    major, _ = torch.cuda.get_device_capability()
    if major != 9 or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("I1 anchor requires H100")
    device = torch.device("cuda")
    tables_cpu = build_page_tables(config, scatter_seed=int(seed))
    table = tables_cpu[layout].to(device)
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
        dtype=_torch_dtype(config.dtype),
        device=device,
    ).view(
        config.batch_size * config.query_length, config.num_query_heads, config.head_dim
    )
    cu_seqlens_q = torch.arange(
        0,
        (config.batch_size + 1) * config.query_length,
        config.query_length,
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.full(
        (config.batch_size,), config.context_length, dtype=torch.int32, device=device
    )
    cache_shape = (
        config.total_backing_pages,
        config.page_size,
        config.num_kv_heads,
        config.head_dim,
    )
    k_cache = torch.empty(cache_shape, dtype=_torch_dtype(config.dtype), device=device)
    v_cache = torch.empty_like(k_cache)
    _materialize_layout(
        config=config,
        layout=layout,
        page_table=table,
        logical=logical,
        k_cache=k_cache,
        v_cache=v_cache,
    )

    def call() -> torch.Tensor:
        return _call_fa3(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            config=config,
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    _start_profiler()
    result_tensor = call()
    _stop_profiler()
    torch.cuda.synchronize()
    result = {
        "layout": layout,
        "seed": int(seed),
        "warmup": int(warmup),
        "output_sha256": _output_digest(result_tensor),
        "output_shape": list(result_tensor.shape),
        "output_dtype": str(result_tensor.dtype),
        "config": asdict(config),
        "device": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "call": {
            "callable": "sgl_kernel.flash_attn.flash_attn_with_kvcache",
            "fa_version": 3,
            "causal": True,
            "softmax_scale": 1.0 / math.sqrt(config.head_dim),
            "profiled_calls": 1,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def validate_profile_pair(a_path: Path, b_path: Path, output: Path) -> dict[str, Any]:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    differences = []
    for key in (
        "seed",
        "warmup",
        "output_shape",
        "output_dtype",
        "config",
        "device",
        "call",
    ):
        if a[key] != b[key]:
            differences.append(key)
    if a["layout"] != ALIASED or b["layout"] != CONTIGUOUS:
        differences.append("layout_roles")
    output_equal = a["output_sha256"] == b["output_sha256"]
    result = {
        "valid": not differences and output_equal,
        "metadata_differences": differences,
        "output_sha256_equal": output_equal,
        "a_output_sha256": a["output_sha256"],
        "b_output_sha256": b["output_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def validate_latency_gate(summary_path: Path, output: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    config = summary.get("config", {})
    config_mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in I1_ANCHOR_CONFIG.items()
        if config.get(key) != expected
    }
    expected_samples = summary.get("expected_samples")
    observed_samples = summary.get("observed_samples")
    complete = (
        isinstance(expected_samples, int)
        and expected_samples > 0
        and observed_samples == expected_samples
    )
    status = summary.get("i0_status")
    minimum_effect = summary.get("minimum_effect_percent")
    measured_effect = summary.get("warm_alias_speedup_percent")
    effect_pass = (
        isinstance(minimum_effect, (int, float))
        and isinstance(measured_effect, (int, float))
        and math.isfinite(float(minimum_effect))
        and math.isfinite(float(measured_effect))
        and float(measured_effect) >= float(minimum_effect)
    )
    required_support = summary.get("required_stratum_support")
    observed_support = summary.get("alias_supporting_strata")
    support_pass = (
        isinstance(required_support, int)
        and required_support > 0
        and isinstance(observed_support, int)
        and observed_support >= required_support
    )
    valid = bool(
        status in I1_LATENCY_PASS_STATUSES
        and summary.get("outputs_valid") is True
        and complete
        and effect_pass
        and support_pass
        and not config_mismatches
    )
    result = {
        "valid": valid,
        "i0_status": status,
        "accepted_i0_statuses": list(I1_LATENCY_PASS_STATUSES),
        "outputs_valid": summary.get("outputs_valid"),
        "complete": complete,
        "expected_samples": expected_samples,
        "observed_samples": observed_samples,
        "effect_pass": effect_pass,
        "minimum_effect_percent": minimum_effect,
        "warm_alias_speedup_percent": measured_effect,
        "support_pass": support_pass,
        "required_stratum_support": required_support,
        "alias_supporting_strata": observed_support,
        "config_mismatches": config_mismatches,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("counter-preflight")
    preflight.add_argument("--output", type=Path, required=True)
    profile = subparsers.add_parser("profile-layout")
    profile.add_argument("--layout", choices=PROFILE_LAYOUTS, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--seed", type=int, default=17)
    profile.add_argument("--warmup", type=int, default=10)
    validate = subparsers.add_parser("validate-pair")
    validate.add_argument("--a", type=Path, required=True)
    validate.add_argument("--b", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    latency_gate = subparsers.add_parser("validate-latency-gate")
    latency_gate.add_argument("--summary", type=Path, required=True)
    latency_gate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "counter-preflight":
        print(json.dumps(run_counter_preflight(args.output), indent=2, sort_keys=True))
        return 0
    if args.command == "profile-layout":
        print(
            json.dumps(
                profile_layout(
                    layout=args.layout,
                    output=args.output,
                    seed=args.seed,
                    warmup=args.warmup,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-latency-gate":
        result = validate_latency_gate(args.summary, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 2
    result = validate_profile_pair(args.a, args.b, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
