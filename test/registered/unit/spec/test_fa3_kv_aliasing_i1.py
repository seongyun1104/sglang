import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/benchmark/fa3_kv_aliasing_i1.py"
)
_SPEC = importlib.util.spec_from_file_location("_fa3_kv_aliasing_i1", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_PREFLIGHT = (
    Path(__file__).resolve().parents[4] / "benchmark/fa3_kv_aliasing_i1/preflight_i1.sh"
)


def _profile(layout: str, digest: str = "same"):
    return {
        "layout": layout,
        "seed": 17,
        "warmup": 10,
        "output_sha256": digest,
        "output_shape": [16, 24, 256],
        "output_dtype": "torch.bfloat16",
        "config": {"model_id": "Qwen/Qwen3.8-27B"},
        "device": {"name": "NVIDIA H100 NVL"},
        "call": {"fa_version": 3, "profiled_calls": 1},
    }


def test_pair_validation_requires_identical_outputs_and_metadata(tmp_path):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    output = tmp_path / "pair.json"
    a_path.write_text(json.dumps(_profile(_MODULE.ALIASED)))
    b_path.write_text(json.dumps(_profile(_MODULE.CONTIGUOUS)))

    result = _MODULE.validate_profile_pair(a_path, b_path, output)

    assert result["valid"] is True
    assert result["output_sha256_equal"] is True
    assert json.loads(output.read_text()) == result


def test_pair_validation_fails_on_digest_or_role_mismatch(tmp_path):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    output = tmp_path / "pair.json"
    a_path.write_text(json.dumps(_profile(_MODULE.CONTIGUOUS)))
    b_path.write_text(json.dumps(_profile(_MODULE.CONTIGUOUS, digest="different")))

    result = _MODULE.validate_profile_pair(a_path, b_path, output)

    assert result["valid"] is False
    assert result["output_sha256_equal"] is False
    assert "layout_roles" in result["metadata_differences"]


def _latency_summary():
    return {
        "i0_status": "I0_ALIASING_CANDIDATE",
        "outputs_valid": True,
        "expected_samples": 3240,
        "observed_samples": 3240,
        "minimum_effect_percent": 2.0,
        "warm_alias_speedup_percent": 13.0,
        "required_stratum_support": 12,
        "alias_supporting_strata": 16,
        "config": dict(_MODULE.I1_ANCHOR_CONFIG),
    }


def test_latency_gate_requires_complete_supported_anchor(tmp_path):
    summary_path = tmp_path / "i0-summary.json"
    output = tmp_path / "gate.json"
    summary_path.write_text(json.dumps(_latency_summary()))

    result = _MODULE.validate_latency_gate(summary_path, output)

    assert result["valid"] is True
    assert result["config_mismatches"] == {}
    assert json.loads(output.read_text()) == result


def test_latency_gate_fails_closed_on_null_or_shape_mismatch(tmp_path):
    summary = _latency_summary()
    summary["i0_status"] = "I0_NO_ISOLATED_SIGNAL"
    summary["warm_alias_speedup_percent"] = 0.5
    summary["config"]["context_length"] = 8192
    summary_path = tmp_path / "i0-summary.json"
    output = tmp_path / "gate.json"
    summary_path.write_text(json.dumps(summary))

    result = _MODULE.validate_latency_gate(summary_path, output)

    assert result["valid"] is False
    assert result["effect_pass"] is False
    assert "context_length" in result["config_mismatches"]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source))
    path.chmod(0o755)


def _run_mocked_preflight(tmp_path: Path, *, compute_apps: str = ""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "nvidia-smi",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        args="$*"
        if [[ "$args" == *"--query-gpu=index,uuid,name"* ]]; then
          printf '%s\\n' '0, GPU-A, NVIDIA H100' '1, GPU-B, NVIDIA H100'
        elif [[ "$args" == *"-i 1 --query-gpu=uuid,name"* ]]; then
          printf '%s\\n' 'GPU-B, NVIDIA H100'
        elif [[ "$args" == *"--query-compute-apps"* ]]; then
          printf '%s' {compute_apps!r}
        fi
        """,
    )
    _write_executable(
        bin_dir / "ncu",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "$*" == *"--query-metrics"* ]]; then
          echo mock_metric
          exit 0
        fi
        export_path=""
        while (( $# )); do
          if [[ "$1" == "--export" ]]; then
            export_path="$2"
            break
          fi
          shift
        done
        [[ -n "$export_path" ]]
        printf '%s\\n' mock-report >"${export_path}.ncu-rep"
        """,
    )
    result_root = tmp_path / "result"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "I1_GPU_INDEX": "1",
            "I1_SM_CLOCK_MHZ": "1410",
            "RESULT_ROOT": str(result_root),
        }
    )
    completed = subprocess.run(
        ["bash", str(_PREFLIGHT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    return completed, result_root


def test_preflight_pins_one_gpu_on_idle_exclusive_multi_gpu_host(tmp_path):
    completed, result_root = _run_mocked_preflight(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert (result_root / "PRECHECK_PASS").exists()
    assert (result_root / "selected-gpu-index.txt").read_text().strip() == "1"
    assert (result_root / "selected-gpu-uuid.txt").read_text().strip() == "GPU-B"
    assert len((result_root / "all-visible-gpus.txt").read_text().splitlines()) == 2


def test_preflight_rejects_non_idle_multi_gpu_host(tmp_path):
    completed, result_root = _run_mocked_preflight(
        tmp_path,
        compute_apps="GPU-A, 42, python, 100 MiB\\n",
    )

    assert completed.returncode == 12
    assert (result_root / "PRECHECK_HOST_NOT_IDLE").exists()
    assert (result_root / "PRECHECK_FAILED").exists()
    assert not (result_root / "PRECHECK_PASS").exists()
