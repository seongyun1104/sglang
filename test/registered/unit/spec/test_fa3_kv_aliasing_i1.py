import importlib.util
import json
import sys
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
