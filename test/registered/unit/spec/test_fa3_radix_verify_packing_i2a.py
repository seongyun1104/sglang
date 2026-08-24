import importlib.util
import sys
from pathlib import Path

import torch


_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/benchmark/fa3_radix_verify_packing_i2a.py"
)
_SPEC = importlib.util.spec_from_file_location("_fa3_radix_verify_packing_i2a", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BALANCED_ARM_ORDERS = _MODULE.BALANCED_ARM_ORDERS
CLUSTERED = _MODULE.CLUSTERED
INTERLEAVED = _MODULE.INTERLEAVED
RANDOM = _MODULE.RANDOM
ROW_ORDERS = _MODULE.ROW_ORDERS
I2AConfig = _MODULE.I2AConfig
analyze_rows = _MODULE.analyze_rows
build_canonical_page_table = _MODULE.build_canonical_page_table
build_plan = _MODULE.build_plan
group_adjacency_ratio = _MODULE.group_adjacency_ratio
request_group_ids = _MODULE.request_group_ids
row_permutations = _MODULE.row_permutations
shared_pages = _MODULE.shared_pages
verify_width = _MODULE.verify_width
validate_axes = _MODULE.validate_axes


def test_target_verify_width_matches_topk1_eagle_chain():
    assert verify_width(1) == 2
    assert verify_width(2) == 3
    assert verify_width(4) == 5


def test_page_table_has_group_aliases_and_unique_suffixes():
    config = I2AConfig(
        batch_size=8,
        group_count=2,
        shared_prefix_ratio=0.75,
        repetitions=2,
    )
    table = build_canonical_page_table(config, context_length=32, speculative_steps=4)
    assert table.shape == (8, 32 + verify_width(4))
    shared = shared_pages(config, 32, 4)
    groups = request_group_ids(config)

    for left in range(config.batch_size):
        for right in range(left + 1, config.batch_size):
            if groups[left] == groups[right]:
                assert torch.equal(table[left, :shared], table[right, :shared])
            else:
                assert not torch.equal(table[left, :shared], table[right, :shared])
            assert not torch.isin(table[left, shared:], table[right, shared:]).any()


def test_row_orders_change_only_full_request_permutation():
    config = I2AConfig(batch_size=16, group_count=4)
    permutations = row_permutations(config, seed=17)
    expected = torch.arange(config.batch_size)
    for permutation in permutations.values():
        assert torch.equal(torch.sort(permutation).values, expected)

    assert group_adjacency_ratio(config, permutations[CLUSTERED]) == 0.8
    assert group_adjacency_ratio(config, permutations[INTERLEAVED]) == 0.0
    assert permutations[RANDOM].tolist() != permutations[CLUSTERED].tolist()


def test_all_six_arm_orders_balance_position():
    assert len(BALANCED_ARM_ORDERS) == 6
    for position in range(3):
        counts = {arm: 0 for arm in ROW_ORDERS}
        for order in BALANCED_ARM_ORDERS:
            counts[order[position]] += 1
        assert counts == {arm: 2 for arm in ROW_ORDERS}


def test_plan_records_same_page_table_and_real_verify_width():
    config = I2AConfig(
        batch_size=8,
        group_count=2,
        repetitions=2,
    )
    plan = build_plan(
        config,
        contexts=[64],
        speculative_steps=[1, 4],
        seeds=[17],
    )
    assert plan["cells"]["ctx64-k1"]["target_verify_width"] == 2
    assert plan["cells"]["ctx64-k4"]["target_verify_width"] == 5
    assert plan["expected_samples"] == 72
    assert plan["expected_output_checks"] == 4
    assert plan["semantic_contract"]["candidate_kv_present_in_cache"] is True
    assert plan["cells"]["ctx64-k4"]["target_kv_length"] == 64 + verify_width(4)
    stats = plan["cells"]["ctx64-k4"]["canonical_page_table"]
    assert stats["physical_page_references"] == 8 * (64 + verify_width(4))
    assert stats["page_reuse_ratio"] > 0


def test_axes_reject_duplicates_and_hardware_run_requires_anchor():
    try:
        validate_axes(
            contexts=[8192, 8192],
            speculative_steps=[1, 2, 4],
            seeds=[17, 29, 41],
            require_primary_anchor=False,
        )
    except ValueError as error:
        assert "contexts must be unique" in str(error)
    else:
        raise AssertionError("duplicate contexts must fail before GPU execution")

    try:
        validate_axes(
            contexts=[8192],
            speculative_steps=[1, 2],
            seeds=[17, 29, 41],
            require_primary_anchor=True,
        )
    except ValueError as error:
        assert "16K/K=4 anchor" in str(error)
    else:
        raise AssertionError("hardware run without the primary anchor must fail")


def _rows(config: I2AConfig, *, clustered: float, interleaved: float, random: float):
    latency = {CLUSTERED: clustered, INTERLEAVED: interleaved, RANDOM: random}
    rows = []
    for seed in (17, 29, 41):
        for order_index, arm_order in enumerate(BALANCED_ARM_ORDERS):
            for position, arm in enumerate(arm_order):
                for repetition in range(config.repetitions):
                    rows.append(
                        {
                            "context_length": 16384,
                            "speculative_steps": 4,
                            "target_verify_width": 5,
                            "seed": seed,
                            "order_index": order_index,
                            "arm_order": ">".join(arm_order),
                            "position": position,
                            "row_order": arm,
                            "repetition": repetition,
                            "latency_ms": latency[arm],
                        }
                    )
    return rows


def test_analyzer_requires_effect_and_stratum_support():
    config = I2AConfig(repetitions=2, minimum_effect_percent=2.0)
    positive = analyze_rows(
        _rows(config, clustered=8.0, interleaved=10.0, random=9.0),
        config=config,
        contexts=[16384],
        speculative_steps=[4],
        seeds=[17, 29, 41],
        outputs_valid=True,
    )
    null = analyze_rows(
        _rows(config, clustered=9.9, interleaved=10.0, random=10.0),
        config=config,
        contexts=[16384],
        speculative_steps=[4],
        seeds=[17, 29, 41],
        outputs_valid=True,
    )
    assert positive["i2a_status"] == "I2A_ROW_ORDER_SIGNAL"
    assert positive["primary_anchor"]["supporting_strata"] == 18
    assert positive["primary_anchor"]["same_arm_p95_noise_percent"] == 0.0
    assert positive["primary_anchor"]["resolved_effect_floor_percent"] == 2.0
    assert null["i2a_status"] == "I2A_NO_ROW_ORDER_SIGNAL"


def test_analyzer_marks_provider_noise_above_effect_as_unpowered():
    config = I2AConfig(repetitions=2, minimum_effect_percent=2.0)
    rows = _rows(config, clustered=9.7, interleaved=10.0, random=10.0)
    drift = (0.90, 0.94, 0.98, 1.02, 1.06, 1.10)
    for row in rows:
        row["latency_ms"] *= drift[int(row["order_index"])]

    result = analyze_rows(
        rows,
        config=config,
        contexts=[16384],
        speculative_steps=[4],
        seeds=[17, 29, 41],
        outputs_valid=True,
    )

    assert result["i2a_status"] == "I2A_UNPOWERED"
    anchor = result["primary_anchor"]
    assert anchor["same_arm_p95_noise_percent"] > 2.0
    assert (
        anchor["resolved_effect_floor_percent"] == anchor["same_arm_p95_noise_percent"]
    )


def test_analyzer_fails_closed_on_missing_or_invalid_output():
    config = I2AConfig(repetitions=2)
    rows = _rows(config, clustered=8.0, interleaved=10.0, random=9.0)
    incomplete = analyze_rows(
        rows[:-1],
        config=config,
        contexts=[16384],
        speculative_steps=[4],
        seeds=[17, 29, 41],
        outputs_valid=True,
    )
    invalid = analyze_rows(
        rows,
        config=config,
        contexts=[16384],
        speculative_steps=[4],
        seeds=[17, 29, 41],
        outputs_valid=False,
    )
    assert incomplete["i2a_status"] == "I2A_INCOMPLETE"
    assert invalid["i2a_status"] == "I2A_INVALID"


def test_analyzer_fails_closed_when_primary_anchor_is_absent():
    config = I2AConfig(repetitions=2)
    rows = _rows(config, clustered=8.0, interleaved=10.0, random=9.0)
    for row in rows:
        row["context_length"] = 8192
    result = analyze_rows(
        rows,
        config=config,
        contexts=[8192],
        speculative_steps=[4],
        seeds=[17, 29, 41],
        outputs_valid=True,
    )
    assert result["i2a_status"] == "I2A_INCOMPLETE"
    assert result["missing_primary_anchor"]["context_length"] == 16384
