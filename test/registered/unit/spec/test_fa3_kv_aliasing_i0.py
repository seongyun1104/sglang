import pytest
import torch

from sglang.benchmark.fa3_kv_aliasing_i0 import (
    ALIASED,
    BALANCED_ORDERS,
    CACHE_STATES,
    CONTIGUOUS,
    LAYOUTS,
    SCATTERED,
    I0Config,
    analyze_rows,
    build_page_tables,
    page_table_statistics,
)


def test_page_tables_isolate_aliasing_and_scattering():
    config = I0Config(
        batch_size=4,
        context_length=32,
        shared_prefix_ratio=0.75,
        page_size=1,
        repetitions=2,
    )
    tables = build_page_tables(config, scatter_seed=17)

    shared_prefix = tables[ALIASED][:, : config.shared_pages]
    assert torch.equal(shared_prefix, shared_prefix[0].expand_as(shared_prefix))
    assert torch.unique(tables[CONTIGUOUS]).numel() == tables[CONTIGUOUS].numel()
    assert torch.unique(tables[SCATTERED]).numel() == tables[SCATTERED].numel()
    assert torch.all(torch.diff(tables[CONTIGUOUS], dim=1) == 1)
    assert not torch.any(torch.abs(torch.diff(tables[SCATTERED], dim=1)) == 1)

    aliased_stats = page_table_statistics(tables[ALIASED])
    contiguous_stats = page_table_statistics(tables[CONTIGUOUS])
    scattered_stats = page_table_statistics(tables[SCATTERED])
    assert aliased_stats["page_reuse_ratio"] > 0
    assert contiguous_stats["page_reuse_ratio"] == 0
    assert scattered_stats["page_reuse_ratio"] == 0
    assert contiguous_stats["adjacent_step_ratio"] == 1
    assert scattered_stats["adjacent_step_ratio"] == 0


def test_default_shape_tracks_qwen38_full_attention():
    config = I0Config()
    assert config.model_id == "Qwen/Qwen3.8-27B"
    assert (config.num_query_heads, config.num_kv_heads, config.head_dim) == (
        24,
        4,
        256,
    )


def test_all_six_orders_balance_every_layout_and_position():
    assert len(BALANCED_ORDERS) == 6
    assert len(set(BALANCED_ORDERS)) == 6
    for position in range(3):
        counts = {layout: 0 for layout in LAYOUTS}
        for order in BALANCED_ORDERS:
            counts[order[position]] += 1
        assert counts == {layout: 2 for layout in LAYOUTS}


def _rows(
    config: I0Config, *, alias_ms: float, contiguous_ms: float, scattered_ms: float
):
    latencies = {
        ALIASED: alias_ms,
        CONTIGUOUS: contiguous_ms,
        SCATTERED: scattered_ms,
    }
    rows = []
    for cache_state in CACHE_STATES:
        for order_index, order in enumerate(BALANCED_ORDERS):
            for position, layout in enumerate(order):
                for repetition in range(config.repetitions):
                    rows.append(
                        {
                            "seed": 17,
                            "cache_state": cache_state,
                            "order_index": order_index,
                            "order": ">".join(order),
                            "position": position,
                            "layout": layout,
                            "repetition": repetition,
                            "latency_ms": latencies[layout],
                        }
                    )
    return rows


@pytest.mark.parametrize(
    "latencies, expected_status",
    [
        ((8.0, 10.0, 10.0), "I0_ALIASING_CANDIDATE"),
        ((10.0, 10.0, 12.0), "I0_LOCALITY_CANDIDATE"),
        ((8.0, 10.0, 12.0), "I0_MIXED_CANDIDATE"),
        ((10.0, 10.1, 10.2), "I0_NO_ISOLATED_SIGNAL"),
        ((12.0, 10.0, 10.0), "I0_UNEXPECTED_DIRECTION"),
        ((10.0, 12.0, 10.0), "I0_UNEXPECTED_DIRECTION"),
    ],
)
def test_analyzer_classifies_preregistered_patterns(latencies, expected_status):
    config = I0Config(
        batch_size=2,
        context_length=16,
        shared_prefix_ratio=0.5,
        repetitions=2,
    )
    summary = analyze_rows(
        _rows(
            config,
            alias_ms=latencies[0],
            contiguous_ms=latencies[1],
            scattered_ms=latencies[2],
        ),
        config=config,
        seeds=[17],
        outputs_valid=True,
    )
    assert summary["i0_status"] == expected_status
    assert summary["observed_samples"] == summary["expected_samples"]


def test_analyzer_fails_closed_on_missing_or_invalid_controls():
    config = I0Config(
        batch_size=2,
        context_length=16,
        shared_prefix_ratio=0.5,
        repetitions=2,
    )
    rows = _rows(config, alias_ms=8.0, contiguous_ms=10.0, scattered_ms=10.0)
    incomplete = analyze_rows(rows[:-1], config=config, seeds=[17], outputs_valid=True)
    invalid = analyze_rows(rows, config=config, seeds=[17], outputs_valid=False)
    duplicate_replacing_last = [*rows[:-1], rows[0].copy()]
    duplicate = analyze_rows(
        duplicate_replacing_last,
        config=config,
        seeds=[17],
        outputs_valid=True,
    )
    invalid_timing_rows = [row.copy() for row in rows]
    invalid_timing_rows[0]["latency_ms"] = float("nan")
    invalid_timing = analyze_rows(
        invalid_timing_rows,
        config=config,
        seeds=[17],
        outputs_valid=True,
    )
    assert incomplete["i0_status"] == "I0_INCOMPLETE"
    assert invalid["i0_status"] == "I0_INVALID"
    assert duplicate["i0_status"] == "I0_INCOMPLETE"
    assert duplicate["missing_cells"] == 1
    assert invalid_timing["i0_status"] == "I0_INVALID"


def test_config_rejects_unaligned_or_non_aliasing_inputs():
    with pytest.raises(ValueError, match="page-size aligned"):
        I0Config(context_length=17, page_size=4).validate()
    with pytest.raises(ValueError, match="at least 2"):
        I0Config(batch_size=1).validate()
    with pytest.raises(ValueError, match="between zero and one"):
        I0Config(shared_prefix_ratio=1.0).validate()
