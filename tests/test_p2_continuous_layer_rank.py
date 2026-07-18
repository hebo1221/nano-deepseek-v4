from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from nano_deepseek_v4 import RankedBlock, ReplayQuery, TrainingFreeControllerConfig
from nano_deepseek_v4.memory_controller import compute_controller_layer_signal

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_continuous_layer_rank as rank  # noqa: E402


def _query(scores: tuple[float, ...], *, layer: int = 2) -> ReplayQuery:
    blocks = tuple(
        RankedBlock(block_id=f"l{layer}:b0:e{4 * (index + 1)}", score=score)
        for index, score in enumerate(scores)
    )
    return ReplayQuery(
        trace_id="continuous-rank-test",
        request_id="calibration",
        layer_index=layer,
        batch_index=0,
        query_position=64,
        phase="prefill",
        logical_block_count=len(blocks),
        block_bytes=64,
        native_block_ids=tuple(block.block_id for block in blocks[:2]),
        ranked_blocks=blocks,
    )


def _config() -> TrainingFreeControllerConfig:
    return TrainingFreeControllerConfig(
        global_block_budget=12,
        dense_fallback_block_budget=24,
        top_p=0.8,
        min_blocks_per_layer=2,
        max_extra_blocks_per_layer=6,
        entropy_weight=0.35,
        margin_weight=0.2,
        temporal_weight=0.25,
        cross_layer_weight=0.2,
    )


def _observations(
    layer_values: dict[int, tuple[float, ...]], *, slices: tuple[str, ...] = ("f:80", "g:128")
) -> list[rank.DemandObservation]:
    rows: list[rank.DemandObservation] = []
    for slice_id in slices:
        for trace_index in range(4):
            for layer, values in layer_values.items():
                rows.append(
                    rank.DemandObservation(
                        slice_id=slice_id,
                        trace_batch_id=f"{slice_id}:trace-{trace_index}",
                        layer_index=layer,
                        value=values[trace_index],
                    )
                )
    return rows


def test_preceil_demand_reuses_frozen_signal_and_only_relaxes_final_ceiling() -> None:
    query = _query((4.0, 2.0, 1.0, 0.0))
    config = _config()
    signal = compute_controller_layer_signal(query, config, (), ())

    observed = rank.compute_preceil_demand(query, config)
    expected = min(
        signal.candidate_blocks,
        max(config.min_blocks_per_layer, signal.top_p_cardinality)
        + signal.uncertainty * config.max_extra_blocks_per_layer,
    )

    assert observed == pytest.approx(expected)
    assert observed <= signal.requested_blocks
    assert observed > signal.requested_blocks - 1.0


def test_preceil_demand_preserves_empty_candidate_special_case() -> None:
    assert rank.compute_preceil_demand(_query(()), _config()) == 0.0


def test_empirical_es95_uses_exact_fractional_boundary_mass() -> None:
    assert rank.empirical_es95(range(1, 11)) == pytest.approx(10.0)
    assert rank.empirical_es95(range(1, 26)) == pytest.approx((25.0 + 0.25 * 24.0) / 1.25)
    assert rank.empirical_es95(range(1, 41)) == pytest.approx((40.0 + 39.0) / 2.0)

    with pytest.raises(ValueError, match="at least one"):
        rank.empirical_es95([])


def test_slice_aggregation_weights_slices_not_query_counts() -> None:
    observations = [
        *(rank.DemandObservation("large", f"trace-{index}", 2, 0.0) for index in range(20)),
        rank.DemandObservation("small", "trace-0", 2, 10.0),
    ]

    assert rank.slice_scores(observations, layer_index=2) == {"large": 0.0, "small": 10.0}
    assert rank.equal_weight_slice_score(observations, layer_index=2) == pytest.approx(5.0)


def test_trace_batch_split_is_canonical_within_slice_and_shared_across_layers() -> None:
    observations = [
        rank.DemandObservation("slice", trace, layer, 1.0)
        for trace in ("trace-c", "trace-a", "trace-b")
        for layer in (2, 4)
    ]

    assignments = rank.deterministic_trace_batch_split(observations)

    assert assignments == {
        ("slice", "trace-a"): "A",
        ("slice", "trace-b"): "B",
        ("slice", "trace-c"): "A",
    }


def test_stratified_paired_bootstrap_is_deterministic_and_json_safe() -> None:
    observations = _observations({2: (4.0, 5.0, 6.0, 7.0), 4: (3.0, 4.0, 5.0, 6.0)})

    first = rank.stratified_paired_bootstrap(observations, top_layer=2, bottom_layer=4, seed=17)
    repeated = rank.stratified_paired_bootstrap(observations, top_layer=2, bottom_layer=4, seed=17)

    assert first == repeated
    assert first.resamples == 10_000
    assert first.point_difference == pytest.approx(1.0)
    assert first.lower_bound == pytest.approx(1.0)
    assert first.upper_bound == pytest.approx(1.0)
    json.dumps(first.to_dict(), allow_nan=False)


def test_boundary_requires_point_bootstrap_and_both_halves_to_agree() -> None:
    identified_rows = _observations({2: (4.0, 5.0, 6.0, 7.0), 4: (3.0, 4.0, 5.0, 6.0)})
    decision = rank.pairwise_boundary_decision(
        identified_rows, top_layer=2, bottom_layer=4, seed=19
    )

    assert decision.identified is True
    assert all(decision.to_dict()["checks"].values())

    disagreeing_rows = _observations({2: (5.0, 0.0, 5.0, 0.0), 4: (4.0, 1.0, 4.0, 1.0)})
    unresolved = rank.pairwise_boundary_decision(
        disagreeing_rows, top_layer=2, bottom_layer=4, seed=19
    )

    assert unresolved.split_a_difference > 0.0
    assert unresolved.split_b_difference < 0.0
    assert unresolved.identified is False

    one_trace = [
        rank.DemandObservation("slice", "trace", layer, value)
        for layer, value in ((2, 2.0), (4, 1.0))
    ]
    with pytest.raises(ValueError, match="at least two trace batches"):
        rank.pairwise_boundary_decision(
            one_trace, top_layer=2, bottom_layer=4, seed=19, resamples=10
        )


def test_top_bottom_boundaries_do_not_break_point_ties_by_layer_id() -> None:
    observations = _observations(
        {
            2: (5.0, 5.0, 5.0, 5.0),
            4: (5.0, 5.0, 5.0, 5.0),
            6: (1.0, 1.0, 1.0, 1.0),
        }
    )

    result = rank.top_bottom_boundary_decisions(observations, seed=23)

    assert result["top"]["point_boundary_candidates"] == [2, 4]
    assert result["top"]["pairwise_decisions"] == []
    assert result["top"]["identified"] is False
    assert result["bottom"]["point_boundary_candidates"] == [6]
    assert result["bottom"]["identified"] is True
    assert "no layer-ID or digest tie-break" in result["tie_policy"]
    json.dumps(result, allow_nan=False)


def test_json_metric_rejects_nonfinite_values_and_preserves_exact_hex() -> None:
    metric = rank.json_safe_float(0.1)

    assert metric == {"value": 0.1, "hex": (0.1).hex()}
    json.dumps(metric, allow_nan=False)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            rank.json_safe_float(value)


def test_exact_path_slice_bootstrap_does_not_treat_repeat_orders_as_samples() -> None:
    scores = {
        2: {f"slice-{index}": float(index + 2) for index in range(45)},
        4: {f"slice-{index}": float(index + 1) for index in range(45)},
    }

    inference = rank.paired_slice_bootstrap(
        scores,
        top_layer=2,
        bottom_layer=4,
        seed=31,
        resamples=1_000,
    )

    assert inference.point_difference == pytest.approx(1.0)
    assert inference.lower_bound == pytest.approx(1.0)
    assert inference.upper_bound == pytest.approx(1.0)
    assert inference.slices == 45
    assert inference.to_dict()["resampling_unit"] == "paired frozen family-context slice"


def test_exact_path_top_bottom_gate_preserves_ties() -> None:
    scores = {
        2: {f"slice-{index}": 5.0 for index in range(45)},
        4: {f"slice-{index}": 5.0 for index in range(45)},
        6: {f"slice-{index}": 1.0 for index in range(45)},
    }

    result = rank.top_bottom_slice_boundary_decisions(scores, seed=37, resamples=1_000)

    assert result["top"]["point_boundary_candidates"] == [2, 4]
    assert result["top"]["identified"] is False
    assert result["bottom"]["identified"] is True
    assert result["repeat_orders_used_as_resampling_units"] is False
