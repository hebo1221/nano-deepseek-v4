from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    AdaptiveMemoryTraceCollector,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    MemoryTraceConfig,
    ReplayFeatureRow,
    ReplayPolicyConfig,
    analyze_budget_signals,
    build_replay_queries,
    calibrate_index_reuse,
    calibrate_layer_budgets,
    exhaustive_sufficient_subset,
    extract_replay_features,
    load_memory_trace,
    run_replay,
)
from nano_deepseek_v4.memory_replay import main as replay_main


def _trace_model(
    *,
    trace_id: str = "replay",
    layer_types: list[str] | None = None,
    length: int = 20,
):
    torch.manual_seed(0)
    config = DeepSeekV4Config(layer_types=layer_types)
    model = DeepSeekV4ForCausalLM(config).eval()
    collector = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id=trace_id))
    ids = torch.arange(length).remainder(config.vocab_size).unsqueeze(0)
    model(ids, use_cache=True, memory_trace=collector)
    return collector


def _end_positions(block_ids: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(block_id.rsplit(":e", 1)[1]) for block_id in block_ids)


def test_trace_v2_exposes_ranked_candidates_and_block_bytes():
    collector = _trace_model(length=20)
    queries = build_replay_queries(collector.result())

    assert queries
    assert all(query.block_bytes > 0 for query in queries)
    populated = [query for query in queries if query.ranked_blocks]
    assert populated
    for query in populated:
        scores = [block.score for block in query.ranked_blocks]
        assert scores == sorted(scores, reverse=True)
        assert set(query.native_block_ids).issubset(block.block_id for block in query.ranked_blocks)


def test_replay_baselines_are_deterministic_and_budget_accounted():
    trace = _trace_model(length=24).result()
    native = run_replay(trace, ReplayPolicyConfig(name="native"))
    topk = run_replay(trace, ReplayPolicyConfig(name="fixed_top_k", budget=2))
    recency = run_replay(trace, ReplayPolicyConfig(name="recency", budget=2))
    top_p = run_replay(trace, ReplayPolicyConfig(name="fixed_top_p", top_p=0.75))
    per_layer = run_replay(
        trace,
        ReplayPolicyConfig(name="per_layer", layer_budgets=((2, 1),)),
    )
    random_a = run_replay(trace, ReplayPolicyConfig(name="random", budget=2, seed=17))
    random_b = run_replay(trace, ReplayPolicyConfig(name="random", budget=2, seed=17))

    assert native.mean_native_recall == 1.0
    assert native.exact_native_match_rate == 1.0
    assert all(decision.selected_blocks <= 2 for decision in topk.decisions)
    assert all(decision.selected_blocks <= 2 for decision in recency.decisions)
    assert all(decision.selected_blocks <= 1 for decision in per_layer.decisions)
    assert top_p.total_selected_blocks >= 0
    assert random_a == random_b
    assert topk.total_selected_bytes == sum(decision.selected_bytes for decision in topk.decisions)


def test_full_and_chunked_execution_produce_the_same_replay_decisions():
    torch.manual_seed(0)
    config = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(config).eval()
    ids = torch.arange(20).unsqueeze(0)

    full_trace = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="full-replay"))
    model(ids, use_cache=True, memory_trace=full_trace)

    chunked_trace = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="chunked-replay"))
    first = model(ids[:, :7], use_cache=True, memory_trace=chunked_trace)
    assert first.past_key_values is not None
    model(ids[:, 7:], past_key_values=first.past_key_values, use_cache=True)

    policy = ReplayPolicyConfig(name="fixed_top_k", budget=2)
    full = run_replay(full_trace.result(), policy)
    chunked = run_replay(chunked_trace.result(), policy)
    full_decisions = tuple(
        (decision.layer_index, decision.query_position, _end_positions(decision.selected_block_ids))
        for decision in full.decisions
    )
    chunked_decisions = tuple(
        (decision.layer_index, decision.query_position, _end_positions(decision.selected_block_ids))
        for decision in chunked.decisions
    )
    assert chunked_decisions == full_decisions
    assert chunked.total_selected_bytes == full.total_selected_bytes


def test_index_reuse_translates_source_positions_to_target_layer():
    trace = _trace_model(
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        length=20,
    ).result()
    result = run_replay(
        trace,
        ReplayPolicyConfig(name="index_reuse", budget=2, reuse_layers=((2, 1),)),
    )
    decisions = {
        (decision.layer_index, decision.batch_index, decision.query_position): decision
        for decision in result.decisions
    }
    compared = 0
    for (layer, batch, position), target in decisions.items():
        if layer != 2:
            continue
        source = decisions[(1, batch, position)]
        if not source.selected_block_ids:
            continue
        assert _end_positions(target.selected_block_ids) == _end_positions(
            source.selected_block_ids
        )
        compared += 1
    assert compared > 0


def test_static_layer_and_index_reuse_calibration_use_only_supplied_traces():
    trace = _trace_model(
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        length=20,
    ).result()
    budgets = calibrate_layer_budgets((trace,), quantile=1.0)
    assert {layer for layer, _ in budgets} == {1, 2}
    assert all(budget >= 0 for _, budget in budgets)

    reuse = calibrate_index_reuse((trace,), minimum_mean_jaccard=0.0)
    assert reuse.reuse_layers == ((2, 1),)
    assert reuse.mean_position_jaccard[0][0] == 2
    assert 0.0 <= reuse.mean_position_jaccard[0][1] <= 1.0


def test_v1_trace_remains_loadable_for_native_replay(tmp_path: Path):
    collector = _trace_model(length=12)
    trace_dir = tmp_path / "v1"
    collector.write(trace_dir)

    event_path = trace_dir / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    for event in events:
        event["schema_version"] = 1
        if event["event_type"] == "csa_selection":
            event.pop("block_bytes")
            for selection in event["selections"]:
                selection.pop("ranked_blocks")
    payload = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events
    )
    event_path.write_text(payload, encoding="utf-8")
    manifest_path = trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest["events_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_memory_trace(trace_dir)
    assert loaded.manifest.schema_version == 1
    native = run_replay(loaded, ReplayPolicyConfig(name="native"))
    assert native.mean_native_recall == 1.0
    with pytest.raises(ValueError, match="schema v2"):
        run_replay(loaded, ReplayPolicyConfig(name="fixed_top_k", budget=1))


def test_exhaustive_oracle_finds_minimum_quality_sufficient_subset():
    candidates = ("a", "b", "c")
    required = {"a", "c"}

    def quality(subset: tuple[str, ...]) -> float:
        return len(set(subset) & required) / len(required)

    result = exhaustive_sufficient_subset(candidates, quality, threshold=1.0)
    assert result.selected_block_ids == ("a", "c")
    assert result.sufficient_budget == 2
    assert result.quality == 1.0
    assert result.evaluated_subsets == 7

    with pytest.raises(ValueError, match="maximum"):
        exhaustive_sufficient_subset(tuple(map(str, range(21))), quality, threshold=1.0)


def test_feature_extraction_and_signal_analysis_are_deterministic():
    trace = _trace_model(length=32).result()
    first = extract_replay_features(trace)
    second = extract_replay_features(trace)
    assert first == second
    assert first
    assert all(math.isfinite(row.score_entropy) for row in first)

    synthetic = tuple(
        replace(
            first[index % len(first)],
            trace_id=f"synthetic-{index}",
            request_id=f"request-{index}",
            context_blocks=4,
            layer_index=2,
            score_entropy=float(index),
            top1_probability=float(index) / 10.0,
        )
        for index in range(8)
    )
    labels = tuple(index for index in range(8))
    report = analyze_budget_signals(synthetic, labels, minimum_relative_improvement=0.0)
    assert report.examples == 8
    assert math.isfinite(report.baseline_mae)
    assert math.isfinite(report.full_feature_mae)


def test_replay_cli_writes_machine_readable_result(tmp_path: Path):
    collector = _trace_model(length=16)
    trace_dir = tmp_path / "trace"
    collector.write(trace_dir)
    output_path = tmp_path / "result.json"

    assert (
        replay_main(
            [
                str(trace_dir),
                "--policy",
                "fixed_top_k",
                "--budget",
                "2",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    result = json.loads(output_path.read_text())
    assert result["policy"]["name"] == "fixed_top_k"
    assert result["query_count"] > 0


def test_signal_analysis_requires_aligned_nontrivial_data():
    row = ReplayFeatureRow(
        trace_id="x",
        request_id="y",
        layer_index=0,
        batch_index=0,
        query_position=0,
        context_blocks=0,
        native_budget=0,
        score_entropy=0.0,
        top1_probability=0.0,
        top50_cardinality=0,
        top90_cardinality=0,
        top95_cardinality=0,
        boundary_margin=0.0,
        score_mean=0.0,
        score_std=0.0,
        temporal_jaccard=1.0,
        cross_layer_jaccard=1.0,
    )
    with pytest.raises(ValueError, match="at least four"):
        analyze_budget_signals((row,), (0,))
