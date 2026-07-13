from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    AdaptiveMemoryTraceCollector,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    MemoryTraceConfig,
    ReplayPolicyConfig,
    analyze_budget_signals,
    build_replay_queries,
    calibrate_index_reuse,
    calibrate_layer_budgets,
    exhaustive_sufficient_subset,
    extract_replay_features,
    run_replay,
)

SEEDS = (0, 1, 2)
CONTEXT_LENGTHS = (16, 24, 32)
LAYER_TYPES = [
    "sliding_attention",
    "compressed_sparse_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
]


def _make_trace(seed: int, context_length: int):
    torch.manual_seed(seed)
    config = DeepSeekV4Config(layer_types=LAYER_TYPES)
    model = DeepSeekV4ForCausalLM(config).eval()
    token_ids = (
        torch.arange(context_length).mul(17).add(seed * 31).remainder(config.vocab_size)
    ).unsqueeze(0)
    collector = AdaptiveMemoryTraceCollector(
        MemoryTraceConfig(
            trace_id=f"tier-t-seed-{seed}-length-{context_length}",
            request_id="synthetic",
        )
    )
    with torch.no_grad():
        model(token_ids, use_cache=True, memory_trace=collector)
    return collector.result()


def _aggregate(results):
    queries = sum(result.query_count for result in results)
    return {
        "traces": len(results),
        "queries": queries,
        "selected_blocks": sum(result.total_selected_blocks for result in results),
        "selected_bytes": sum(result.total_selected_bytes for result in results),
        "mean_selected_blocks": (sum(result.total_selected_blocks for result in results) / queries),
        "mean_native_recall": (
            sum(decision.native_recall for result in results for decision in result.decisions)
            / queries
        ),
        "mean_native_jaccard": (
            sum(decision.native_jaccard for result in results for decision in result.decisions)
            / queries
        ),
    }


def run() -> dict:
    traces = [
        _make_trace(seed, context_length) for seed in SEEDS for context_length in CONTEXT_LENGTHS
    ]
    calibration = tuple(trace for trace in traces if trace.manifest.trace_id.endswith("length-16"))
    evaluation = tuple(trace for trace in traces if trace not in calibration)
    layer_budgets = calibrate_layer_budgets(calibration, quantile=0.95)
    reuse = calibrate_index_reuse(calibration, minimum_mean_jaccard=0.0)
    policies = {
        "native": ReplayPolicyConfig(name="native"),
        "recency-2": ReplayPolicyConfig(name="recency", budget=2),
        "random-2": ReplayPolicyConfig(name="random", budget=2, seed=20260714),
        "fixed-top-k-2": ReplayPolicyConfig(name="fixed_top_k", budget=2),
        "fixed-top-p-0.9": ReplayPolicyConfig(name="fixed_top_p", top_p=0.9),
        "per-layer-calibrated-p95": ReplayPolicyConfig(
            name="per_layer",
            layer_budgets=layer_budgets,
        ),
        "index-reuse-calibrated": ReplayPolicyConfig(
            name="index_reuse",
            budget=2,
            reuse_layers=reuse.reuse_layers,
        ),
    }
    policy_results = {
        name: _aggregate([run_replay(trace, policy) for trace in evaluation])
        for name, policy in policies.items()
    }

    oracle_runs = []
    feature_rows = []
    labels = []
    for trace in evaluation:
        queries = build_replay_queries(trace)
        populated = [query for query in queries if query.ranked_blocks]
        final_query = populated[-1]
        candidates = tuple(block.block_id for block in final_query.ranked_blocks)
        required = set(final_query.native_block_ids)

        def quality(subset: tuple[str, ...], required_blocks: set[str] = required) -> float:
            return (
                len(set(subset) & required_blocks) / len(required_blocks)
                if required_blocks
                else 1.0
            )

        oracle_runs.append(asdict(exhaustive_sufficient_subset(candidates, quality, threshold=1.0)))
        rows = extract_replay_features(trace)
        feature_rows.extend(rows)
        labels.extend(row.native_budget for row in rows)

    signal = analyze_budget_signals(feature_rows, labels)
    return {
        "schema_version": 1,
        "experiment_id": "m1-tier-t-smoke-v1",
        "claim_eligible": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "trace_schema_version": traces[0].manifest.schema_version,
        "calibration_trace_count": len(calibration),
        "evaluation_trace_count": len(evaluation),
        "calibrated_layer_budgets": layer_budgets,
        "calibrated_index_reuse": asdict(reuse),
        "policy_results": policy_results,
        "oracle_runs": oracle_runs,
        "native_budget_proxy_signal": asdict(signal),
        "m1_predictive_gate": "not_evaluated",
        "next_required_evidence": (
            "quality-derived sufficient-budget labels from trained Tier-S models"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
