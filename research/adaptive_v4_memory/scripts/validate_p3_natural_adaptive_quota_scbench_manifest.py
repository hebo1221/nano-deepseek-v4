from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODES = ("multi-turn", "multi-request")
TASKS = (
    "scbench_choice_eng",
    "scbench_kv",
    "scbench_many_shot",
    "scbench_mf",
    "scbench_prefix_suffix",
    "scbench_qa_chn",
    "scbench_qa_eng",
    "scbench_repoqa",
    "scbench_repoqa_and_kv",
    "scbench_summary",
    "scbench_summary_with_needles",
    "scbench_vt",
)
ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
PREDICTIONS_PER_ARM = 10_286


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-scbench-v1"
        and payload.get("status") == "frozen_before_any_adaptive_scbench_prediction",
        "Adaptive SCBench cohort is not frozen before prediction.",
    )
    benchmark = payload.get("benchmark", {})
    _require(
        benchmark.get("dataset_revision") == "283310bb8c5ba6909dd9a6b1be087d2937f76f6d"
        and benchmark.get("code_revision") == "a4eb395f949ea39e871f9bc586d683390692c6be"
        and tuple(benchmark.get("modes", ())) == MODES
        and tuple(benchmark.get("tasks", ())) == TASKS
        and benchmark.get("shared_context_rows_per_mode") == 922
        and benchmark.get("turn_predictions_per_mode") == 5_143
        and benchmark.get("predictions_per_arm") == PREDICTIONS_PER_ARM
        and benchmark.get("paired_predictions_total") == PREDICTIONS_PER_ARM * len(ARMS),
        "Adaptive SCBench coverage drifted.",
    )
    _require(tuple(payload.get("arms", {})) == ARMS, "Adaptive SCBench arm pair drifted.")
    selection = payload.get("scorer_selection", {})
    _require(
        tuple(selection.get("eligible_arms", ()))
        == ("streaming_llm", "snapkv", "critical_expected_attention")
        and selection.get("scbench_outcome_inspection_allowed") is False,
        "Adaptive SCBench scorer selection drifted.",
    )
    lifecycle = payload.get("cache_lifecycle_contract", {})
    _require(
        lifecycle.get("adaptive_allocation_scope") == "initial shared-context prefill only"
        and lifecycle.get("same_initial_global_kept_tokens") is True
        and lifecycle.get("same_scorer") is True
        and lifecycle.get("same_protected_prefix") is True
        and lifecycle.get("continuous_refresh_claim_available") is False
        and "append identically" in lifecycle.get("post_prefill_turn_tokens", ""),
        "Adaptive SCBench lifecycle boundary drifted.",
    )
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("paired_cluster_bootstrap_resamples") == 10_000
        and statistics.get("paired_cluster_bootstrap_seed") == 9_371_504
        and statistics.get("holm_family_size") == len(MODES) * len(TASKS)
        and "scores zero" in statistics.get("failure_policy", ""),
        "Adaptive SCBench statistics drifted.",
    )
    gate = payload.get("confirmation_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("overall_score_difference_minimum") == 0.0
        and gate.get("paired_cluster_bootstrap_lower_bound_minimum") == -0.01
        and gate.get("maximum_initial_global_kept_token_relative_error") == 0.0,
        "Adaptive SCBench gate drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "initial causal layer-quota heuristic",
                "does not establish continuous adaptive reallocation",
                "cross-model transfer",
                "official DeepSeek-V4",
            )
        ),
        "Adaptive SCBench claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the adaptive SCBench cohort.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-scbench-v1.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    validate_manifest(payload)
    print(f"validated {PREDICTIONS_PER_ARM} paired adaptive SCBench turns per arm")


if __name__ == "__main__":
    main()
