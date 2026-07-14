from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
ALLOWED_FAILURE_TYPES = (
    "unsupported-context",
    "empty-generation",
    "oom",
    "runtime-error",
    "judge-blocked",
)
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_DIGEST_SET = "c01e398afbd27d139b203e4b4b13d34dedec6d0a2db521083f55c50522c76e35"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-longmemeval-v1"
        and payload.get("status")
        == "frozen_before_any_adaptive_longmemeval_generation",
        "Adaptive LongMemEval cohort is not frozen before generation.",
    )
    model = payload.get("model", {})
    _require(
        model.get("revision") == MODEL_REVISION
        and model.get("num_hidden_layers") == 36
        and model.get("snapshot_digest_set_sha256") == MODEL_DIGEST_SET,
        "Adaptive LongMemEval model contract drifted.",
    )
    benchmark = payload.get("benchmark", {})
    _require(
        benchmark.get("dataset_revision")
        == "98d7416c24c778c2fee6e6f3006e7a073259d48f"
        and benchmark.get("dataset_sha256")
        == "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
        and benchmark.get("official_judge_source_sha256")
        == "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
        and benchmark.get("examples_per_arm") == 500
        and benchmark.get("paired_predictions_total") == 1_000
        and benchmark.get("generation_reserve_tokens") == 512
        and "never truncate" in benchmark.get("overflow_policy", ""),
        "Adaptive LongMemEval benchmark coverage drifted.",
    )
    _require(tuple(payload.get("arms", {})) == ARMS, "Adaptive LongMemEval arms drifted.")
    selection = payload.get("scorer_selection", {})
    _require(
        tuple(selection.get("eligible_arms", ()))
        == ("streaming_llm", "snapkv", "critical_expected_attention")
        and selection.get("longmemeval_outcome_inspection_allowed") is False,
        "Adaptive LongMemEval scorer selection drifted.",
    )
    lifecycle = payload.get("cache_lifecycle_contract", {})
    _require(
        lifecycle.get("adaptive_allocation_scope")
        == "initial history-context prefill only"
        and lifecycle.get("same_initial_global_kept_tokens") is True
        and lifecycle.get("same_scorer") is True
        and lifecycle.get("same_protected_prefix") is True
        and "append identically" in lifecycle.get("current_question_tokens", "")
        and lifecycle.get("continuous_refresh_claim_available") is False,
        "Adaptive LongMemEval lifecycle boundary drifted.",
    )
    official = payload.get("official_metric_contract", {})
    _require(
        official.get("judge_model") == "gpt-4o-2024-08-06"
        and official.get("initial_execution_mode") == "blocked"
        and official.get("quality_classification_before_official_judging")
        == "unverified"
        and official.get("confirmation_gate_available_before_official_judging")
        is False
        and "no deterministic" in official.get("auxiliary_metric_policy", "")
        and "never invoke" in official.get("paid_judge_policy", ""),
        "Adaptive LongMemEval official-metric boundary drifted.",
    )
    failure = payload.get("failure_reporting", {})
    _require(
        tuple(failure.get("allowed_failure_types", ())) == ALLOWED_FAILURE_TYPES
        and failure.get("unknown_or_missing_failure_type_policy")
        == "reject the artifact during audit"
        and "score null" in failure.get(
            "successful_generation_with_blocked_judge_policy", ""
        ),
        "Adaptive LongMemEval failure reporting drifted.",
    )
    reporting = payload.get("reporting", {})
    _require(
        reporting.get("cross_benchmark_suite_membership") is False
        and reporting.get("quality")
        == "report official quality as unverified until an explicit paid-judge artifact exists"
        and "no proxy substitution"
        in reporting.get("reason_excluded_from_adaptive_confirmation_suite", ""),
        "Adaptive LongMemEval reporting boundary drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "paired response-generation completion",
                "no model-quality success",
                "negative-result claim",
                "continuous adaptive reallocation",
                "cross-model transfer",
                "official DeepSeek-V4",
            )
        ),
        "Adaptive LongMemEval claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate adaptive LongMemEval cohort.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longmemeval-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print("validated 500 adaptive LongMemEval generations per arm")


if __name__ == "__main__":
    main()
