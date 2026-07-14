from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CATEGORIES = (
    "single-document-qa",
    "multi-document-qa",
    "long-in-context-learning",
    "long-dialogue-history-understanding",
    "code-repository-understanding",
    "long-structured-data-understanding",
)
ARMS = ("fixed+pins", "cross-family-adaptive-quota+pins")
PREDICTIONS_PER_ARM = 503
ALLOWED_FAILURE_TYPES = (
    "unsupported-context",
    "empty-generation",
    "oom",
    "runtime-error",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id")
        == "p3-cross-family-adaptive-quota-longbench-v2-v1"
        and payload.get("status")
        == "frozen_before_any_phi_longbench_v2_prediction",
        "Phi adaptive LongBench-v2 cohort is not frozen before prediction.",
    )
    relationship = payload.get("relationship_to_other_cohorts", {})
    _require(
        relationship.get("design") == "Qwen3/Phi-4 by RULER/LongBench-v2 transfer grid"
        and relationship.get("primary_natural_model_family") == "Qwen3"
        and relationship.get("transfer_model_family") == "Phi-4"
        and relationship.get("pooled_with_qwen_longbench_v2") is False
        and relationship.get("pooled_with_phi_ruler") is False
        and relationship.get("phi_specific_tuning_or_reselection_allowed") is False,
        "Phi LongBench-v2 independence contract drifted.",
    )
    model = payload.get("model", {})
    _require(
        model.get("revision") == "cfbefacb99257ffa30c83adab238a50856ac3083"
        and model.get("num_hidden_layers") == 32
        and model.get("maximum_supported_context_tokens") == 131_072
        and model.get("snapshot_digest_set_sha256")
        == "102f70c901015c02c6157e947ffbe2e1aa5667c8b28a0b3df17f4bc269e9889c",
        "Phi LongBench-v2 model contract drifted.",
    )
    benchmark = payload.get("benchmark", {})
    _require(
        benchmark.get("dataset_revision")
        == "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
        and benchmark.get("code_revision")
        == "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
        and benchmark.get("dataset_sha256")
        == "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
        and benchmark.get("prompt_sha256")
        == "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"
        and benchmark.get("answer_parser")
        == "research/adaptive_v4_memory/scripts/p3_natural_metrics.py"
        and tuple(benchmark.get("categories", ())) == CATEGORIES
        and benchmark.get("predictions_per_arm") == PREDICTIONS_PER_ARM
        and benchmark.get("paired_predictions_total")
        == PREDICTIONS_PER_ARM * len(ARMS)
        and benchmark.get("generation_reserve_tokens") == 128
        and benchmark.get("secondary_slice_fields")
        == ["sub_domain", "difficulty", "length_stratum"]
        and "no secondary slice is a confirmation gate"
        in benchmark.get("secondary_slice_policy", "")
        and "never truncate" in benchmark.get("overflow_policy", ""),
        "Phi adaptive LongBench-v2 immutable benchmark contract drifted.",
    )
    _require(tuple(payload.get("arms", {})) == ARMS, "Phi adaptive arm pair drifted.")
    selection = payload.get("scorer_selection", {})
    _require(
        tuple(selection.get("eligible_arms", ()))
        == ("streaming_llm", "snapkv", "critical_expected_attention")
        and selection.get("phi_outcome_inspection_allowed") is False
        and selection.get("phi_specific_reselection_allowed") is False,
        "Phi LongBench-v2 scorer selection drifted.",
    )
    lifecycle = payload.get("cache_lifecycle_contract", {})
    _require(
        lifecycle.get("adaptive_allocation_scope") == "initial context prefill only"
        and lifecycle.get("same_initial_global_kept_tokens") is True
        and lifecycle.get("same_scorer") is True
        and lifecycle.get("same_protected_prefix") is True
        and "append identically" in lifecycle.get("question_tokens", "")
        and lifecycle.get("continuous_refresh_claim_available") is False,
        "Phi LongBench-v2 lifecycle boundary drifted.",
    )
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("paired_bootstrap_seed") == 9_671_507
        and statistics.get("holm_family_size") == len(CATEGORIES)
        and statistics.get("report_by_category") is True
        and statistics.get("report_by_sub_domain") is True
        and statistics.get("report_by_difficulty") is True
        and statistics.get("report_by_length_stratum") is True
        and statistics.get("secondary_slices_are_descriptive") is True
        and statistics.get("outcome_dependent_model_or_slice_selection") is False
        and "scores zero" in statistics.get("failure_policy", ""),
        "Phi LongBench-v2 statistics drifted.",
    )
    failure = payload.get("failure_reporting", {})
    _require(
        tuple(failure.get("allowed_failure_types", ())) == ALLOWED_FAILURE_TYPES
        and failure.get("unknown_or_missing_failure_type_policy")
        == "reject the artifact during audit"
        and failure.get("successful_record_policy") == "failure_type must be null"
        and "scores every allowed failure zero"
        in failure.get("paired_scoring_policy", ""),
        "Phi LongBench-v2 failure reporting drifted.",
    )
    gate = payload.get("confirmation_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("overall_accuracy_difference_minimum") == 0.0
        and gate.get("paired_bootstrap_lower_bound_minimum") == -0.02
        and gate.get("nonnegative_category_count_minimum") == 4
        and gate.get("worst_category_regression_minimum") == -0.08
        and gate.get("maximum_failure_rate_increase") == 0.02
        and gate.get("maximum_initial_global_kept_token_relative_error") == 0.0
        and gate.get("maximum_initial_hot_resident_byte_relative_difference") == 0.01,
        "Phi LongBench-v2 confirmation gate drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "one held-out Phi-4-mini checkpoint",
                "does not establish continuous adaptive reallocation",
                "second full natural-language suite",
                "broad model-population generalization",
                "official DeepSeek-V4",
            )
        ),
        "Phi LongBench-v2 claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Phi adaptive LongBench-v2 cohort."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-cross-family-adaptive-quota-longbench-v2-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print(f"validated {PREDICTIONS_PER_ARM} paired Phi LongBench-v2 examples per arm")


if __name__ == "__main__":
    main()
