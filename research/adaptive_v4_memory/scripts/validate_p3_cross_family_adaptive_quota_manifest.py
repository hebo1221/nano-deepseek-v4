from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LENGTHS = (8_192, 32_768, 131_072)
TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
ARMS = ("fixed+pins", "cross-family-adaptive-quota+pins")
SAMPLES_PER_TASK_LENGTH = 100
EXPECTED_EXAMPLES = len(LENGTHS) * len(TASKS) * SAMPLES_PER_TASK_LENGTH
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
        payload.get("experiment_id") == "p3-cross-family-adaptive-quota-ruler-v1"
        and payload.get("status") == "frozen_before_any_cross_family_adaptive_prediction",
        "Cross-family adaptive cohort is not frozen before prediction.",
    )
    relationship = payload.get("relationship_to_other_cohorts", {})
    _require(
        relationship.get("primary_natural_model_family") == "Qwen3"
        and relationship.get("transfer_model_family") == "Phi-4"
        and relationship.get("pooled_with_qwen") is False
        and relationship.get("pooled_with_phi_native_fixed_transfer") is False
        and relationship.get("phi_specific_tuning_or_reselection_allowed") is False,
        "Cross-family independence contract drifted.",
    )
    model = payload.get("model", {})
    _require(
        model.get("revision") == "cfbefacb99257ffa30c83adab238a50856ac3083"
        and model.get("num_hidden_layers") == 32
        and model.get("maximum_supported_context_tokens") == 131_072
        and len(model.get("snapshot_digest_set_sha256", "")) == 64,
        "Phi model contract drifted.",
    )
    benchmark = payload.get("benchmark", {})
    _require(
        tuple(benchmark.get("lengths_tokens", ())) == LENGTHS
        and tuple(benchmark.get("tasks", ())) == TASKS
        and benchmark.get("samples_per_task_length") == SAMPLES_PER_TASK_LENGTH
        and benchmark.get("predictions_per_arm") == EXPECTED_EXAMPLES
        and benchmark.get("paired_predictions_total") == EXPECTED_EXAMPLES * len(ARMS),
        "Cross-family adaptive RULER coverage drifted.",
    )
    _require(tuple(payload.get("arms", {})) == ARMS, "Adaptive arm pair drifted.")
    scorer = payload.get("scorer_selection", {})
    _require(
        tuple(scorer.get("eligible_arms", ()))
        == ("streaming_llm", "snapkv", "critical_expected_attention")
        and scorer.get("phi_outcome_inspection_allowed") is False
        and "overrides compress()" in scorer.get("excluded", {}).get("pyramidkv", ""),
        "Score-compatible selection contract drifted.",
    )
    physical = payload.get("physical_contract", {})
    _require(
        physical.get("same_global_kept_tokens") is True
        and physical.get("same_compression_ratio") == 0.5
        and physical.get("same_scorer") is True
        and physical.get("same_protected_prefix") is True,
        "Same-budget causal contract drifted.",
    )
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("paired_bootstrap_seed") == 9_271_503
        and statistics.get("exact_task_sign_flip_assignments_per_length") == 8192
        and statistics.get("holm_family_size") == len(LENGTHS)
        and "score zero" in statistics.get("failure_policy", ""),
        "Statistical contract drifted.",
    )
    failure_reporting = payload.get("failure_reporting", {})
    _require(
        tuple(failure_reporting.get("allowed_failure_types", ())) == ALLOWED_FAILURE_TYPES
        and failure_reporting.get("unknown_failure_type_policy")
        == "reject the artifact during audit"
        and failure_reporting.get("missing_failure_type_policy")
        == "reject the artifact during audit"
        and failure_reporting.get("successful_record_policy") == "failure_type must be null"
        and "effective score zero" in failure_reporting.get("scoring_policy", ""),
        "Operational failure reporting contract drifted.",
    )
    gate = payload.get("confirmation_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("overall_accuracy_difference_minimum") == 0.0
        and gate.get("paired_bootstrap_lower_bound_minimum") == -0.01
        and gate.get("nonnegative_length_count_minimum") == 2
        and gate.get("maximum_global_kept_token_relative_error") == 0.0,
        "Confirmation gate drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "one held-out Phi-family RULER cohort",
                "not unchanged transfer of the synthetic P2 controller",
                "not a second full natural-language suite",
                "official DeepSeek-V4",
            )
        ),
        "Cross-family adaptive claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Phi adaptive cohort.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-cross-family-adaptive-quota-ruler-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print(f"validated {EXPECTED_EXAMPLES} paired Phi adaptive examples per arm")


if __name__ == "__main__":
    main()
