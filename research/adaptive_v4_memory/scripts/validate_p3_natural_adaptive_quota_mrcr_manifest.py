from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
NEEDLE_COUNTS = (2, 4, 8)
PREDICTIONS_PER_ARM = 1_500
ALLOWED_FAILURE_TYPES = (
    "unsupported-context",
    "empty-generation",
    "oom",
    "runtime-error",
)
DATASET_FILE_SHA256 = (
    "1c297b254bf64a31856b74918cd7db889a214503e0b67daa834e84f20df6aa93",
    "a5a1dc9ccc945623253d04d33c03d89aee2d676c88955ce368da2ab16a0ce94d",
    "4d4fa3d11ce064749de3cd039eef1a621e30a81c2c9b3e64f1df37f8afeaf312",
    "8dfdb94a208cf3eee73c4e7ac6ee8a5ccb7236c6934c13c6c5f67c0a9928cdf3",
    "65df601a2e0ae4a3cfb56920a6ef99f26c0de37c6b1018695e8aed684e6a94c1",
    "c80b19573bff1d38e1c157d6a0bdf9cfd1a8ab6372296174c9a7015e164189e3",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-mrcr-v1"
        and payload.get("status") == "frozen_before_any_adaptive_mrcr_prediction",
        "Adaptive MRCR cohort is not frozen before prediction.",
    )
    benchmark = payload.get("benchmark", {})
    model = payload.get("model", {})
    _require(
        benchmark.get("dataset_revision")
        == "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
        and tuple(benchmark.get("dataset_file_sha256", ())) == DATASET_FILE_SHA256
        and benchmark.get("scorer_sha256")
        == "9d933913f1a30f8435be09c8fa657163d10f63b5776b84d255bff1a15d6ef658"
        and tuple(benchmark.get("needle_counts", ())) == NEEDLE_COUNTS
        and benchmark.get("primary_token_bins") == 5
        and benchmark.get("samples_per_needle_bin_arm") == 100
        and benchmark.get("predictions_per_arm") == PREDICTIONS_PER_ARM
        and benchmark.get("paired_predictions_total")
        == PREDICTIONS_PER_ARM * len(ARMS)
        and "never truncate" in benchmark.get("overflow_policy", ""),
        "Adaptive MRCR benchmark coverage drifted.",
    )
    _require(
        model.get("revision") == "cdbee75f17c01a7cc42f958dc650907174af0554"
        and model.get("num_hidden_layers") == 36
        and model.get("snapshot_digest_set_sha256")
        == "c01e398afbd27d139b203e4b4b13d34dedec6d0a2db521083f55c50522c76e35",
        "Adaptive MRCR model contract drifted.",
    )
    _require(tuple(payload.get("arms", {})) == ARMS, "Adaptive MRCR arm pair drifted.")
    selection = payload.get("scorer_selection", {})
    _require(
        tuple(selection.get("eligible_arms", ()))
        == ("streaming_llm", "snapkv", "critical_expected_attention")
        and selection.get("mrcr_outcome_inspection_allowed") is False,
        "Adaptive MRCR scorer selection drifted.",
    )
    lifecycle = payload.get("cache_lifecycle_contract", {})
    _require(
        lifecycle.get("adaptive_allocation_scope") == "initial context prefill only"
        and lifecycle.get("same_initial_global_kept_tokens") is True
        and lifecycle.get("same_scorer") is True
        and lifecycle.get("same_protected_prefix") is True
        and "append identically" in lifecycle.get("final_query_tokens", "")
        and lifecycle.get("continuous_refresh_claim_available") is False,
        "Adaptive MRCR lifecycle boundary drifted.",
    )
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("paired_bootstrap_seed") == 9_571_506
        and statistics.get("holm_family_size") == 15
        and statistics.get("report_by_needle_count") is True
        and statistics.get("report_by_token_bin") is True
        and statistics.get("outcome_dependent_model_or_cell_selection") is False
        and "scores zero" in statistics.get("failure_policy", ""),
        "Adaptive MRCR statistics drifted.",
    )
    failure = payload.get("failure_reporting", {})
    _require(
        tuple(failure.get("allowed_failure_types", ())) == ALLOWED_FAILURE_TYPES
        and failure.get("unknown_or_missing_failure_type_policy")
        == "reject the artifact during audit"
        and failure.get("successful_record_policy") == "failure_type must be null",
        "Adaptive MRCR failure reporting drifted.",
    )
    gate = payload.get("confirmation_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("overall_score_difference_minimum") == 0.0
        and gate.get("paired_bootstrap_lower_bound_minimum") == -0.02
        and gate.get("nonnegative_needle_bin_count_minimum") == 10
        and gate.get("worst_needle_bin_regression_minimum") == -0.08
        and gate.get("maximum_initial_global_kept_token_relative_error") == 0.0,
        "Adaptive MRCR gate drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "initial causal layer-quota heuristic",
                "2/4/8-needle MRCR",
                "does not establish continuous adaptive reallocation",
                "cross-model transfer",
                "official DeepSeek-V4",
            )
        ),
        "Adaptive MRCR claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate adaptive MRCR cohort.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-mrcr-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print(f"validated {PREDICTIONS_PER_ARM} paired MRCR examples per arm")


if __name__ == "__main__":
    main()
