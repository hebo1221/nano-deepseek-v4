from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
SLICE_FIELDS = ("domain", "sub_domain", "difficulty", "length_stratum")
ALLOWED_FAILURE_TYPES = (
    "unsupported-context",
    "empty-generation",
    "oom",
    "runtime-error",
)
EXPECTED_EXAMPLES = 503


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id")
        == "p3-natural-adaptive-quota-longbench-v2-v1"
        and payload.get("status")
        == "frozen_before_any_adaptive_longbench_v2_prediction",
        "Adaptive LongBench v2 cohort is not frozen before prediction.",
    )
    model = payload.get("model", {})
    _require(
        model.get("revision") == "cdbee75f17c01a7cc42f958dc650907174af0554"
        and model.get("num_hidden_layers") == 36
        and model.get("maximum_supported_context_tokens") == 262_144
        and len(model.get("snapshot_digest_set_sha256", "")) == 64,
        "Adaptive LongBench v2 model contract drifted.",
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
        and benchmark.get("examples_per_arm") == EXPECTED_EXAMPLES
        and benchmark.get("paired_predictions_total") == EXPECTED_EXAMPLES * len(ARMS)
        and benchmark.get("generation_reserve_tokens") == 128
        and tuple(benchmark.get("secondary_slice_fields", ())) == SLICE_FIELDS
        and "n >= 10" in benchmark.get("secondary_slice_policy", ""),
        "Adaptive LongBench v2 benchmark contract drifted.",
    )
    _require(tuple(payload.get("arms", {})) == ARMS, "Adaptive LongBench v2 arm pair drifted.")
    selection = payload.get("scorer_selection", {})
    _require(
        tuple(selection.get("eligible_arms", ()))
        == ("streaming_llm", "snapkv", "critical_expected_attention")
        and selection.get("longbench_v2_outcome_inspection_allowed") is False,
        "Adaptive LongBench v2 scorer selection drifted.",
    )
    physical = payload.get("physical_contract", {})
    _require(
        physical.get("adaptive_allocation_scope") == "document-context prefill only"
        and physical.get("same_global_kept_tokens") is True
        and physical.get("same_scorer") is True
        and physical.get("same_protected_prefix") is True
        and "append identically" in physical.get("question_tokens", ""),
        "Adaptive LongBench v2 physical contract drifted.",
    )
    failure = payload.get("failure_reporting", {})
    _require(
        tuple(failure.get("allowed_failure_types", ())) == ALLOWED_FAILURE_TYPES
        and failure.get("unknown_or_missing_failure_type_policy")
        == "reject the artifact during audit"
        and "score allowed failures zero" in failure.get("paired_scoring_policy", ""),
        "Adaptive LongBench v2 failure reporting drifted.",
    )
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("paired_bootstrap_seed") == 9_471_505
        and statistics.get("secondary_slices_are_descriptive") is True
        and statistics.get("outcome_dependent_model_or_slice_selection") is False
        and "scores zero" in statistics.get("failure_policy", ""),
        "Adaptive LongBench v2 statistics drifted.",
    )
    gate = payload.get("confirmation_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("overall_score_difference_minimum") == 0.0
        and gate.get("paired_bootstrap_lower_bound_minimum") == -0.01
        and gate.get("maximum_global_kept_token_relative_error") == 0.0,
        "Adaptive LongBench v2 confirmation gate drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "one Qwen3-4B LongBench v2 multiple-choice cohort",
                "does not establish performance on open-ended generation",
                "cross-model transfer",
                "official DeepSeek-V4",
            )
        ),
        "Adaptive LongBench v2 claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate adaptive LongBench v2 cohort.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longbench-v2-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print(f"validated {EXPECTED_EXAMPLES} paired adaptive LongBench v2 examples per arm")


if __name__ == "__main__":
    main()
