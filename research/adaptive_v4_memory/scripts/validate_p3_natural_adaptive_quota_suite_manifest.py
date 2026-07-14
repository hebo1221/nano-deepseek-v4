from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BENCHMARKS = ("RULER", "SCBench", "LongBench-v2", "MRCR")
ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
PREDICTIONS_PER_ARM = {
    "RULER": 32_500,
    "SCBench": 10_286,
    "LongBench-v2": 503,
    "MRCR": 1_500,
}
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_DIGEST_SET = "c01e398afbd27d139b203e4b4b13d34dedec6d0a2db521083f55c50522c76e35"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-suite-v1"
        and payload.get("status") == "frozen_before_any_adaptive_suite_summary",
        "Adaptive natural suite is not frozen before aggregation.",
    )
    model = payload.get("model", {})
    _require(
        model.get("revision") == MODEL_REVISION
        and model.get("snapshot_digest_set_sha256") == MODEL_DIGEST_SET,
        "Adaptive natural suite model contract drifted.",
    )
    _require(tuple(payload.get("arms", ())) == ARMS, "Adaptive natural suite arms drifted.")
    components = payload.get("components", {})
    _require(tuple(components) == BENCHMARKS, "Adaptive natural suite components drifted.")
    for benchmark, expected_predictions in PREDICTIONS_PER_ARM.items():
        component = components[benchmark]
        _require(
            component.get("predictions_per_arm") == expected_predictions
            and component.get("paired_predictions_total") == expected_predictions * len(ARMS)
            and isinstance(component.get("manifest_sha256"), str)
            and len(component["manifest_sha256"]) == 64
            and component.get("summary_experiment_id", "").endswith("-audit-v1")
            and bool(component.get("required_gate_checks")),
            f"Adaptive natural suite {benchmark} contract drifted.",
        )
    coverage = payload.get("coverage", {})
    expected_per_arm = sum(PREDICTIONS_PER_ARM.values())
    _require(
        coverage.get("benchmarks") == len(BENCHMARKS)
        and coverage.get("predictions_per_arm") == expected_per_arm
        and coverage.get("paired_predictions_total") == expected_per_arm * len(ARMS)
        and tuple(coverage.get("context_range_tokens", ())) == (8_192, 131_072)
        and "official GPT-4o judge unavailable"
        in coverage.get("excluded_from_adaptive_suite", {}).get("LongMemEval", ""),
        "Adaptive natural suite coverage drifted.",
    )
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("no_cross_benchmark_score_pooling") is True
        and statistics.get("no_cross_benchmark_p_value_pooling") is True
        and statistics.get("component_confidence_intervals_preserved") is True
        and statistics.get("component_multiplicity_corrections_preserved") is True
        and statistics.get("suite_gate_uses_component_p_values") is False
        and statistics.get("outcome_dependent_benchmark_selection") is False,
        "Adaptive natural suite statistics drifted.",
    )
    gate = payload.get("confirmation_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("required_terminal_components") == len(BENCHMARKS)
        and gate.get("minimum_component_confirmation_gates_passed") == 3
        and gate.get("minimum_nonnegative_overall_effects") == len(BENCHMARKS)
        and gate.get("all_required_physical_and_failure_checks_must_pass") is True
        and gate.get("all_components_same_model_revision") is True
        and gate.get("all_components_same_arm_pair") is True
        and gate.get("all_components_same_initial_global_token_budget") is True,
        "Adaptive natural suite confirmation gate drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "four frozen Qwen3-4B natural-language cohorts",
                "does not pool incomparable benchmark scores",
                "LongMemEval",
                "continuous adaptive reallocation",
                "cross-model transfer",
                "official DeepSeek-V4",
            )
        ),
        "Adaptive natural suite claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the adaptive natural suite contract.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-suite-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print(f"validated {len(BENCHMARKS)} adaptive natural benchmark contracts")


if __name__ == "__main__":
    main()
