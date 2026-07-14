from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verify_p3_natural_model import canonical_digest_set

EXPECTED_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
EXPECTED_LENGTHS = (8192, 32768, 131072)
EXPECTED_MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
EXPECTED_RULER_REVISION = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
EXPECTED_KVPRESS_REVISION = "6d965557a5b9f0201a2301b23c454473dd681d0d"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("experiment_id") == "p3-cross-family-ruler-transfer-v1"
        and payload.get("status") == "frozen_before_any_p3_model_prediction",
        "Cross-family experiment identity or freeze status drifted.",
    )
    amendments = payload.get("amendments")
    _require(
        isinstance(amendments, list)
        and len(amendments) == 2
        and "before any P3 model prediction" in amendments[0].get("timing", "")
        and "before any cross-family dataset row" in amendments[1].get("timing", ""),
        "Cross-family pre-outcome amendment record drifted.",
    )

    relationship = payload.get("relationship_to_primary", {})
    _require(
        relationship.get("primary_model_family") == "Qwen3"
        and relationship.get("transfer_model_family") == "Phi-4"
        and relationship.get("pooled_with_primary") is False
        and relationship.get("outcome_dependent_execution") is False
        and relationship.get("phi_specific_tuning_allowed") is False,
        "Cross-family independence boundary drifted.",
    )

    sequence = payload.get("sequence_gate", {})
    _require(
        sequence.get("required_primary_core_summary", "").endswith(
            "p2-core-quality-matrix.strict.summary.json"
        )
        and sequence.get("required_nine_seed_core_summary", "").endswith(
            "p2-nine-seed-core.summary.json"
        )
        and sequence.get("required_nine_seed_causal_summary", "").endswith(
            "p2-nine-seed-causal.summary.json"
        )
        and sequence.get("required_fixed_selection", "").endswith("fixed-baseline-selection.json")
        and "every model prediction require terminal nine-seed P2 evidence"
        in sequence.get("policy", ""),
        "Cross-family sequence gate drifted.",
    )

    model = payload.get("model", {})
    files = model.get("snapshot_files_sha256")
    _require(
        model.get("repo_id") == "microsoft/Phi-4-mini-instruct"
        and model.get("revision") == EXPECTED_MODEL_REVISION
        and model.get("license") == "mit"
        and model.get("architecture") == "Phi3ForCausalLM"
        and model.get("parameter_count") == 3_836_021_760
        and model.get("dtype") == "bfloat16"
        and model.get("trust_remote_code") is False
        and model.get("maximum_supported_context_tokens") == 131_072
        and model.get("snapshot_bytes") == 7_694_054_130
        and model.get("weight_shard_file_bytes") == 7_672_066_216
        and model.get("indexed_tensor_bytes") == 7_672_043_520
        and isinstance(files, dict)
        and len(files) == 21
        and all(
            isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
            for name, digest in files.items()
        )
        and canonical_digest_set(files) == model.get("snapshot_digest_set_sha256"),
        "Cross-family frozen model contract drifted.",
    )

    upstreams = payload.get("upstreams", {})
    _require(
        upstreams.get("ruler", {}).get("revision") == EXPECTED_RULER_REVISION
        and upstreams.get("kvpress", {}).get("revision") == EXPECTED_KVPRESS_REVISION,
        "Cross-family public-code revisions drifted.",
    )

    benchmark = payload.get("benchmark", {})
    predictions_per_arm = len(EXPECTED_LENGTHS) * len(EXPECTED_TASKS) * 100
    _require(
        benchmark.get("name") == "RULER"
        and tuple(benchmark.get("tasks", ())) == EXPECTED_TASKS
        and tuple(benchmark.get("lengths_tokens", ())) == EXPECTED_LENGTHS
        and benchmark.get("samples_per_task_length") == 100
        and benchmark.get("generation_seed") == 42
        and benchmark.get("tasks_per_length") == len(EXPECTED_TASKS)
        and benchmark.get("predictions_per_arm") == predictions_per_arm
        and benchmark.get("paired_predictions_total") == 2 * predictions_per_arm
        and "reject rather than silently truncate" in benchmark.get("exact_token_rule", ""),
        "Cross-family RULER coverage drifted.",
    )

    arms = payload.get("arms", {})
    _require(
        set(arms) == {"native-dense", "qwen-selected-memory-matched"}
        and arms["native-dense"] == {"press_name": "no_press", "compression_ratio": 0.0}
        and arms["qwen-selected-memory-matched"].get("compression_ratio") == 0.5
        and "without Phi-specific reselection"
        in arms["qwen-selected-memory-matched"].get("selection_rule", ""),
        "Cross-family arm contract drifted.",
    )

    statistics = payload.get("statistics", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("paired_bootstrap_confidence") == 0.95
        and statistics.get("paired_bootstrap_seed") == 9_171_501
        and statistics.get("exact_task_sign_flip_assignments_per_length") == 8192
        and statistics.get("length_holm_family_size") == len(EXPECTED_LENGTHS),
        "Cross-family statistical contract drifted.",
    )
    gate = payload.get("transfer_gate", {})
    _require(
        gate.get("all_required") is True
        and gate.get("overall_mean_accuracy_difference_minimum") == -0.01
        and gate.get("overall_paired_bootstrap_lower_bound_minimum") == -0.02
        and gate.get("worst_task_length_regression_minimum") == -0.05
        and gate.get("maximum_failure_rate_increase") == 0.01
        and gate.get("maximum_realized_kv_fraction") == 0.51
        and "maximum cell ratio" in gate.get("realized_kv_fraction_aggregation", ""),
        "Cross-family transfer gate drifted.",
    )

    return {
        "model_revision": model["revision"],
        "snapshot_files": len(files),
        "lengths": list(EXPECTED_LENGTHS),
        "tasks": len(EXPECTED_TASKS),
        "predictions_per_arm": predictions_per_arm,
        "paired_predictions_total": 2 * predictions_per_arm,
        "task_sign_flip_assignments_per_length": 8192,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Phi-4-mini cross-family RULER transfer contract."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    _require(isinstance(payload, dict), "Cross-family manifest must be a JSON object.")
    print(json.dumps(validate_manifest(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
