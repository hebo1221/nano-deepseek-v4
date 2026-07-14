from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
LENGTHS = (8192, 16384, 32768, 65536, 131072)
SCORE_COMPATIBLE = (
    "streaming_llm",
    "snapkv",
    "critical_expected_attention",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-ruler-v1"
        and payload.get("status") == "frozen_before_any_compatibility_arm_prediction",
        "Natural adaptive-quota identity or freeze status drifted.",
    )
    model = payload.get("model", {})
    _require(
        model.get("revision") == "cdbee75f17c01a7cc42f958dc650907174af0554"
        and model.get("architecture") == "Qwen3ForCausalLM"
        and model.get("num_hidden_layers") == 36,
        "Natural adaptive-quota model contract drifted.",
    )
    benchmark = payload.get("benchmark", {})
    expected = len(LENGTHS) * 13 * 500
    _require(
        tuple(benchmark.get("lengths_tokens", ())) == LENGTHS
        and benchmark.get("predictions_per_arm") == expected
        and benchmark.get("paired_predictions_total") == 2 * expected
        and benchmark.get("generation_seed") == 42,
        "Natural adaptive-quota benchmark coverage drifted.",
    )
    selection = payload.get("score_compatible_baseline_selection", {})
    amendments = payload.get("amendments", [])
    _require(
        tuple(selection.get("eligible_arms", ())) == SCORE_COMPATIBLE
        and selection.get("compression_ratio") == 0.5
        and "do not inspect Qwen3-4B" in selection.get("rule", ""),
        "Natural adaptive-quota baseline selection drifted.",
    )
    _require(
        isinstance(amendments, list)
        and len(amendments) == 1
        and "before any compatibility-arm prediction" in amendments[0].get("timing", "")
        and "PyramidKV" in amendments[0].get("change", ""),
        "Natural adaptive-quota pre-outcome amendment drifted.",
    )
    arms = payload.get("arms", {})
    adaptive = arms.get(ARMS[1], {})
    _require(
        tuple(arms) == ARMS
        and adaptive.get("maximum_layer_adjustment_fraction") == 0.25
        and "earlier-layer" in adaptive.get("causal_prior", "")
        and "exact residual" in adaptive.get("feasibility_rule", ""),
        "Natural adaptive-quota controller contract drifted.",
    )
    pins = payload.get("pins", {})
    _require(
        pins.get("span") == [0, 4]
        and pins.get("same_budget") is True
        and "fail closed" in pins.get("failure_policy", ""),
        "Natural adaptive-quota pin contract drifted.",
    )
    statistics = payload.get("statistics", {})
    gate = payload.get("confirmation_gate", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("paired_bootstrap_seed") == 9_171_502
        and statistics.get("holm_family_size") == len(LENGTHS)
        and gate.get("maximum_global_kept_token_relative_error") == 0.0
        and gate.get("maximum_hot_resident_byte_relative_difference") == 0.01,
        "Natural adaptive-quota statistics or gate drifted.",
    )
    _require(
        "not an unchanged transfer" in payload.get("claim_boundary", ""),
        "Natural adaptive-quota claim boundary drifted.",
    )
    return {
        "arms": list(ARMS),
        "lengths": list(LENGTHS),
        "predictions_per_arm": expected,
        "paired_predictions_total": 2 * expected,
        "score_compatible_candidates": list(SCORE_COMPATIBLE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the natural adaptive-quota cohort.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-ruler-v1.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    _require(isinstance(payload, dict), "Natural adaptive-quota manifest must be an object.")
    print(json.dumps(validate_manifest(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
