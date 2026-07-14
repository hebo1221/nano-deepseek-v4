from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

EXPECTED_CONTEXTS = (80, 128, 256, 512, 1024)
EXPECTED_MULTIPLIERS = (1, 2, 4)
QUERY_TURNS_PER_CONVERSATION = {
    "single-remote-retrieval": 1,
    "multiple-independent-needles": 4,
    "associative-recall": 1,
    "multi-turn-query-shift": 4,
    "dense-global-aggregation": 8,
    "irrelevant-context-local-only": 1,
    "instruction-persistence": 4,
    "adversarial-lexical-distractors": 1,
    "long-generation-changing-evidence": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(
        payload.get("experiment_id") == "p1-layer-quota-calibration-pilot-v1",
        f"Wrong calibration experiment id: {path}",
    )
    _require(payload.get("source", {}).get("dirty") is False, f"Dirty source: {path}")
    _require(
        tuple(payload.get("families", ())) == PAPER_GRADE_WORKLOAD_FAMILIES,
        f"Workload family drift: {path}",
    )
    _require(
        tuple(payload.get("contexts", ())) == EXPECTED_CONTEXTS,
        f"Context grid drift: {path}",
    )
    examples = payload.get("examples_per_family")
    _require(isinstance(examples, int) and examples > 0, f"Invalid sample count: {path}")
    layers = tuple(sorted(int(layer) for layer in payload["captured_queries_per_layer"]))
    expected_per_layer = examples * sum(QUERY_TURNS_PER_CONVERSATION.values())
    observed_by_layer = {
        int(layer): count for layer, count in payload["captured_queries_per_layer"].items()
    }
    _require(
        all(observed_by_layer[layer] == expected_per_layer for layer in layers),
        f"Captured query count mismatch: {path}",
    )
    _require(
        payload.get("captured_query_count") == expected_per_layer * len(layers),
        f"Total captured query mismatch: {path}",
    )

    minimum = payload["minimum_blocks_per_layer"]
    calibrations = payload.get("calibrations", {})
    _require(
        tuple(int(key[:-1]) for key in calibrations) == EXPECTED_MULTIPLIERS,
        f"Budget multiplier drift: {path}",
    )
    digests: set[str] = set()
    audited_calibrations = {}
    for multiplier in EXPECTED_MULTIPLIERS:
        key = f"{multiplier}x"
        item = calibrations[key]
        signal = item["signal_config"]
        quota = item["quota"]
        normal = tuple(tuple(pair) for pair in quota["layer_budgets"])
        dense = tuple(tuple(pair) for pair in quota["dense_layer_budgets"])
        _require(tuple(layer for layer, _ in normal) == layers, f"Layer drift: {path}/{key}")
        _require(
            signal["global_block_budget"] == minimum * len(layers) * multiplier,
            f"Global budget drift: {path}/{key}",
        )
        _require(
            signal["max_extra_blocks_per_layer"] == minimum * (multiplier - 1),
            f"Uncertainty allowance drift: {path}/{key}",
        )
        _require(
            all(budget >= minimum for _, budget in normal),
            f"Layer floor violated: {path}/{key}",
        )
        _require(
            sum(budget for _, budget in normal) <= signal["global_block_budget"],
            f"Normal total budget exceeded: {path}/{key}",
        )
        _require(
            all(dense_budget >= dict(normal)[layer] for layer, dense_budget in dense),
            f"Dense quota below normal quota: {path}/{key}",
        )
        _require(
            sum(budget for _, budget in dense) <= signal["dense_fallback_block_budget"],
            f"Dense total budget exceeded: {path}/{key}",
        )
        digest = quota["calibration_digest"]
        _require(isinstance(digest, str) and len(digest) == 64, "Invalid calibration digest.")
        digests.add(digest)
        audited_calibrations[key] = {
            "signal_config": signal,
            "layer_budgets": normal,
            "dense_layer_budgets": dense,
            "score_demand_quantiles": quota["score_demand_quantiles"],
            "candidate_demand_quantiles": quota["candidate_demand_quantiles"],
            "calibration_digest": digest,
        }
    _require(len(digests) == len(EXPECTED_MULTIPLIERS), f"Non-unique digests: {path}")

    checkpoint = payload["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    _require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
    _require(checkpoint_path.stat().st_size == checkpoint["bytes"], "Checkpoint byte drift.")
    _require(_sha256(checkpoint_path) == checkpoint["sha256"], "Checkpoint digest drift.")
    return {
        "scale": payload["scale"],
        "seed": payload["seed"],
        "examples_per_family": examples,
        "captured_queries_per_layer": expected_per_layer,
        "captured_query_count": payload["captured_query_count"],
        "checkpoint": checkpoint,
        "source": payload["source"],
        "raw_artifact": {
            "path": str(path),
            "sha256": _sha256(path),
        },
        "calibrations": audited_calibrations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit two-scale P1 quota calibration pilots.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audited = [_audit(path) for path in args.input]
    _require({item["scale"] for item in audited} == {"s55", "s151"}, "Need both scales.")
    _require(
        len({item["source"]["commit"] for item in audited}) == 1,
        "Calibration pilots used different source commits.",
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-layer-quota-calibration-pilot-audit-v1",
        "interpretation": "two-scale calibration-only pilot; no held-out quality result",
        "validation": {
            "both_scales_present": True,
            "all_sources_clean": True,
            "all_checkpoint_digests_verified": True,
            "all_budget_bounds_verified": True,
            "targets_used_for_fit": False,
        },
        "runs": sorted(audited, key=lambda item: item["scale"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
