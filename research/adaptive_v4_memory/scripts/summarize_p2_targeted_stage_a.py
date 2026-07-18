from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import audit_p2_targeted_stage_a_integrity as integrity
import evaluate_p2_targeted_stage_a_shard as stage_a
import numpy as np

SUMMARY_EXPERIMENT_ID = "p2-targeted-stage-a-clean-layer-identity-summary-v1"
DEFAULT_OUTPUT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2-targeted-stage-a-quality.summary.json"
)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_BASE_SEED = 9_171_803
CELL_ORDER = (("s151", "2x"), ("s151", "4x"), ("s55", "2x"), ("s55", "4x"))
MINIMUM_IDENTIFIED_PER_SEED_CELL = 200


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_terminal_integrity(
    path: Path,
    *,
    manifest_path: Path = stage_a.MANIFEST_PATH,
) -> dict[str, Any]:
    """Verify the outcome-blind terminal artifact before opening raw outcomes."""

    payload = json.loads(path.read_text())
    source = stage_a.source_state()
    audit = payload.get("audit", {})
    runs = payload.get("runs", [])
    _require(
        payload.get("schema_version") == 1
        and payload.get("experiment_id") == integrity.INTEGRITY_EXPERIMENT_ID
        and payload.get("status") == "terminal_pass"
        and payload.get("terminal") is True
        and payload.get("stage") == "A",
        "A terminal passing Stage-A integrity artifact is required.",
    )
    _require(
        payload.get("manifest")
        == {"path": str(manifest_path), "sha256": stage_a.sha256(manifest_path)},
        "Stage-A integrity manifest binding drifted.",
    )
    _require(source.get("dirty") is False, "Stage-A quality summary requires clean source.")
    _require(
        payload.get("source", {}).get("dirty") is False
        and payload.get("source", {}).get("implementation_digest")
        == source.get("implementation_digest"),
        "Stage-A integrity implementation drifted.",
    )
    required_audit = {
        "all_required_coordinates_present": True,
        "all_raw_artifact_digests_verified": True,
        "all_dependency_digests_verified": True,
        "prospective_3600_run_sequential_tiered_integrity_verified": True,
        "all_exact_quota_multiset_and_total_controls_verified": True,
        "all_pin_set_and_fallback_controls_verified": True,
        "paired_measured_hot_memory_exactly_equal": True,
        "protected_pin_exposure_verified_by_family_and_config": True,
        "no_budget_violations": True,
        "all_uniform_mapping_exclusions_predeclared_and_sufficient": True,
        "post_core_807_input_reuse_disclosed": True,
        "outcomes_inspected": False,
        "accuracy_computed": False,
        "paired_effect_computed": False,
    }
    _require(audit == required_audit, "Stage-A outcome-blind integrity checks drifted.")
    measured_memory = payload.get("paired_measured_hot_memory", {})
    _require(
        measured_memory.get("expected_batch_pairs")
        == stage_a.EXPECTED_COORDINATES * stage_a.BATCHES_PER_SHARD
        and measured_memory.get("observed_batch_pairs")
        == stage_a.EXPECTED_COORDINATES * stage_a.BATCHES_PER_SHARD
        and measured_memory.get("tier_hot_byte_mismatches") == 0
        and measured_memory.get("tier_hot_block_mismatches") == 0
        and measured_memory.get("hot_resident_byte_mismatches") == 0
        and measured_memory.get("exact_equality_required_before_outcome_access") is True
        and measured_memory.get("passed") is True,
        "Stage-A measured hot-memory audit drifted.",
    )
    pin_exposure = payload.get("pin_exposure_by_family", [])
    _require(
        isinstance(pin_exposure, list)
        and len(pin_exposure) == len(stage_a.PAPER_GRADE_WORKLOAD_FAMILIES)
        and {row.get("family") for row in pin_exposure}
        == set(stage_a.PAPER_GRADE_WORKLOAD_FAMILIES)
        and all(row.get("passed") is True for row in pin_exposure),
        "Stage-A protected-pin exposure audit drifted.",
    )
    pin_config_exposure = payload.get("pin_exposure_by_family_arm_config", [])
    _require(
        isinstance(pin_config_exposure, list)
        and bool(pin_config_exposure)
        and all(row.get("passed") is True for row in pin_config_exposure),
        "Stage-A family/arm/config pin-exposure audit drifted.",
    )
    _require(
        payload.get("expected_shards") == stage_a.EXPECTED_COORDINATES
        and payload.get("observed_shards") == stage_a.EXPECTED_COORDINATES
        and payload.get("expected_arm_conversations") == stage_a.EXPECTED_STAGE_A_CONVERSATIONS
        and payload.get("coordinate_digest") == stage_a.coordinate_digest()
        and isinstance(runs, list)
        and len(runs) == stage_a.EXPECTED_COORDINATES
        and payload.get("raw_matrix_digest") == stage_a.json_digest(runs),
        "Stage-A terminal integrity coverage drifted.",
    )
    expected_coordinates = set(stage_a.expected_coordinates())
    seen_coordinates = {
        (
            run.get("scale"),
            run.get("training_seed"),
            run.get("budget"),
            run.get("family"),
            run.get("context"),
            run.get("replicate"),
        )
        for run in runs
    }
    _require(seen_coordinates == expected_coordinates, "Integrity coordinate set drifted.")
    for run in runs:
        metadata = run.get("raw_artifact", {})
        raw_path = Path(str(metadata.get("path", "")))
        _require(
            raw_path.is_file() and metadata.get("sha256") == stage_a.sha256(raw_path),
            f"Stage-A raw shard drifted after integrity: {raw_path}",
        )
    return payload


def _numeric_token_rows(value: object, *, label: str) -> list[list[int]]:
    _require(
        isinstance(value, list) and bool(value),
        f"{label} is not a non-empty row list.",
    )
    assert isinstance(value, list)
    rows: list[list[int]] = []
    for row in value:
        _require(isinstance(row, list) and bool(row), f"{label} contains an empty row.")
        assert isinstance(row, list)
        _require(
            all(type(token) is int and token >= 0 for token in row),
            f"{label} contains an invalid token.",
        )
        rows.append(cast(list[int], row))
    return rows


def extract_paired_differences(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Read outcomes only after require_terminal_integrity has returned."""

    if raw.get("arm_contract", {}).get("mapping_identified") is not True:
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in raw["batches"]:
        workload = batch["workload"]
        conversation_ids = workload["conversation_ids"]
        targets = _numeric_token_rows(workload["targets"]["values"], label="targets")
        query_positions = _numeric_token_rows(
            workload["query_positions"]["values"], label="query positions"
        )
        candidate_predictions = _numeric_token_rows(
            batch["arm_runs"][stage_a.ARMS[0]]["predictions"],
            label="candidate predictions",
        )
        comparator_predictions = _numeric_token_rows(
            batch["arm_runs"][stage_a.ARMS[1]]["predictions"],
            label="comparator predictions",
        )
        _require(
            len(conversation_ids)
            == len(targets)
            == len(query_positions)
            == len(candidate_predictions)
            == len(comparator_predictions)
            == stage_a.BATCH_SIZE,
            "Stage-A paired row count drifted.",
        )
        for index, conversation_id in enumerate(conversation_ids):
            _require(
                isinstance(conversation_id, str) and conversation_id not in seen,
                "Duplicate Stage-A paired conversation.",
            )
            seen.add(conversation_id)
            target = targets[index]
            candidate = candidate_predictions[index]
            comparator = comparator_predictions[index]
            positions = query_positions[index]
            _require(
                len(target) == len(candidate) == len(comparator) == len(positions) > 0,
                "Stage-A target/prediction/query-position mismatch.",
            )
            candidate_accuracy = sum(
                prediction == expected
                for prediction, expected in zip(candidate, target, strict=True)
            ) / len(target)
            comparator_accuracy = sum(
                prediction == expected
                for prediction, expected in zip(comparator, target, strict=True)
            ) / len(target)
            difference = candidate_accuracy - comparator_accuracy
            _require(
                all(
                    math.isfinite(value)
                    for value in (candidate_accuracy, comparator_accuracy, difference)
                ),
                "Stage-A conversation outcome is non-finite.",
            )
            result.append(
                {
                    "scale": raw["scale"],
                    "budget": raw["budget"],
                    "training_seed": raw["training_seed"],
                    "family": raw["family"],
                    "context": raw["context"],
                    "conversation_id": conversation_id,
                    "candidate_accuracy": candidate_accuracy,
                    "comparator_accuracy": comparator_accuracy,
                    "paired_difference": difference,
                    "registered_queries": len(target),
                }
            )
    _require(
        len(result) == stage_a.EXAMPLES_PER_SHARD,
        "Stage-A shard paired-conversation count drifted.",
    )
    return result


def _group_means(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    result: list[dict[str, Any]] = []
    for coordinate, rows in sorted(groups.items()):
        result.append(
            {
                **dict(zip(keys, coordinate, strict=True)),
                "eligible_paired_conversations": len(rows),
                "candidate_mean_accuracy": sum(row["candidate_accuracy"] for row in rows)
                / len(rows),
                "comparator_mean_accuracy": sum(row["comparator_accuracy"] for row in rows)
                / len(rows),
                "mean_difference": sum(row["paired_difference"] for row in rows) / len(rows),
            }
        )
    return result


def exact_seed_sign_flip(seed_effects: list[float]) -> float:
    _require(len(seed_effects) == 5, "The exact sign-flip audit requires five seeds.")
    observed = abs(sum(seed_effects) / len(seed_effects))
    tolerance = 16 * np.finfo(np.float64).eps
    assignments = itertools.product((-1.0, 1.0), repeat=len(seed_effects))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, seed_effects, strict=True)) / 5)
        >= observed - tolerance
        for signs in assignments
    )
    return extreme / 32.0


def cell_statistics(
    by_seed: list[dict[str, Any]],
    *,
    scale: str,
    budget: str,
    cell_offset: int,
) -> dict[str, Any]:
    rows = sorted(
        (row for row in by_seed if row["scale"] == scale and row["budget"] == budget),
        key=lambda row: row["training_seed"],
    )
    _require(
        [row["training_seed"] for row in rows] == list(stage_a.TRAINING_SEEDS),
        f"Stage-A cell {scale}/{budget} is missing a frozen seed.",
    )
    _require(
        all(
            row["eligible_paired_conversations"] >= MINIMUM_IDENTIFIED_PER_SEED_CELL for row in rows
        ),
        f"Stage-A cell {scale}/{budget} lacks identified conversations.",
    )
    effects = np.asarray([row["mean_difference"] for row in rows], dtype=np.float64)
    _require(bool(np.isfinite(effects).all()), "Stage-A seed effects are non-finite.")
    rng = np.random.default_rng(BOOTSTRAP_BASE_SEED + cell_offset)
    indices = rng.integers(0, len(effects), size=(BOOTSTRAP_RESAMPLES, len(effects)))
    distribution = effects[indices].mean(axis=1)
    interval = np.quantile(distribution, [0.025, 0.975], method="linear")
    pooled = float(effects.mean())
    positive_seeds = int((effects > 0.0).sum())
    lower = float(interval[0])
    return {
        "scale": scale,
        "budget": budget,
        "independent_training_seed_clusters": len(effects),
        "ordered_training_seeds": list(stage_a.TRAINING_SEEDS),
        "seed_effects": effects.tolist(),
        "eligible_paired_conversations": sum(row["eligible_paired_conversations"] for row in rows),
        "pooled_effect": pooled,
        "positive_seed_effects": positive_seeds,
        "bootstrap": {
            "algorithm": "numpy.default_rng integer-index seed-cluster bootstrap",
            "seed": BOOTSTRAP_BASE_SEED + cell_offset,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "quantile_method": "linear",
            "confidence_interval": [lower, float(interval[1])],
        },
        "exact_seed_sign_flip": {
            "assignments": 32,
            "two_sided_p_value": exact_seed_sign_flip(effects.tolist()),
            "used_as_gate": False,
        },
        "gate_checks": {
            "pooled_effect_strictly_positive": pooled > 0.0,
            "at_least_four_of_five_seed_effects_strictly_positive": positive_seeds >= 4,
            "descriptive_95pct_seed_cluster_lower_bound_nonnegative": lower >= 0.0,
        },
    }


def summarize(
    integrity_payload: dict[str, Any],
    *,
    integrity_path: Path,
    manifest_path: Path = stage_a.MANIFEST_PATH,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    excluded_uniform_mappings: list[dict[str, Any]] = []
    implementation_digest = cast(str, integrity_payload["source"]["implementation_digest"])
    for run in integrity_payload["runs"]:
        coordinate = (
            run["scale"],
            run["training_seed"],
            run["budget"],
            run["family"],
            run["context"],
            run["replicate"],
        )
        raw_path = Path(run["raw_artifact"]["path"])
        raw = json.loads(raw_path.read_text())
        metadata = integrity.validate_raw_shard(
            raw,
            coordinate=coordinate,
            implementation_digest=implementation_digest,
            manifest_path=manifest_path,
        )
        if metadata["mapping_identified"]:
            records.extend(extract_paired_differences(raw))
        else:
            excluded_uniform_mappings.append(
                {
                    "scale": run["scale"],
                    "training_seed": run["training_seed"],
                    "budget": run["budget"],
                    "family": run["family"],
                    "context": run["context"],
                    "paired_conversations_retained_in_raw_but_excluded": stage_a.EXAMPLES_PER_SHARD,
                }
            )
    by_seed_family_context = _group_means(
        records,
        ("scale", "budget", "training_seed", "family", "context"),
    )
    by_seed_family = _group_means(records, ("scale", "budget", "training_seed", "family"))
    by_seed_context = _group_means(records, ("scale", "budget", "training_seed", "context"))
    by_seed = _group_means(records, ("scale", "budget", "training_seed"))
    expected_seed_cells = len(stage_a.SCALES) * len(stage_a.BUDGETS) * len(stage_a.TRAINING_SEEDS)
    _require(len(by_seed) == expected_seed_cells, "Stage-A seed-cell coverage drifted.")
    _require(
        all(
            row["eligible_paired_conversations"] >= MINIMUM_IDENTIFIED_PER_SEED_CELL
            for row in by_seed
        ),
        "Stage-A uniform-mapping exclusion rule blocks quality inference.",
    )
    cells = [
        cell_statistics(by_seed, scale=scale, budget=budget, cell_offset=offset)
        for offset, (scale, budget) in enumerate(CELL_ORDER)
    ]
    stage_a_go = all(all(cell["gate_checks"].values()) for cell in cells)
    source = stage_a.source_state()
    _require(
        source.get("dirty") is False
        and source.get("implementation_digest") == implementation_digest,
        "Stage-A source changed during outcome summarization.",
    )
    for run in integrity_payload["runs"]:
        raw_metadata = run["raw_artifact"]
        _require(
            stage_a.sha256(Path(raw_metadata["path"])) == raw_metadata["sha256"],
            "A Stage-A raw shard changed during outcome summarization.",
        )
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal_exploratory_go" if stage_a_go else "terminal_exploratory_no_go",
        "stage": "A",
        "contrast": "calibrated+pins - shuffled-quota+pins",
        "estimand": (
            "Effect of assigning the identical non-uniform quota multiset to calibrated "
            "layer identities rather than the frozen digest-derived cyclic shuffle."
        ),
        "manifest": {"path": str(manifest_path), "sha256": stage_a.sha256(manifest_path)},
        "terminal_integrity_artifact": {
            "path": str(integrity_path),
            "sha256": stage_a.sha256(integrity_path),
            "experiment_id": integrity.INTEGRITY_EXPERIMENT_ID,
            "outcomes_inspected_before_terminal_integrity": False,
        },
        "source": source,
        "execution_corner": "sequential-tiered",
        "data_reuse_disclosure": {
            "reused_post_core_807_series_inputs": True,
            "interpretation": "exploratory decision panel, not fresh confirmatory evidence",
        },
        "protected_pin_exposure_audit": {
            "by_family": integrity_payload["pin_exposure_by_family"],
            "by_family_arm_config": integrity_payload["pin_exposure_by_family_arm_config"],
            "interpretation": (
                "Protected-pin controls were materially exercised where the frozen workload "
                "defines protected positions; this remains a layer-identity estimand and is "
                "not a protected-pin effect claim."
            ),
        },
        "paired_measured_hot_memory_audit": integrity_payload["paired_measured_hot_memory"],
        "coverage": {
            "raw_shards": stage_a.EXPECTED_COORDINATES,
            "raw_arm_conversations": stage_a.EXPECTED_STAGE_A_CONVERSATIONS,
            "eligible_paired_conversations": len(records),
            "excluded_uniform_mapping_coordinates": len(excluded_uniform_mappings),
            "excluded_uniform_mapping_arm_conversations": (
                len(excluded_uniform_mappings) * stage_a.EXAMPLES_PER_SHARD * len(stage_a.ARMS)
            ),
        },
        "uniform_mapping_exclusions": excluded_uniform_mappings,
        "statistics_contract": {
            "paired_unit": "complete conversation",
            "conversation_outcome": (
                "correct registered query predictions / total registered query positions"
            ),
            "cell_seed_effect": (
                "unweighted mean of eligible paired conversation differences across all "
                "nine families, five contexts, and twenty conversations"
            ),
            "pooled_cell_effect": "unweighted arithmetic mean of five seed effects",
            "independent_cluster": "training checkpoint seed",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
            "bootstrap_cell_order": [list(cell) for cell in CELL_ORDER],
            "bootstrap_quantile_method": "linear",
            "exact_seed_sign_flip_assignments": 32,
            "multiplicity": "descriptive only; no confirmatory multiplicity claim",
        },
        "statistics": {
            "cells": cells,
            "by_seed": by_seed,
            "by_seed_family": by_seed_family,
            "by_seed_context": by_seed_context,
            "by_seed_family_context": by_seed_family_context,
        },
        "preliminary_decision": {
            "stage_a_clean_layer_identity": "GO" if stage_a_go else "NO_GO",
            "passed": stage_a_go,
            "required_cells": 4,
            "all_cells_passed": stage_a_go,
            "stage_b_execution_permitted": stage_a_go,
            "stage_b_fixed_memory_match_prerequisites_permitted": stage_a_go,
            "full_16_arm_factorial_automatically_resumed": False,
        },
        "nonclaims": [
            "not confirmatory inference or a paper-grade causal effect estimate",
            "not a protected-pin effect estimate",
            "not a replacement for the paused 16-arm causal factorial",
            "not natural-language, model-family, long-context, or production generalization",
            "not evidence for latency, throughput, HBM, transfer, or cost benefit",
            "not evidence for HCA conditional-information hypotheses",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score the Stage-A clean layer-identity panel only after a terminal "
            "outcome-blind integrity audit."
        )
    )
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=stage_a.MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Stage-A quality summary already exists: {args.output}")
    integrity_payload = require_terminal_integrity(args.integrity, manifest_path=args.manifest)
    payload = summarize(
        integrity_payload,
        integrity_path=args.integrity,
        manifest_path=args.manifest,
    )
    stage_a.write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "stage_a_decision": payload["preliminary_decision"]["stage_a_clean_layer_identity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
