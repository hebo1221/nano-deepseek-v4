from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any, cast

import evaluate_p2_causal_factorial_shard as shard
import numpy as np
from summarize_p2_core_matrix import (
    bootstrap_paired_mean,
    holm_bonferroni,
    seed_cluster_statistics,
)

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

EXPECTED_SHARDS = 9_000
PRIMARY_CANDIDATE = "calibrated+pins"
PRIMARY_COMPARATOR = "fixed+pins"
PRIMARY_CELL_CONFIDENCE = 1.0 - 0.05 / 4.0
MAXIMUM_MEMORY_DIFFERENCE = 0.01
EXPECTED_PHYSICAL_BATCHES_PER_SEED_CELL = (
    len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(shard.CONTEXTS)
    * len(shard.REPLICATES)
    * shard.BATCHES_PER_SHARD
)
CONTRASTS = {
    "adaptive_quota_with_pins": ("calibrated+pins", "fixed+pins"),
    "adaptive_quota_without_pins": ("calibrated-no-pins", "fixed"),
    "pins_under_fixed_quota": ("fixed+pins", "fixed"),
    "pins_under_calibrated_quota": ("calibrated+pins", "calibrated-no-pins"),
    "layer_identity_with_pins": ("calibrated+pins", "shuffled-quota+pins"),
    "layer_identity_without_pins": ("calibrated-no-pins", "shuffled-quota"),
    "local_adaptation_with_pins": ("local+pins", "calibrated+pins"),
    "hierarchy_beyond_local": ("hierarchical+pins", "local+pins"),
    "score_concentration": ("hierarchical+pins", "hierarchical+pins-no-score"),
    "temporal_reuse": ("hierarchical+pins", "hierarchical+pins-no-temporal"),
    "refresh_reuse": ("hierarchical+pins", "hierarchical+pins-no-refresh"),
    "dense_fallback": ("hierarchical+pins+fallback", "hierarchical+pins"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_exact_config_reuse(
    rows: list[dict[str, Any]], *, expected_arms: tuple[str, ...]
) -> dict[str, int]:
    """Audit executed/reused rows without treating reuse as a new forward."""

    by_batch_arm = {
        (row.get("schedule_batch_index"), row.get("arm")): row for row in rows
    }
    _require(len(by_batch_arm) == len(rows), "Duplicate causal execution-accounting row.")
    batches = {key[0] for key in by_batch_arm}
    _require(
        all(
            {arm for batch, arm in by_batch_arm if batch == batch_index}
            == set(expected_arms)
            for batch_index in batches
        ),
        "Causal execution-accounting arm coverage drifted.",
    )
    counts = {"executed": 0, "reused_exact_config": 0}
    for (batch_index, arm), row in by_batch_arm.items():
        digest = row.get("config_sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "Causal execution config digest drifted.",
        )
        mode = row.get("execution_mode")
        reused_from = row.get("reused_from_arm")
        wall_ms = row.get("wall_ms")
        _require(
            isinstance(wall_ms, (int, float)) and not isinstance(wall_ms, bool),
            "Causal execution wall time drifted.",
        )
        assert isinstance(wall_ms, (int, float)) and not isinstance(wall_ms, bool)
        if mode == "executed":
            _require(reused_from is None and wall_ms > 0, "Executed causal row drifted.")
            counts["executed"] += 1
            continue
        _require(
            mode == "reused-exact-config"
            and isinstance(reused_from, str)
            and reused_from != arm
            and wall_ms == 0.0,
            "Reused causal row drifted.",
        )
        source = by_batch_arm.get((batch_index, reused_from))
        source_index = source.get("execution_index") if source is not None else None
        reuse_index = row.get("execution_index")
        _require(
            type(source_index) is int and type(reuse_index) is int,
            "Causal exact-config reuse order drifted.",
        )
        assert isinstance(source_index, int) and isinstance(reuse_index, int)
        _require(
            source is not None
            and source.get("execution_mode") == "executed"
            and source.get("config_sha256") == digest
            and source_index < reuse_index,
            "Causal exact-config reuse source drifted.",
        )
        counts["reused_exact_config"] += 1
    return counts


def _verify_dependency(
    metadata: dict[str, Any],
    name: str,
    verified: set[tuple[str, str, str]] | None = None,
) -> Path:
    path = Path(metadata.get("path", ""))
    expected = metadata.get("sha256")
    _require(path.is_file(), f"Missing {name}: {path}")
    _require(isinstance(expected, str), f"Missing {name} digest: {path}")
    identity = (name, str(path), cast(str, expected))
    if verified is None or identity not in verified:
        _require(expected == sha256(path), f"{name} digest drifted: {path}")
        if verified is not None:
            verified.add(identity)
    return path


def _merge(groups: Iterable[list[float]]) -> list[float]:
    result: list[float] = []
    for group in groups:
        result.extend(group)
    return result


def contrast_statistics(
    differences: dict[tuple[str, str, int, str, int], list[float]],
    *,
    name: str,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    for scale in ("s55", "s151"):
        for budget in shard.BUDGET_LABELS:
            values = _merge(
                group
                for (item_scale, item_budget, _seed, _family, _context), group in differences.items()
                if item_scale == scale and item_budget == budget
            )
            cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    **bootstrap_paired_mean(values, label=f"causal:{name}:cell:{scale}:{budget}"),
                }
            )
            cell_seed_means: list[float] = []
            for training_seed in shard.TRAINING_SEEDS:
                seed_values = _merge(
                    group
                    for (
                        item_scale,
                        item_budget,
                        item_seed,
                        _family,
                        _context,
                    ), group in differences.items()
                    if item_scale == scale
                    and item_budget == budget
                    and item_seed == training_seed
                )
                seeds.append(
                    {
                        "scale": scale,
                        "budget": budget,
                        "training_seed": training_seed,
                        "paired_units": len(seed_values),
                        "mean_difference": float(np.mean(seed_values)),
                        "mean_difference_percentage_points": float(
                            np.mean(seed_values) * 100.0
                        ),
                    }
                )
                cell_seed_means.append(float(np.mean(seed_values)))
            cells[-1]["seed_cluster_inference"] = seed_cluster_statistics(
                cell_seed_means,
                label=f"causal:{name}:seed-cluster:{scale}:{budget}",
            )
            cell_families: list[dict[str, Any]] = []
            for family in PAPER_GRADE_WORKLOAD_FAMILIES:
                family_values = _merge(
                    group
                    for (
                        item_scale,
                        item_budget,
                        _seed,
                        item_family,
                        _context,
                    ), group in differences.items()
                    if item_scale == scale
                    and item_budget == budget
                    and item_family == family
                )
                row = {
                    "scale": scale,
                    "budget": budget,
                    "family": family,
                    **bootstrap_paired_mean(
                        family_values,
                        label=f"causal:{name}:family:{scale}:{budget}:{family}",
                    ),
                }
                family_seed_means = [
                    float(
                        np.mean(
                            _merge(
                                group
                                for (
                                    item_scale,
                                    item_budget,
                                    item_seed,
                                    item_family,
                                    _context,
                                ), group in differences.items()
                                if item_scale == scale
                                and item_budget == budget
                                and item_seed == training_seed
                                and item_family == family
                            )
                        )
                    )
                    for training_seed in shard.TRAINING_SEEDS
                ]
                row["seed_cluster_inference"] = seed_cluster_statistics(
                    family_seed_means,
                    label=f"causal:{name}:family-seeds:{scale}:{budget}:{family}",
                )
                cell_families.append(row)
            adjusted = holm_bonferroni(
                {
                    row["family"]: row["seed_cluster_inference"][
                        "paired_randomization_two_sided_p"
                    ]
                    for row in cell_families
                }
            )
            for row in cell_families:
                row["holm_adjusted_p"] = adjusted[row["family"]]
            families.extend(cell_families)
            for family in PAPER_GRADE_WORKLOAD_FAMILIES:
                for context in shard.CONTEXTS:
                    context_values = _merge(
                        group
                        for (
                            item_scale,
                            item_budget,
                            _seed,
                            item_family,
                            item_context,
                        ), group in differences.items()
                        if item_scale == scale
                        and item_budget == budget
                        and item_family == family
                        and item_context == context
                    )
                    slices.append(
                        {
                            "scale": scale,
                            "budget": budget,
                            "family": family,
                            "context": context,
                            "paired_units": len(context_values),
                            "mean_difference": float(np.mean(context_values)),
                            "mean_difference_percentage_points": float(
                                np.mean(context_values) * 100.0
                            ),
                        }
                    )
    return {
        "name": name,
        "candidate": candidate,
        "comparator": comparator,
        "cells": cells,
        "by_seed": seeds,
        "by_family_with_holm_bonferroni": families,
        "by_family_context": slices,
        "worst_slice": min(slices, key=lambda row: row["mean_difference"]),
    }


def physical_memory_statistics(
    values: dict[tuple[str, str, int, str], list[int]],
) -> dict[str, Any]:
    seed_cells: list[dict[str, Any]] = []
    aggregate_cells: list[dict[str, Any]] = []
    for scale in ("s55", "s151"):
        for budget in shard.BUDGET_LABELS:
            for training_seed in shard.TRAINING_SEEDS:
                fixed = values[(scale, budget, training_seed, PRIMARY_COMPARATOR)]
                calibrated = values[(scale, budget, training_seed, PRIMARY_CANDIDATE)]
                _require(
                    len(fixed) == len(calibrated) == EXPECTED_PHYSICAL_BATCHES_PER_SEED_CELL,
                    "Physical held-out batch coverage drifted.",
                )
                fixed_mean = float(np.mean(fixed))
                calibrated_mean = float(np.mean(calibrated))
                relative = abs(calibrated_mean - fixed_mean) / max(calibrated_mean, 1.0)
                seed_cells.append(
                    {
                        "scale": scale,
                        "budget": budget,
                        "training_seed": training_seed,
                        "physical_batches_per_arm": len(fixed),
                        "fixed_mean_hot_resident_bytes": fixed_mean,
                        "calibrated_mean_hot_resident_bytes": calibrated_mean,
                        "relative_difference": relative,
                        "within_one_percent": relative <= MAXIMUM_MEMORY_DIFFERENCE,
                    }
                )
            fixed_all = _merge(
                [float(value) for value in values[(scale, budget, seed, PRIMARY_COMPARATOR)]]
                for seed in shard.TRAINING_SEEDS
            )
            calibrated_all = _merge(
                [float(value) for value in values[(scale, budget, seed, PRIMARY_CANDIDATE)]]
                for seed in shard.TRAINING_SEEDS
            )
            fixed_mean = float(np.mean(fixed_all))
            calibrated_mean = float(np.mean(calibrated_all))
            aggregate_cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    "physical_batches_per_arm": len(fixed_all),
                    "fixed_mean_hot_resident_bytes": fixed_mean,
                    "calibrated_mean_hot_resident_bytes": calibrated_mean,
                    "relative_difference": abs(calibrated_mean - fixed_mean)
                    / max(calibrated_mean, 1.0),
                    "all_seed_cells_within_one_percent": all(
                        row["within_one_percent"]
                        for row in seed_cells
                        if row["scale"] == scale and row["budget"] == budget
                    ),
                }
            )
    return {"by_seed": seed_cells, "aggregate": aggregate_cells}


def primary_causal_gate(
    statistics: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for scale in ("s55", "s151"):
        for budget in shard.BUDGET_LABELS:
            quality = next(
                row
                for row in statistics["cells"]
                if row["scale"] == scale and row["budget"] == budget
            )
            seed_rows = [
                row
                for row in statistics["by_seed"]
                if row["scale"] == scale and row["budget"] == budget
            ]
            corrected = quality["four_cell_corrected_bootstrap"]
            memory_row = next(
                row
                for row in memory["aggregate"]
                if row["scale"] == scale and row["budget"] == budget
            )
            cell = {
                "scale": scale,
                "budget": budget,
                "pooled_effect_positive": quality["mean_difference"] > 0.0,
                "four_cell_corrected_lower_bound": corrected["confidence_interval"][0],
                "four_cell_corrected_lower_bound_positive": corrected[
                    "confidence_interval"
                ][0]
                > 0.0,
                "positive_seed_effects": sum(
                    row["mean_difference"] > 0.0 for row in seed_rows
                ),
                "required_seed_effects": len(shard.TRAINING_SEEDS),
                "all_seed_effects_positive": all(
                    row["mean_difference"] > 0.0 for row in seed_rows
                ),
                "memory_match_relative_difference": memory_row["relative_difference"],
                "all_seed_memory_cells_within_one_percent": memory_row[
                    "all_seed_cells_within_one_percent"
                ],
            }
            cell["passed"] = all(
                (
                    cell["pooled_effect_positive"],
                    cell["four_cell_corrected_lower_bound_positive"],
                    cell["all_seed_effects_positive"],
                    cell["all_seed_memory_cells_within_one_percent"],
                )
            )
            cells.append(cell)
    return {
        "candidate": PRIMARY_CANDIDATE,
        "comparator": PRIMARY_COMPARATOR,
        "scales": ["s55", "s151"],
        "budgets": list(shard.BUDGET_LABELS),
        "seeds_per_scale": len(shard.TRAINING_SEEDS),
        "required_cells": len(cells),
        "cells": cells,
        "passed": all(cell["passed"] for cell in cells),
        "failure_interpretation": "failed-or-bounded causal claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the full P2 causal factorial matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    _require(
        matrix.get("experiment_id") == "p2-causal-factorial-matrix-progress-v1",
        "Wrong causal matrix id.",
    )
    _require(matrix.get("completed_shards") == EXPECTED_SHARDS, "Causal matrix is incomplete.")
    design = matrix.get("frozen_design", {})
    _require(design.get("total_expected_shards") == EXPECTED_SHARDS, "Shard design drifted.")
    _require(tuple(design.get("scales", ())) == ("s55", "s151"), "Scale drifted.")
    _require(tuple(design.get("training_seeds", ())) == shard.TRAINING_SEEDS, "Seed drifted.")
    _require(tuple(design.get("budgets", ())) == shard.BUDGET_LABELS, "Budget drifted.")
    _require(
        tuple(design.get("families", ())) == PAPER_GRADE_WORKLOAD_FAMILIES,
        "Family drifted.",
    )
    _require(tuple(design.get("contexts", ())) == shard.CONTEXTS, "Context drifted.")
    _require(tuple(design.get("replicates", ())) == shard.REPLICATES, "Replicate drifted.")
    runs = matrix.get("runs")
    _require(isinstance(runs, list) and len(runs) == EXPECTED_SHARDS, "Run count drifted.")
    implementation_digest = matrix.get("implementation_digest")
    _require(
        implementation_digest == shard.implementation_digest(),
        "Causal implementation is not the checked-out implementation.",
    )
    expected = set(
        product(
            ("s55", "s151"),
            shard.TRAINING_SEEDS,
            shard.BUDGET_LABELS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
            shard.REPLICATES,
        )
    )
    seen: set[tuple[str, int, str, str, int, int]] = set()
    differences: dict[
        str, dict[tuple[str, str, int, str, int], list[float]]
    ] = {name: defaultdict(list) for name in CONTRASTS}
    physical: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    raw_digests: list[str] = []
    verified_dependencies: set[tuple[str, str, str]] = set()
    validated_memory_matches: set[Path] = set()
    validated_equivalences: set[Path] = set()
    quality_execution_counts = {"executed": 0, "reused_exact_config": 0}
    physical_execution_counts = {"executed": 0, "reused_exact_config": 0}
    for run in runs:
        identity = (
            run["scale"],
            run["training_seed"],
            run["budget"],
            run["family"],
            run["context"],
            run["replicate"],
        )
        _require(identity not in seen, f"Duplicate causal shard: {identity}")
        seen.add(identity)
        raw_path = _verify_dependency(run["raw_artifact"], "causal shard")
        raw_digests.append(run["raw_artifact"]["sha256"])
        raw = json.loads(raw_path.read_text())
        _require(raw.get("experiment_id") == "p2-causal-factorial-shard-v1", "Wrong shard id.")
        _require(raw.get("source", {}).get("dirty") is False, "Dirty causal shard source.")
        _require(
            raw.get("source", {}).get("implementation_digest") == implementation_digest,
            "Causal implementation digest drifted.",
        )
        for key in ("scale", "training_seed", "budget", "family", "context", "replicate"):
            _require(raw.get(key) == run.get(key), f"Causal run metadata drifted: {key}")
        records_raw = raw.get("records")
        _require(isinstance(records_raw, list), "Causal records are not a list.")
        records = cast(list[dict[str, Any]], records_raw)
        _require(
            len(records) == shard.EXAMPLES_PER_SHARD * len(shard.ALL_ARM_NAMES),
            "Causal record count drifted.",
        )
        _require(records_digest(records) == raw.get("records_digest"), "Record digest drifted.")
        checkpoint_path = _verify_dependency(
            raw["checkpoint"], "checkpoint", verified_dependencies
        )
        del checkpoint_path
        calibration_path = _verify_dependency(
            raw["calibration_artifact"], "calibration artifact", verified_dependencies
        )
        memory_match_path = _verify_dependency(
            raw["memory_match_artifact"], "memory-match artifact", verified_dependencies
        )
        equivalence_path = _verify_dependency(
            raw["equivalence_artifact"], "equivalence artifact", verified_dependencies
        )
        _verify_dependency(raw["design_manifest"], "causal design", verified_dependencies)
        if memory_match_path not in validated_memory_matches:
            shard._memory_match(
                memory_match_path,
                scale=raw["scale"],
                training_seed=raw["training_seed"],
                calibration_path=calibration_path,
            )
            validated_memory_matches.add(memory_match_path)
        if equivalence_path not in validated_equivalences:
            shard._equivalence(
                equivalence_path,
                raw["scale"],
                training_seed=raw["training_seed"],
            )
            validated_equivalences.add(equivalence_path)
        batch_metrics = raw["batch_metrics"]
        _require(
            len(batch_metrics)
            == shard.BATCHES_PER_SHARD * len(shard.ALL_ARM_NAMES),
            "Causal batch metric coverage drifted.",
        )
        _require(
            all(metric.get("budget_violations") == 0 for metric in batch_metrics),
            "Causal shard contains a budget violation.",
        )
        shard_quality_counts = validate_exact_config_reuse(
            batch_metrics, expected_arms=shard.ALL_ARM_NAMES
        )
        for key, value in shard_quality_counts.items():
            quality_execution_counts[key] += value
        quality_by_schedule_arm = {
            (metric["schedule_batch_index"], metric["arm"]): metric
            for metric in batch_metrics
        }
        _require(
            raw.get("arm_metadata", {}).get("fixed_match_source")
            == "calibration-physical-hot-bytes",
            "Causal shard did not use its frozen physical-memory match.",
        )
        by_arm_conversation = {
            (record["arm"], record["conversation_id"]): record for record in records
        }
        _require(
            len(by_arm_conversation) == len(records),
            "Duplicate causal arm-conversation record.",
        )
        conversation_ids = sorted(
            record["conversation_id"] for record in records if record["arm"] == PRIMARY_COMPARATOR
        )
        _require(
            len(conversation_ids) == shard.EXAMPLES_PER_SHARD,
            "Causal conversation coverage drifted.",
        )
        for record in records:
            metric = quality_by_schedule_arm[
                (record["schedule_batch_index"], record["arm"])
            ]
            _require(
                record.get("execution_mode") == metric.get("execution_mode")
                and record.get("reused_from_arm") == metric.get("reused_from_arm")
                and record.get("config_sha256") == metric.get("config_sha256"),
                "Causal record execution provenance drifted.",
            )
            predictions = record["predictions"]
            targets = record["targets"]
            correctness = [
                prediction == target
                for prediction, target in zip(predictions, targets, strict=True)
            ]
            _require(correctness == record["correct"], "Causal correctness drifted.")
            _require(sum(correctness) == record["correct_count"], "Correct count drifted.")
            _require(len(correctness) == record["total"], "Query total drifted.")
        difference_key = (
            raw["scale"],
            raw["budget"],
            raw["training_seed"],
            raw["family"],
            raw["context"],
        )
        for conversation_id in conversation_ids:
            reference = by_arm_conversation[(PRIMARY_COMPARATOR, conversation_id)]
            for name, (candidate_name, comparator_name) in CONTRASTS.items():
                candidate = by_arm_conversation[(candidate_name, conversation_id)]
                comparator = by_arm_conversation[(comparator_name, conversation_id)]
                for field in ("targets", "query_positions", "evidence_positions"):
                    _require(
                        candidate[field] == comparator[field] == reference[field],
                        f"Paired causal {field} drifted.",
                    )
                _require(candidate["total"] == comparator["total"], "Query count drifted.")
                differences[name][difference_key].append(
                    candidate["correct_count"] / candidate["total"]
                    - comparator["correct_count"] / comparator["total"]
                )
        measurements = raw.get("physical_measurements", [])
        _require(
            len(measurements) == shard.BATCHES_PER_SHARD * len(shard.PHYSICAL_ARM_NAMES),
            "Physical measurement count drifted.",
        )
        shard_physical_counts = validate_exact_config_reuse(
            measurements, expected_arms=shard.PHYSICAL_ARM_NAMES
        )
        for key, value in shard_physical_counts.items():
            physical_execution_counts[key] += value
        by_batch_arm = {
            (measurement["batch_index"], measurement["arm"]): measurement
            for measurement in measurements
        }
        _require(len(by_batch_arm) == len(measurements), "Duplicate physical measurement.")
        for batch_index in range(shard.BATCHES_PER_SHARD):
            fixed_batch = by_batch_arm[(batch_index, PRIMARY_COMPARATOR)]
            calibrated_batch = by_batch_arm[(batch_index, PRIMARY_CANDIDATE)]
            _require(
                fixed_batch["conversation_ids"] == calibrated_batch["conversation_ids"],
                "Physical arms are not paired on the same conversations.",
            )
        for measurement in measurements:
            _require(
                measurement.get("predictions_identical_to_chunked") is True,
                "Physical/chunked prediction mismatch.",
            )
            arm = measurement["arm"]
            _require(arm in shard.PHYSICAL_ARM_NAMES, "Unexpected physical arm.")
            physical[(raw["scale"], raw["budget"], raw["training_seed"], arm)].append(
                int(measurement["accounting"]["hot_resident_bytes"])
            )
    _require(seen == expected, "Causal Cartesian shard coverage drifted.")
    expected_groups = (
        len(("s55", "s151"))
        * len(shard.BUDGET_LABELS)
        * len(shard.TRAINING_SEEDS)
        * len(PAPER_GRADE_WORKLOAD_FAMILIES)
        * len(shard.CONTEXTS)
    )
    expected_conversations = len(shard.REPLICATES) * shard.EXAMPLES_PER_SHARD
    _require(
        all(
            len(values) == expected_groups
            and all(len(group) == expected_conversations for group in values.values())
            for values in differences.values()
        ),
        "Causal paired-difference coverage drifted.",
    )
    contrast_payload = {
        name: contrast_statistics(
            values,
            name=name,
            candidate=CONTRASTS[name][0],
            comparator=CONTRASTS[name][1],
        )
        for name, values in differences.items()
    }
    for scale in ("s55", "s151"):
        for budget in shard.BUDGET_LABELS:
            cell_rows = {
                name: next(
                    row
                    for row in payload["cells"]
                    if row["scale"] == scale and row["budget"] == budget
                )
                for name, payload in contrast_payload.items()
            }
            adjusted = holm_bonferroni(
                {
                    name: row["seed_cluster_inference"][
                        "paired_randomization_two_sided_p"
                    ]
                    for name, row in cell_rows.items()
                }
            )
            for name, row in cell_rows.items():
                row["holm_adjusted_p_across_contrasts"] = adjusted[name]
    primary = contrast_payload["adaptive_quota_with_pins"]
    for cell in primary["cells"]:
        seed_means = [
            row["mean_difference"]
            for row in primary["by_seed"]
            if row["scale"] == cell["scale"] and row["budget"] == cell["budget"]
        ]
        corrected = seed_cluster_statistics(
            seed_means,
            label=f"causal:primary:four-cell:{cell['scale']}:{cell['budget']}",
            confidence=PRIMARY_CELL_CONFIDENCE,
        )
        cell["four_cell_corrected_bootstrap"] = {
            "confidence_level": PRIMARY_CELL_CONFIDENCE,
            "confidence_interval": corrected["seed_cluster_bootstrap_ci"],
            "independent_seed_clusters": corrected["independent_seed_clusters"],
            "bootstrap_resamples": corrected["bootstrap_resamples"],
            "bootstrap_seed": corrected["inference_seed"],
        }
    memory = physical_memory_statistics(physical)
    gate = primary_causal_gate(primary, memory)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Causal summarization requires a clean source tree.")
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-causal-ablation-audit-v1",
        "source": {"commit": source_commit, "dirty": False},
        "raw_matrix": {"path": str(args.matrix), "sha256": sha256(args.matrix)},
        "implementation_digest": implementation_digest,
        "audit": {
            "all_raw_shards_verified": True,
            "all_dependency_digests_verified": True,
            "all_record_digests_verified": True,
            "no_budget_violations": True,
            "all_physical_predictions_identical": True,
            "exact_config_reuse_verified": True,
            "quality_execution_counts": quality_execution_counts,
            "physical_execution_counts": physical_execution_counts,
            "unique_shards": len(seen),
            "raw_shard_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
        },
        "paired_statistics": contrast_payload,
        "physical_hot_memory": memory,
        "primary_causal_gate": gate,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "claim_boundary": (
            "The gate supports a Tier-S synthetic causal claim only. Natural-language, "
            "large-model, and production-HBM claims remain gated by P3-P5."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"verified_shards": len(seen), "primary_causal_gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
