from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any, cast

import evaluate_p2_core_shard as shard
import numpy as np

EXPECTED_SHARDS = 4_500
BOOTSTRAP_RESAMPLES = 10_000
STRICT_RAW_AUDIT = {
    "held_out_seed_contract_verified": True,
    "paired_conversation_coverage_verified": True,
    "execution_order_coverage_verified": True,
    "aggregate_recomputed": True,
    "batch_coverage_verified": True,
}
CONFIDENCE_LEVEL = 0.95
BUDGETS = (1, 2, 4)
CORE_ANALYSIS_PATH = "research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py"
PARALLEL_ORCHESTRATOR_PATH = "research/adaptive_v4_memory/scripts/run_p2_core_parallel.py"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def analysis_implementation(paths: tuple[str, ...]) -> dict[str, Any]:
    """Bind analysis code independently from the raw evaluator implementation."""

    canonical_paths = tuple(sorted(paths))
    tracked = subprocess.run(
        ["git", "ls-files", "-s", "--", *canonical_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    observed = tuple(
        line.split("\t", 1)[1] for line in tracked.splitlines() if "\t" in line
    )
    if observed != canonical_paths:
        raise ValueError(
            f"Analysis implementation paths are untracked or reordered: {observed}"
        )
    return {
        "paths": list(canonical_paths),
        "tracked_file_count": len(canonical_paths),
        "git_index_sha256": hashlib.sha256(tracked.encode()).hexdigest(),
    }


@cache
def implementation_digest_at_commit(commit: str) -> str:
    """Reconstruct the evaluator's index digest from a committed Git tree."""

    _require(COMMIT_PATTERN.fullmatch(commit) is not None, "Invalid P2 execution commit.")
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--full-tree", commit, "--", *shard.IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(bool(tree), f"P2 execution commit has no implementation tree: {commit}")
    normalized: list[str] = []
    for line in tree.splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, object_id = metadata.split()
        _require(object_type == "blob", f"Non-blob P2 implementation entry: {path}")
        normalized.append(f"{mode} {object_id} 0\t{path}\n")
    return hashlib.sha256("".join(normalized).encode()).hexdigest()


@cache
def _commit_is_ancestor(commit: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, descendant],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


@cache
def file_sha256_at_commit(commit: str, path: str) -> str:
    """Hash a file exactly as stored in a committed Git tree."""

    _require(COMMIT_PATTERN.fullmatch(commit) is not None, "Invalid P2 execution commit.")
    content = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def verify_execution_provenance(
    raw: dict[str, Any], implementation_digest: str, analysis_commit: str
) -> tuple[str, str]:
    """Bind each raw shard to its committed evaluator and parallel orchestrator."""

    source = raw.get("source", {})
    execution_commit = source.get("commit")
    _require(
        isinstance(execution_commit, str)
        and COMMIT_PATTERN.fullmatch(execution_commit) is not None,
        "Invalid P2 raw execution commit.",
    )
    _require(
        _commit_is_ancestor(execution_commit, analysis_commit),
        f"P2 execution commit is not an ancestor of analysis HEAD: {execution_commit}",
    )
    _require(
        implementation_digest_at_commit(execution_commit) == implementation_digest,
        f"P2 committed implementation digest drifted: {execution_commit}",
    )
    orchestration = raw.get("orchestration", {})
    expected_orchestrator_sha256 = file_sha256_at_commit(
        execution_commit, PARALLEL_ORCHESTRATOR_PATH
    )
    worker = orchestration.get("worker")
    workers = orchestration.get("workers")
    _require(
        orchestration.get("mode") == "single-gpu-disjoint-processes"
        and orchestration.get("path") == PARALLEL_ORCHESTRATOR_PATH
        and orchestration.get("sha256") == expected_orchestrator_sha256
        and type(worker) is int
        and type(workers) is int
        and workers > 0
        and 0 <= worker < workers,
        "P2 parallel orchestration provenance drifted.",
    )
    return execution_commit, expected_orchestrator_sha256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_digest(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def bootstrap_paired_mean(
    values: Iterable[float],
    *,
    label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Paired bootstrap requires at least one scalar difference.")
    unique, counts = np.unique(array, return_counts=True)
    probabilities = counts / len(array)
    rng = np.random.default_rng(stable_seed(label))
    sampled_counts = rng.multinomial(len(array), probabilities, size=resamples)
    means = sampled_counts @ unique / len(array)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    lower_tail = (np.count_nonzero(means <= 0.0) + 1) / (resamples + 1)
    upper_tail = (np.count_nonzero(means >= 0.0) + 1) / (resamples + 1)
    absolute_values, absolute_counts = np.unique(np.abs(array), return_counts=True)
    positive_counts = rng.binomial(absolute_counts, 0.5, size=(resamples, len(absolute_counts)))
    null_means = ((2 * positive_counts - absolute_counts) @ absolute_values) / len(array)
    sign_flip_p = (np.count_nonzero(np.abs(null_means) >= abs(array.mean())) + 1) / (resamples + 1)
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    mean = float(array.mean())
    return {
        "paired_units": len(array),
        "mean_difference": mean,
        "mean_difference_percentage_points": mean * 100.0,
        "paired_cluster_bootstrap_95_ci": [float(lower), float(upper)],
        "paired_cluster_bootstrap_95_ci_percentage_points": [
            float(lower * 100.0),
            float(upper * 100.0),
        ],
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "paired_sign_flip_two_sided_p": float(sign_flip_p),
        "cohens_dz": mean / standard_deviation if standard_deviation > 0.0 else None,
        "sample_standard_deviation": standard_deviation,
        "bootstrap_resamples": resamples,
        "confidence_level": confidence,
        "bootstrap_seed": stable_seed(label),
        "sign_flip_resamples": resamples,
        "bootstrap_support_values": len(unique),
    }


def seed_cluster_statistics(
    seed_means: Iterable[float],
    *,
    label: str,
    confidence: float = CONFIDENCE_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Infer across independent training seeds, not within-seed examples."""
    values = np.asarray(tuple(seed_means), dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Seed-cluster inference requires at least two independent seeds.")
    rng = np.random.default_rng(stable_seed(label))
    bootstrap_indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        bootstrap_means, (alpha / 2.0, 1.0 - alpha / 2.0)
    )
    lower_tail = (np.count_nonzero(bootstrap_means <= 0.0) + 1) / (resamples + 1)
    upper_tail = (np.count_nonzero(bootstrap_means >= 0.0) + 1) / (resamples + 1)
    observed = float(values.mean())
    threshold = max(0.0, abs(observed) - np.finfo(np.float64).eps * 16.0)
    if len(values) <= 16:
        total_assignments = 1 << len(values)
        assignments = np.arange(total_assignments, dtype=np.uint64)[:, None]
        bit_positions = np.arange(len(values), dtype=np.uint64)[None, :]
        signs = np.where(((assignments >> bit_positions) & 1) == 0, -1.0, 1.0)
        null_means = (signs * values).mean(axis=1)
        randomization_p = np.count_nonzero(np.abs(null_means) >= threshold) / total_assignments
        randomization_method = "exact-sign-flip-enumeration"
        minimum_attainable_p = min(1.0, 2.0 / total_assignments)
    else:
        signs = np.where(
            rng.integers(0, 2, size=(resamples, len(values))) == 0, -1.0, 1.0
        )
        null_means = (signs * values).mean(axis=1)
        randomization_p = (
            np.count_nonzero(np.abs(null_means) >= threshold) + 1
        ) / (resamples + 1)
        total_assignments = resamples
        randomization_method = "monte-carlo-sign-flip"
        minimum_attainable_p = 1.0 / (resamples + 1)
    standard_deviation = float(values.std(ddof=1))
    return {
        "independent_seed_clusters": len(values),
        "seed_means": values.tolist(),
        "mean_difference": observed,
        "mean_difference_percentage_points": observed * 100.0,
        "seed_mean_sample_standard_deviation": standard_deviation,
        "seed_mean_range": [float(values.min()), float(values.max())],
        "cohens_dz_across_seeds": (
            observed / standard_deviation if standard_deviation > 0.0 else None
        ),
        "confidence_level": confidence,
        "seed_cluster_bootstrap_ci": [float(lower), float(upper)],
        "two_sided_seed_cluster_bootstrap_p": min(
            1.0, 2.0 * min(lower_tail, upper_tail)
        ),
        "paired_randomization_two_sided_p": float(randomization_p),
        "paired_randomization_method": randomization_method,
        "paired_randomization_assignments": total_assignments,
        "minimum_attainable_two_sided_p": minimum_attainable_p,
        "bootstrap_resamples": resamples,
        "inference_seed": stable_seed(label),
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def validate_family_holm_contract(
    families: list[dict[str, Any]], scale_families: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify every preregistered core family-wise correction group."""

    family_names = set(shard.PAPER_GRADE_WORKLOAD_FAMILIES)
    expected_pooled = {
        (budget, family) for budget in BUDGETS for family in family_names
    }
    expected_scale = {
        (budget, scale, family)
        for budget in BUDGETS
        for scale in shard.CHUNK_SIZE_BY_SCALE
        for family in family_names
    }
    source = "seed_cluster_exact_paired_randomization_two_sided_p"
    _require(
        {(row["budget_multiplier"], row["family"]) for row in families}
        == expected_pooled
        and all(
            "holm_adjusted_p" in row and row.get("holm_source_p") == source
            for row in families
        ),
        "Core pooled-family Holm coverage drifted.",
    )
    _require(
        {
            (row["budget_multiplier"], row["scale"], row["family"])
            for row in scale_families
        }
        == expected_scale
        and all(
            "holm_adjusted_p" in row and row.get("holm_source_p") == source
            for row in scale_families
        ),
        "Core scale-family Holm coverage drifted.",
    )
    return {
        "family_holm_bonferroni_verified": True,
        "families_per_holm_group": len(family_names),
        "family_holm_groups_per_comparison": len(BUDGETS)
        * (1 + len(shard.CHUNK_SIZE_BY_SCALE)),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_dependency(metadata: dict[str, Any], name: str) -> None:
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing {name}: {path}")
    _require(metadata.get("sha256") == sha256(path), f"{name} digest drifted: {path}")


def verify_raw_shard(
    raw: dict[str, Any],
    run: dict[str, Any],
    implementation_digest: str,
) -> None:
    _require(raw.get("schema_version") == 1, "Wrong shard schema version.")
    _require(raw.get("experiment_id") == "p2-core-quality-shard-v1", "Wrong shard id.")
    _require(raw.get("source", {}).get("dirty") is False, "Dirty P2 shard source.")
    _require(
        raw.get("source", {}).get("implementation_digest") == implementation_digest,
        "P2 implementation digest drifted.",
    )
    for key in ("scale", "training_seed", "family", "context", "replicate"):
        _require(raw.get(key) == run.get(key), f"P2 run metadata drifted: {key}")
    training_seed = raw["training_seed"]
    _require(
        raw.get("evaluation_seed_namespace") == "held_out_evaluation"
        and raw.get("evaluation_seed") == shard._evaluation_seed(training_seed),
        "Held-out evaluation seed drifted.",
    )
    _require(
        raw.get("generation_seed")
        == shard._generation_seed(
            raw["evaluation_seed"], raw["family"], raw["context"], raw["replicate"]
        ),
        "Generation seed drifted.",
    )
    _require(raw.get("examples") == shard.EXAMPLES_PER_SHARD, "Shard size drifted.")
    _require(raw.get("batch_size") == shard.BATCH_SIZE, "Batch size drifted.")
    _require(
        raw.get("chunk_size") == shard.CHUNK_SIZE_BY_SCALE[raw["scale"]],
        "Chunk size drifted.",
    )
    _require(tuple(raw.get("policies", ())) == shard.CORE_POLICIES, "Policy set drifted.")
    _require(
        set(raw.get("policy_configs", {})) == set(shard.CORE_POLICIES),
        "Policy configuration coverage drifted.",
    )
    _require(
        raw.get("leakage_guard")
        == {
            "calibration_seed_used_for_evaluation": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_policies": True,
        },
        "P2 leakage guard drifted.",
    )
    raw_records = raw.get("records")
    _require(isinstance(raw_records, list), "Shard records are not a list.")
    records = cast(list[dict[str, Any]], raw_records)
    _require(
        len(records) == shard.EXAMPLES_PER_SHARD * len(shard.CORE_POLICIES),
        "Shard record count drifted.",
    )
    _require(records_digest(records) == raw.get("records_digest"), "Record digest drifted.")
    conversation_ids_by_policy: dict[str, set[str]] = {
        policy: set() for policy in shard.CORE_POLICIES
    }
    expected_conversation_ids = {
        f"{raw['family']}:{raw['context']}:{index}"
        for index in range(
            raw["replicate"] * shard.EXAMPLES_PER_SHARD,
            (raw["replicate"] + 1) * shard.EXAMPLES_PER_SHARD,
        )
    }
    for record in records:
        _require(record.get("policy") in shard.CORE_POLICIES, "Unknown record policy.")
        for field in ("family", "context", "replicate"):
            _require(record.get(field) == raw[field], f"Record {field} drifted.")
        conversation_id = record.get("conversation_id")
        _require(isinstance(conversation_id, str), "Conversation id drifted.")
        conversation_id = cast(str, conversation_id)
        policy_ids = conversation_ids_by_policy[record["policy"]]
        _require(conversation_id not in policy_ids, "Duplicate policy-conversation record.")
        policy_ids.add(conversation_id)
        predictions = record.get("predictions")
        targets = record.get("targets")
        query_positions = record.get("query_positions")
        evidence_positions = record.get("evidence_positions")
        _require(
            isinstance(predictions, list)
            and isinstance(targets, list)
            and isinstance(query_positions, list)
            and isinstance(evidence_positions, list)
            and len(targets) > 0
            and len(predictions)
            == len(targets)
            == len(query_positions)
            == len(evidence_positions)
            and all(type(value) is int and value >= 0 for value in (*targets, *predictions))
            and all(
                type(value) is int and 0 <= value < raw["context"]
                for value in (*query_positions, *evidence_positions)
            )
            and query_positions == sorted(set(query_positions))
            and all(
                evidence < query
                for evidence, query in zip(
                    evidence_positions, query_positions, strict=True
                )
            ),
            "Record target and position schema drifted.",
        )
        prediction_values = cast(list[int], predictions)
        target_values = cast(list[int], targets)
        correctness = [
            prediction == target
            for prediction, target in zip(
                prediction_values, target_values, strict=True
            )
        ]
        _require(
            isinstance(record.get("correct"), list)
            and all(type(value) is bool for value in record["correct"])
            and correctness == record["correct"],
            "Correctness field drifted.",
        )
        _require(sum(correctness) == record.get("correct_count"), "Correct count drifted.")
        _require(len(correctness) == record.get("total"), "Query total drifted.")
    native_ids = conversation_ids_by_policy["native"]
    _require(
        native_ids == expected_conversation_ids
        and all(ids == native_ids for ids in conversation_ids_by_policy.values()),
        "Paired conversation coverage drifted.",
    )
    _require(raw.get("aggregate") == shard._aggregate(records), "Aggregate drifted.")
    raw_batch_metrics = raw.get("batch_metrics")
    _require(isinstance(raw_batch_metrics, list), "Batch metrics are not a list.")
    batch_metrics = cast(list[dict[str, Any]], raw_batch_metrics)
    expected_batches = shard.EXAMPLES_PER_SHARD // shard.BATCH_SIZE
    _require(
        len(batch_metrics) == expected_batches * len(shard.CORE_POLICIES),
        "Batch metric coverage drifted.",
    )
    for batch_index in range(expected_batches):
        rows = [row for row in batch_metrics if row.get("batch_index") == batch_index]
        rotation = batch_index % len(shard.CORE_POLICIES)
        expected_order = (
            *shard.CORE_POLICIES[rotation:],
            *shard.CORE_POLICIES[:rotation],
        )
        _require(
            len(rows) == len(shard.CORE_POLICIES)
            and all(
                row.get("policy") == policy
                and row.get("execution_index") == execution_index
                and type(row.get("budget_violations")) is int
                and row["budget_violations"] == 0
                and type(row.get("wall_ms")) in (int, float)
                and math.isfinite(float(row["wall_ms"]))
                and row["wall_ms"] >= 0.0
                and (
                    isinstance(row.get("controller"), dict)
                    if policy.startswith("calibrated-hierarchical-")
                    else row.get("controller") is None
                )
                for execution_index, (row, policy) in enumerate(
                    zip(rows, expected_order, strict=True)
                )
            ),
            "Batch policy execution coverage drifted.",
        )
    _verify_dependency(raw["checkpoint"], "checkpoint")
    _verify_dependency(raw["calibration_artifact"], "calibration artifact")
    _verify_dependency(raw["equivalence_artifact"], "equivalence artifact")
    shard._equivalence(Path(raw["equivalence_artifact"]["path"]), raw["scale"])


def _summary_row(
    key: tuple[str, int, str, int, str], values: list[tuple[int, int]]
) -> dict[str, Any]:
    scale, training_seed, family, context, policy = key
    correct = sum(item[0] for item in values)
    total = sum(item[1] for item in values)
    return {
        "scale": scale,
        "training_seed": training_seed,
        "family": family,
        "context": context,
        "policy": policy,
        "conversations": len(values),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }


def _merge(groups: Iterable[list[float]]) -> list[float]:
    merged: list[float] = []
    for group in groups:
        merged.extend(group)
    return merged


def verify_statistical_coverage(
    differences: dict[tuple[int, str, int, str, int], list[float]],
) -> dict[str, int]:
    expected_keys = set(
        product(
            BUDGETS,
            shard.CHUNK_SIZE_BY_SCALE,
            shard.TRAINING_SEEDS,
            shard.PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
        )
    )
    paired_units_per_context = len(shard.REPLICATES) * shard.EXAMPLES_PER_SHARD
    _require(
        set(differences) == expected_keys
        and all(
            len(differences[key]) == paired_units_per_context for key in expected_keys
        ),
        "P2 paired statistical cell coverage drifted.",
    )
    return {
        "paired_units_per_seed_scale_family_context": paired_units_per_context,
        "paired_units_per_seed_scale_family": paired_units_per_context
        * len(shard.CONTEXTS),
        "statistical_cells_per_comparison": len(expected_keys),
    }


def _statistics(
    differences: dict[tuple[int, str, int, str, int], list[float]],
    *,
    comparison: str,
    namespace: str,
) -> dict[str, Any]:
    pooled: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    scale_families: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for scale in shard.CHUNK_SIZE_BY_SCALE:
            values = _merge(
                group
                for (
                    item_budget,
                    item_scale,
                    _seed,
                    _family,
                    _context,
                ), group in differences.items()
                if item_budget == budget and item_scale == scale
            )
            stats = bootstrap_paired_mean(values, label=f"{namespace}:pooled:{budget}:{scale}")
            pooled.append({"budget_multiplier": budget, "scale": scale, **stats})
            cell_seed_means: list[float] = []
            for training_seed in shard.TRAINING_SEEDS:
                seed_values = _merge(
                    group
                    for (
                        item_budget,
                        item_scale,
                        item_seed,
                        _family,
                        _context,
                    ), group in differences.items()
                    if item_budget == budget and item_scale == scale and item_seed == training_seed
                )
                seeds.append(
                    {
                        "budget_multiplier": budget,
                        "scale": scale,
                        "training_seed": training_seed,
                        "paired_units": len(seed_values),
                        "mean_difference": float(np.mean(seed_values)),
                        "mean_difference_percentage_points": float(np.mean(seed_values) * 100.0),
                    }
                )
                cell_seed_means.append(float(np.mean(seed_values)))
            pooled[-1]["seed_cluster_inference"] = seed_cluster_statistics(
                cell_seed_means,
                label=f"{namespace}:pooled-seeds:{budget}:{scale}",
            )
        family_rows: list[dict[str, Any]] = []
        for family in shard.PAPER_GRADE_WORKLOAD_FAMILIES:
            values = _merge(
                group
                for (
                    item_budget,
                    _scale,
                    _seed,
                    item_family,
                    _context,
                ), group in differences.items()
                if item_budget == budget and item_family == family
            )
            stats = bootstrap_paired_mean(values, label=f"{namespace}:family:{budget}:{family}")
            family_seed_means = [
                float(
                    np.mean(
                        _merge(
                            group
                            for (
                                item_budget,
                                _scale,
                                item_seed,
                                item_family,
                                _context,
                            ), group in differences.items()
                            if item_budget == budget
                            and item_seed == training_seed
                            and item_family == family
                        )
                    )
                )
                for training_seed in shard.TRAINING_SEEDS
            ]
            family_rows.append(
                {
                    "budget_multiplier": budget,
                    "family": family,
                    **stats,
                    "seed_cluster_inference": seed_cluster_statistics(
                        family_seed_means,
                        label=f"{namespace}:family-seeds:{budget}:{family}",
                    ),
                }
            )
        adjusted = holm_bonferroni(
            {
                row["family"]: row["seed_cluster_inference"][
                    "paired_randomization_two_sided_p"
                ]
                for row in family_rows
            }
        )
        for row in family_rows:
            row["holm_adjusted_p"] = adjusted[row["family"]]
            row["holm_source_p"] = (
                "seed_cluster_exact_paired_randomization_two_sided_p"
            )
        families.extend(family_rows)
        for scale in shard.CHUNK_SIZE_BY_SCALE:
            for family in shard.PAPER_GRADE_WORKLOAD_FAMILIES:
                scale_family_values = _merge(
                    group
                    for (
                        item_budget,
                        item_scale,
                        _seed,
                        item_family,
                        _context,
                    ), group in differences.items()
                    if item_budget == budget and item_scale == scale and item_family == family
                )
                scale_family_seed_means = [
                    float(
                        np.mean(
                            _merge(
                                group
                                for (
                                    item_budget,
                                    item_scale,
                                    item_seed,
                                    item_family,
                                    _context,
                                ), group in differences.items()
                                if item_budget == budget
                                and item_scale == scale
                                and item_seed == training_seed
                                and item_family == family
                            )
                        )
                    )
                    for training_seed in shard.TRAINING_SEEDS
                ]
                scale_families.append(
                    {
                        "budget_multiplier": budget,
                        "scale": scale,
                        "family": family,
                        **bootstrap_paired_mean(
                            scale_family_values,
                            label=(f"{namespace}:scale-family:{budget}:{scale}:{family}"),
                        ),
                        "seed_cluster_inference": seed_cluster_statistics(
                            scale_family_seed_means,
                            label=(
                                f"{namespace}:scale-family-seeds:{budget}:{scale}:{family}"
                            ),
                        ),
                    }
                )
                for context in shard.CONTEXTS:
                    key_groups = [
                        group
                        for (
                            item_budget,
                            item_scale,
                            _seed,
                            item_family,
                            item_context,
                        ), group in differences.items()
                        if item_budget == budget
                        and item_scale == scale
                        and item_family == family
                        and item_context == context
                    ]
                    values = _merge(key_groups)
                    slices.append(
                        {
                            "budget_multiplier": budget,
                            "scale": scale,
                            "family": family,
                            "context": context,
                            "paired_units": len(values),
                            "mean_difference": float(np.mean(values)),
                            "mean_difference_percentage_points": float(np.mean(values) * 100.0),
                        }
                    )
        for scale in shard.CHUNK_SIZE_BY_SCALE:
            scale_rows = [
                row
                for row in scale_families
                if row["budget_multiplier"] == budget and row["scale"] == scale
            ]
            adjusted = holm_bonferroni(
                {
                    row["family"]: row["seed_cluster_inference"][
                        "paired_randomization_two_sided_p"
                    ]
                    for row in scale_rows
                }
            )
            for row in scale_rows:
                row["holm_adjusted_p"] = adjusted[row["family"]]
                row["holm_source_p"] = (
                    "seed_cluster_exact_paired_randomization_two_sided_p"
                )
    worst = min(slices, key=lambda row: row["mean_difference"])
    worst_by_budget_scale = [
        min(
            (row for row in slices if row["budget_multiplier"] == budget and row["scale"] == scale),
            key=lambda row: row["mean_difference"],
        )
        for budget in BUDGETS
        for scale in shard.CHUNK_SIZE_BY_SCALE
    ]
    seed_variance: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for scale in shard.CHUNK_SIZE_BY_SCALE:
            values = [
                row["mean_difference"]
                for row in seeds
                if row["budget_multiplier"] == budget and row["scale"] == scale
            ]
            seed_variance.append(
                {
                    "budget_multiplier": budget,
                    "scale": scale,
                    "independent_training_seeds": len(values),
                    "mean": float(np.mean(values)),
                    "sample_standard_deviation": float(np.std(values, ddof=1)),
                    "range": [float(min(values)), float(max(values))],
                    "all_seed_values": values,
                }
            )
    return {
        "comparison": comparison,
        "pooled_by_scale": pooled,
        "by_family_with_holm_bonferroni": families,
        "by_scale_family": scale_families,
        "by_seed": seeds,
        "seed_variance": seed_variance,
        "by_scale_family_context": slices,
        "worst_slice": worst,
        "worst_slice_by_budget_scale": worst_by_budget_scale,
        "multiplicity_audit": validate_family_holm_contract(
            families, scale_families
        ),
    }


def _quality_gate(
    fixed_statistics: dict[str, Any], native_statistics: dict[str, Any]
) -> list[dict[str, Any]]:
    results = []
    pooled = fixed_statistics["pooled_by_scale"]
    seeds = fixed_statistics["by_seed"]
    for budget in BUDGETS:
        budget_pooled = [row for row in pooled if row["budget_multiplier"] == budget]
        budget_scale_families = [
            row for row in fixed_statistics["by_scale_family"] if row["budget_multiplier"] == budget
        ]
        budget_seeds = [row for row in seeds if row["budget_multiplier"] == budget]
        native_pooled = [
            row
            for row in native_statistics["pooled_by_scale"]
            if row["budget_multiplier"] == budget
        ]
        native_scale_families = [
            row
            for row in native_statistics["by_scale_family"]
            if row["budget_multiplier"] == budget
        ]
        corrected_positive_by_scale = {
            scale: sum(
                row["mean_difference"] > 0.0
                and row["seed_cluster_inference"]["seed_cluster_bootstrap_ci"][0] >= 0.0
                for row in budget_scale_families
                if row["scale"] == scale
            )
            for scale in shard.CHUNK_SIZE_BY_SCALE
        }
        results.append(
            {
                "budget_multiplier": budget,
                "positive_pooled_lower_ci_on_both_scales": all(
                    row["seed_cluster_inference"]["seed_cluster_bootstrap_ci"][0] > 0.0
                    for row in budget_pooled
                ),
                "positive_seed_effects": sum(row["mean_difference"] > 0.0 for row in budget_seeds),
                "total_seed_effects": len(budget_seeds),
                "all_seed_effects_positive": all(
                    row["mean_difference"] > 0.0 for row in budget_seeds
                ),
                "corrected_positive_families_by_scale": corrected_positive_by_scale,
                "family_holm_p_values_used_as_success_gate": False,
                "minimum_required_improved_families": 2,
                "native_mean_regression_within_1pp_on_both_scales": all(
                    row["mean_difference"] >= -0.01 for row in native_pooled
                ),
                "native_family_regression_within_2pp_on_both_scales": all(
                    row["mean_difference"] >= -0.02 for row in native_scale_families
                ),
            }
        )
    for row in results:
        row["passes_fixed_baseline_component"] = (
            row["positive_pooled_lower_ci_on_both_scales"]
            and row["all_seed_effects_positive"]
            and all(
                count >= row["minimum_required_improved_families"]
                for count in row["corrected_positive_families_by_scale"].values()
            )
            and row["native_mean_regression_within_1pp_on_both_scales"]
            and row["native_family_regression_within_2pp_on_both_scales"]
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize the full P2 core matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    _require(
        matrix.get("experiment_id") == "p2-core-quality-matrix-progress-v1",
        "Wrong P2 matrix id.",
    )
    _require(matrix.get("completed_shards") == EXPECTED_SHARDS, "P2 matrix is incomplete.")
    design = matrix.get("frozen_design", {})
    _require(tuple(design.get("scales", ())) == tuple(shard.CHUNK_SIZE_BY_SCALE), "Scale drift.")
    _require(tuple(design.get("training_seeds", ())) == shard.TRAINING_SEEDS, "Seed drift.")
    _require(
        tuple(design.get("families", ())) == shard.PAPER_GRADE_WORKLOAD_FAMILIES,
        "Family drift.",
    )
    _require(tuple(design.get("contexts", ())) == shard.CONTEXTS, "Context drift.")
    _require(tuple(design.get("replicates", ())) == shard.REPLICATES, "Replicate drift.")
    runs = matrix.get("runs")
    _require(isinstance(runs, list) and len(runs) == EXPECTED_SHARDS, "P2 run count drifted.")
    implementation_digest = matrix["implementation_digest"]
    _require(
        implementation_digest == shard._implementation_digest(),
        "Matrix implementation is not the checked-out P2 implementation.",
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    matrix_source_commit = matrix.get("source_commit")
    _require(
        isinstance(matrix_source_commit, str)
        and COMMIT_PATTERN.fullmatch(matrix_source_commit) is not None,
        "Invalid P2 matrix source commit.",
    )
    _require(
        _commit_is_ancestor(matrix_source_commit, source_commit),
        "P2 matrix source commit is not an ancestor of analysis HEAD.",
    )
    _require(
        implementation_digest_at_commit(matrix_source_commit) == implementation_digest,
        "P2 matrix commit does not contain the recorded implementation.",
    )
    expected_identities = set(
        product(
            shard.CHUNK_SIZE_BY_SCALE,
            shard.TRAINING_SEEDS,
            shard.PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
            shard.REPLICATES,
        )
    )
    seen: set[tuple[str, int, str, int, int]] = set()
    outcomes: dict[tuple[str, int, str, int, str], list[tuple[int, int]]] = defaultdict(list)
    fixed_differences: dict[tuple[int, str, int, str, int], list[float]] = defaultdict(list)
    native_differences: dict[tuple[int, str, int, str, int], list[float]] = defaultdict(list)
    raw_digests: list[str] = []
    execution_source_commits: set[str] = set()
    orchestration_digests: set[str] = set()
    for run in runs:
        identity = (
            run["scale"],
            run["training_seed"],
            run["family"],
            run["context"],
            run["replicate"],
        )
        _require(identity not in seen, f"Duplicate P2 shard: {identity}")
        seen.add(identity)
        raw_metadata = run["raw_artifact"]
        raw_path = Path(raw_metadata["path"])
        _require(raw_path.is_file(), f"Missing P2 shard: {raw_path}")
        digest = sha256(raw_path)
        _require(digest == raw_metadata["sha256"], f"P2 shard digest drifted: {raw_path}")
        raw_digests.append(digest)
        raw = json.loads(raw_path.read_text())
        verify_raw_shard(raw, run, implementation_digest)
        execution_commit, orchestration_digest = verify_execution_provenance(
            raw, implementation_digest, source_commit
        )
        execution_source_commits.add(execution_commit)
        orchestration_digests.add(orchestration_digest)
        by_policy_and_conversation = {
            (record["policy"], record["conversation_id"]): record for record in raw["records"]
        }
        _require(
            len(by_policy_and_conversation) == len(raw["records"]),
            "Duplicate policy-conversation record in a P2 shard.",
        )
        for record in raw["records"]:
            outcomes[
                (
                    raw["scale"],
                    raw["training_seed"],
                    raw["family"],
                    raw["context"],
                    record["policy"],
                )
            ].append((record["correct_count"], record["total"]))
        conversation_ids = sorted(
            record["conversation_id"] for record in raw["records"] if record["policy"] == "native"
        )
        for budget in BUDGETS:
            fixed_name = f"fixed-{budget}x"
            calibrated_name = f"calibrated-hierarchical-{budget}x"
            for conversation_id in conversation_ids:
                native = by_policy_and_conversation[("native", conversation_id)]
                fixed = by_policy_and_conversation[(fixed_name, conversation_id)]
                calibrated = by_policy_and_conversation[(calibrated_name, conversation_id)]
                _require(fixed["total"] == calibrated["total"], "Paired query count drifted.")
                _require(native["total"] == calibrated["total"], "Native query count drifted.")
                for field in ("targets", "query_positions", "evidence_positions"):
                    _require(
                        fixed[field] == calibrated[field] == native[field],
                        f"Paired {field} drifted.",
                    )
                key = (
                    budget,
                    raw["scale"],
                    raw["training_seed"],
                    raw["family"],
                    raw["context"],
                )
                fixed_differences[key].append(
                    calibrated["correct_count"] / calibrated["total"]
                    - fixed["correct_count"] / fixed["total"]
                )
                native_differences[key].append(
                    calibrated["correct_count"] / calibrated["total"]
                    - native["correct_count"] / native["total"]
                )
    _require(seen == expected_identities, "P2 Cartesian shard coverage drifted.")
    expected_outcome_keys = set(
        product(
            shard.CHUNK_SIZE_BY_SCALE,
            shard.TRAINING_SEEDS,
            shard.PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
            shard.CORE_POLICIES,
        )
    )
    paired_units_per_context = len(shard.REPLICATES) * shard.EXAMPLES_PER_SHARD
    _require(
        set(outcomes) == expected_outcome_keys
        and all(len(outcomes[key]) == paired_units_per_context for key in expected_outcome_keys),
        "P2 policy outcome cell coverage drifted.",
    )
    fixed_coverage = verify_statistical_coverage(fixed_differences)
    native_coverage = verify_statistical_coverage(native_differences)
    _require(fixed_coverage == native_coverage, "P2 comparison coverage drifted.")
    policy_summary = [_summary_row(key, values) for key, values in sorted(outcomes.items())]
    fixed_statistics = _statistics(
        fixed_differences,
        comparison="calibrated-hierarchical-minus-memory-matched-fixed",
        namespace="fixed",
    )
    native_statistics = _statistics(
        native_differences,
        comparison="calibrated-hierarchical-minus-native",
        namespace="native",
    )
    multiplicity_audit = fixed_statistics["multiplicity_audit"]
    _require(
        multiplicity_audit == native_statistics["multiplicity_audit"],
        "P2 comparison multiplicity contracts drifted.",
    )
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "P2 summarization requires a clean source tree.")
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-core-quality-matrix-audit-v1",
        "source": {"commit": source_commit, "dirty": False},
        "analysis_implementation": analysis_implementation((CORE_ANALYSIS_PATH,)),
        "raw_matrix": {"path": str(args.matrix), "sha256": sha256(args.matrix)},
        "implementation_digest": implementation_digest,
        "execution_provenance": {
            "raw_source_commits": sorted(execution_source_commits),
            "raw_source_commit_count": len(execution_source_commits),
            "matrix_source_commit": matrix_source_commit,
            "analysis_source_commit": source_commit,
            "implementation_digest": implementation_digest,
            "parallel_orchestrator": {
                "path": PARALLEL_ORCHESTRATOR_PATH,
                "sha256": next(iter(orchestration_digests)),
            },
        },
        "audit": {
            "all_raw_shards_verified": True,
            "all_dependency_digests_verified": True,
            "all_record_digests_verified": True,
            "no_budget_violations": True,
            "exact_seed_randomization_verified": True,
            "independent_seed_clusters_per_cell": len(shard.TRAINING_SEEDS),
            "minimum_attainable_two_sided_seed_p": 2.0
            / (1 << len(shard.TRAINING_SEEDS)),
            "seed_p_values_used_as_success_gate": False,
            "family_holm_p_values_used_as_success_gate": False,
            **multiplicity_audit,
            "exact_record_schema_verified": True,
            "exact_execution_rotation_verified": True,
            "exact_statistical_cell_coverage_verified": True,
            "raw_execution_commits_are_ancestors": True,
            "raw_execution_commit_trees_verified": True,
            "parallel_orchestration_verified": len(orchestration_digests) == 1,
            **fixed_coverage,
            **STRICT_RAW_AUDIT,
            "unique_shards": len(seen),
            "raw_shard_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
        },
        "frozen_design": matrix["frozen_design"],
        "policy_summary": policy_summary,
        "paired_statistics": {
            "calibrated_minus_fixed": fixed_statistics,
            "calibrated_minus_native": native_statistics,
        },
        "quality_gate": _quality_gate(fixed_statistics, native_statistics),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "claim_boundary": (
            "Core statistics compare the frozen calibrated hierarchical and fixed arms. "
            "They do not satisfy the separate fixed+pins causal-factorial gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "verified_shards": len(seen),
                "worst_slice": fixed_statistics["worst_slice"],
                "quality_gate": payload["quality_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
