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

import evaluate_p2_core_shard as shard
import numpy as np

EXPECTED_SHARDS = 4_500
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
BUDGETS = (1, 2, 4)


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


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


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
    _require(raw.get("experiment_id") == "p2-core-quality-shard-v1", "Wrong shard id.")
    _require(raw.get("source", {}).get("dirty") is False, "Dirty P2 shard source.")
    _require(
        raw.get("source", {}).get("implementation_digest") == implementation_digest,
        "P2 implementation digest drifted.",
    )
    for key in ("scale", "training_seed", "family", "context", "replicate"):
        _require(raw.get(key) == run.get(key), f"P2 run metadata drifted: {key}")
    _require(raw.get("examples") == shard.EXAMPLES_PER_SHARD, "Shard size drifted.")
    _require(raw.get("batch_size") == shard.BATCH_SIZE, "Batch size drifted.")
    _require(tuple(raw.get("policies", ())) == shard.CORE_POLICIES, "Policy set drifted.")
    raw_records = raw.get("records")
    _require(isinstance(raw_records, list), "Shard records are not a list.")
    records = cast(list[dict[str, Any]], raw_records)
    _require(
        len(records) == shard.EXAMPLES_PER_SHARD * len(shard.CORE_POLICIES),
        "Shard record count drifted.",
    )
    _require(records_digest(records) == raw.get("records_digest"), "Record digest drifted.")
    for record in records:
        predictions = record.get("predictions", [])
        targets = record.get("targets", [])
        _require(len(predictions) == len(targets), "Prediction/target length drifted.")
        correctness = [
            prediction == target
            for prediction, target in zip(predictions, targets, strict=True)
        ]
        _require(correctness == record.get("correct"), "Correctness field drifted.")
        _require(sum(correctness) == record.get("correct_count"), "Correct count drifted.")
        _require(len(correctness) == record.get("total"), "Query total drifted.")
    _require(
        all(metric.get("budget_violations") == 0 for metric in raw.get("batch_metrics", ())),
        "A shard contains a budget violation.",
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
            family_rows.append({"budget_multiplier": budget, "family": family, **stats})
        adjusted = holm_bonferroni(
            {row["family"]: row["paired_sign_flip_two_sided_p"] for row in family_rows}
        )
        for row in family_rows:
            row["holm_adjusted_p"] = adjusted[row["family"]]
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
                scale_families.append(
                    {
                        "budget_multiplier": budget,
                        "scale": scale,
                        "family": family,
                        **bootstrap_paired_mean(
                            scale_family_values,
                            label=(f"{namespace}:scale-family:{budget}:{scale}:{family}"),
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
    worst = min(slices, key=lambda row: row["mean_difference"])
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
    }


def _quality_gate(
    fixed_statistics: dict[str, Any], native_statistics: dict[str, Any]
) -> list[dict[str, Any]]:
    results = []
    families = fixed_statistics["by_family_with_holm_bonferroni"]
    pooled = fixed_statistics["pooled_by_scale"]
    seeds = fixed_statistics["by_seed"]
    for budget in BUDGETS:
        budget_pooled = [row for row in pooled if row["budget_multiplier"] == budget]
        budget_families = [row for row in families if row["budget_multiplier"] == budget]
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
        results.append(
            {
                "budget_multiplier": budget,
                "positive_pooled_lower_ci_on_both_scales": all(
                    row["paired_cluster_bootstrap_95_ci"][0] > 0.0 for row in budget_pooled
                ),
                "positive_seed_effects": sum(row["mean_difference"] > 0.0 for row in budget_seeds),
                "total_seed_effects": len(budget_seeds),
                "holm_significant_positive_families": sum(
                    row["mean_difference"] > 0.0 and row["holm_adjusted_p"] < 0.05
                    for row in budget_families
                ),
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
            and row["holm_significant_positive_families"]
            >= row["minimum_required_improved_families"]
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
    runs = matrix.get("runs")
    _require(isinstance(runs, list) and len(runs) == EXPECTED_SHARDS, "P2 run count drifted.")
    implementation_digest = matrix["implementation_digest"]
    seen: set[tuple[str, int, str, int, int]] = set()
    outcomes: dict[tuple[str, int, str, int, str], list[tuple[int, int]]] = defaultdict(list)
    fixed_differences: dict[tuple[int, str, int, str, int], list[float]] = defaultdict(list)
    native_differences: dict[tuple[int, str, int, str, int], list[float]] = defaultdict(list)
    raw_digests: list[str] = []
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
    _require(len(seen) == EXPECTED_SHARDS, "P2 shard identity coverage drifted.")
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
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
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
        "raw_matrix": {"path": str(args.matrix), "sha256": sha256(args.matrix)},
        "implementation_digest": implementation_digest,
        "audit": {
            "all_raw_shards_verified": True,
            "all_dependency_digests_verified": True,
            "all_record_digests_verified": True,
            "no_budget_violations": True,
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
