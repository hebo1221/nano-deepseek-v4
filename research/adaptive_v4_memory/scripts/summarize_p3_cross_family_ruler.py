from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from p3_source_provenance import verify_git_implementation, verify_runtime_kvpress_binding
from run_p3_cross_family_ruler import (
    ARMS,
    CELL_EXPERIMENT_ID,
    EXPECTED_EXAMPLES,
    LENGTHS,
    MODEL_REVISION,
    OFFICIAL_SCORER_SHA256,
    RULER_REVISION,
    TASKS,
    arm_config,
    load_dataset_contracts,
)
from run_p3_mrcr import atomic_json
from run_p3_ruler_matrix import KVPRESS_REVISION, example_score, load_dataset
from summarize_p2_core_matrix import holm_bonferroni
from validate_p3_cross_family_ruler_manifest import validate_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-cross-family-ruler-transfer-audit-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from error
            _require(isinstance(row, dict), f"Non-object record at {path}:{line_number}.")
            records.append(row)
    return records


def _answer_map(dataset_manifests: dict[int, dict[str, Any]]) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    for length in LENGTHS:
        frame = load_dataset(dataset_manifests[length])
        for row in frame.to_dict(orient="records"):
            example_id = f"{length}:{row['row_id']}"
            references = row["answer"]
            _require(
                isinstance(references, list)
                and bool(references)
                and all(isinstance(value, str) for value in references),
                f"Invalid RULER references for {example_id}.",
            )
            _require(example_id not in answers, f"Duplicate dataset example: {example_id}.")
            answers[example_id] = references
    _require(len(answers) == EXPECTED_EXAMPLES, "Cross-family answer grid drifted.")
    return answers


def verify_sequence_dependencies(dependencies: Any, *, arm: str) -> None:
    _require(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {"primary_core", "primary_causal", "nine_seed_causal", "fixed_selection"},
        f"Phi RULER sequence dependencies drifted for {arm}.",
    )
    for name, metadata in dependencies.items():
        _require(isinstance(metadata, dict), f"Malformed Phi RULER {name} dependency.")
        dependency_path = Path(metadata.get("path", ""))
        _require(
            dependency_path.is_file() and metadata.get("sha256") == sha256(dependency_path),
            f"Phi RULER {name} dependency drifted for {arm}.",
        )


def _load_arm(
    root: Path,
    *,
    arm: str,
    manifest_digest: str,
    runner_digest: str,
    dataset_digest_set: str,
    selection_digest: str,
    expected_config: dict[str, Any],
    answers: dict[str, list[str]],
    maximum_context: int,
    model_digest_set: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell_path = root / arm / "cell.json"
    _require(cell_path.is_file(), f"Missing Phi RULER cell: {cell_path}.")
    cell = json.loads(cell_path.read_text())
    identity = cell.get("run_identity", {})
    _require(
        cell.get("experiment_id") == CELL_EXPERIMENT_ID
        and cell.get("benchmark") == "RULER"
        and cell.get("arm") == arm
        and cell.get("status") == "terminal",
        f"Phi RULER cell identity drifted for {arm}.",
    )
    _require(
        cell.get("source", {}).get("dirty") is False
        and cell.get("source", {}).get("implementation_sha256") == runner_digest
        and identity.get("implementation_sha256") == runner_digest,
        f"Phi RULER runner provenance drifted for {arm}.",
    )
    verify_git_implementation(
        cell.get("source"),
        expected_path="research/adaptive_v4_memory/scripts/run_p3_cross_family_ruler.py",
        label=f"Phi RULER {arm}",
    )
    _require(
        cell.get("experiment_manifest", {}).get("sha256") == manifest_digest
        and identity.get("manifest_sha256") == manifest_digest
        and identity.get("dataset_manifest_digest_set_sha256") == dataset_digest_set
        and cell.get("benchmark_dataset_digest_set_sha256") == dataset_digest_set
        and identity.get("fixed_selection_sha256") == selection_digest
        and identity.get("arm_config") == expected_config,
        f"Phi RULER dependency provenance drifted for {arm}.",
    )
    _require(
        identity.get("model_snapshot_digest_set_sha256")
        == cell.get("model_snapshot_digest_set_sha256")
        == model_digest_set
        and identity.get("kvpress_revision") == KVPRESS_REVISION
        and identity.get("official_scorer_sha256") == OFFICIAL_SCORER_SHA256
        and identity.get("seed") == 42,
        f"Phi RULER runtime identity drifted for {arm}.",
    )
    dependencies = identity.get("sequence_gate_dependencies", {})
    _require(
        cell.get("sequence_gate", {}).get("dependencies") == dependencies,
        f"Phi RULER sequence dependencies drifted for {arm}.",
    )
    verify_sequence_dependencies(dependencies, arm=arm)
    verify_runtime_kvpress_binding(cell.get("environment", {}).get("kvpress_binding"))
    raw = cell.get("raw_records", {})
    records_path = Path(raw.get("path", ""))
    _require(
        records_path.is_file() and raw.get("sha256") == sha256(records_path),
        f"Phi RULER raw records drifted for {arm}.",
    )
    records = _read_jsonl(records_path)
    _require(len(records) == EXPECTED_EXAMPLES, f"Phi RULER record count drifted for {arm}.")
    expected_ids = list(answers)
    observed_ids = [row.get("example_id") for row in records]
    _require(observed_ids == expected_ids, f"Phi RULER record order drifted for {arm}.")
    _require(len(set(observed_ids)) == EXPECTED_EXAMPLES, f"Duplicate Phi records for {arm}.")

    for record in records:
        example_id = record["example_id"]
        length_text, task, _row = example_id.split(":", 2)
        length = int(length_text)
        _require(
            record.get("benchmark") == "RULER"
            and record.get("arm") == arm
            and record.get("length_tokens") == length
            and record.get("task") == task
            and length in LENGTHS
            and task in TASKS
            and record.get("arm_config") == expected_config,
            f"Phi RULER coordinates drifted at {example_id}/{arm}.",
        )
        revisions = record.get("revisions", {})
        _require(
            revisions.get("model_revision") == MODEL_REVISION
            and revisions.get("dataset_revision") == RULER_REVISION
            and revisions.get("code_revision") == KVPRESS_REVISION
            and revisions.get("official_scorer_sha256") == OFFICIAL_SCORER_SHA256,
            f"Phi RULER revisions drifted at {example_id}/{arm}.",
        )
        exact_tokens = record.get("exact_input_tokens")
        reserve = record.get("generation_reserve_tokens")
        _require(
            type(exact_tokens) is int
            and exact_tokens > 0
            and exact_tokens <= length
            and type(reserve) is int
            and reserve > 0
            and exact_tokens + reserve <= length
            and length <= maximum_context
            and record.get("silently_truncated") is False,
            f"Phi RULER token contract drifted at {example_id}/{arm}.",
        )
        _require(
            isinstance(record.get("raw_prompt_sha256"), str)
            and len(record["raw_prompt_sha256"]) == 64
            and isinstance(record.get("input_token_ids_sha256"), str)
            and len(record["input_token_ids_sha256"]) == 64
            and type(record.get("token_boundary_retreat")) is int,
            f"Phi RULER prompt identity drifted at {example_id}/{arm}.",
        )
        _require(
            isinstance(record.get("latency_ms"), (int, float))
            and math.isfinite(float(record["latency_ms"]))
            and record["latency_ms"] >= 0
            and type(record.get("peak_hbm_bytes")) is int
            and record["peak_hbm_bytes"] >= 0
            and type(record.get("hot_resident_bytes")) is int
            and record["hot_resident_bytes"] >= 0,
            f"Phi RULER physical measurement drifted at {example_id}/{arm}.",
        )
        status = record.get("status")
        _require(status in {"scored", "failure"}, f"Unknown Phi status at {example_id}/{arm}.")
        if status == "scored":
            response = record.get("raw_response")
            if (
                not isinstance(response, str)
                or not response.strip()
                or record.get("failure_type") is not None
            ):
                raise ValueError(f"Invalid scored Phi response at {example_id}/{arm}.")
            recomputed = example_score(task, response, answers[example_id])
            _require(
                isinstance(record.get("score"), (int, float))
                and math.isfinite(float(record["score"]))
                and abs(float(record["score"]) - recomputed) <= 1e-12,
                f"Phi RULER score drifted at {example_id}/{arm}.",
            )
            record["effective_score"] = recomputed
        else:
            _require(
                record.get("score") is None and isinstance(record.get("failure_type"), str),
                f"Invalid Phi failure accounting at {example_id}/{arm}.",
            )
            record["effective_score"] = 0.0
    return cell, records


def paired_bootstrap(differences: Iterable[float], *, seed: int, resamples: int) -> dict[str, Any]:
    values = np.asarray(tuple(differences), dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 0, "Paired bootstrap requires differences.")
    unique, counts = np.unique(values, return_counts=True)
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(len(values), counts / len(values), size=resamples)
    means = sampled @ unique / len(values)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "paired_examples": len(values),
        "mean_difference": float(values.mean()),
        "mean_difference_percentage_points": float(values.mean() * 100.0),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "paired_bootstrap_95_ci_percentage_points": [float(lower * 100.0), float(upper * 100.0)],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
    }


def exact_task_sign_flip(task_differences: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(tuple(task_differences), dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 0, "Exact sign flip requires task effects.")
    observed = abs(float(values.mean()))
    null = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 8
    return {
        "task_clusters": len(values),
        "exact_assignments": len(null),
        "two_sided_p": float(np.count_nonzero(np.abs(null) + tolerance >= observed) / len(null)),
    }


def analyze_pairs(
    native: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    _require(
        len(native) == len(candidate) and bool(native),
        "Phi arm pairing is incomplete.",
    )
    differences: list[float] = []
    by_cell: dict[tuple[int, str], list[float]] = defaultdict(list)
    failures = {ARMS[0]: 0, ARMS[1]: 0}
    kv: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: {ARMS[0]: 0, ARMS[1]: 0})
    measurable: dict[tuple[int, str], int] = defaultdict(int)
    for dense, compressed in zip(native, candidate, strict=True):
        example_id = dense["example_id"]
        _require(example_id == compressed["example_id"], "Phi input pairing drifted.")
        for field in (
            "raw_prompt_sha256",
            "input_token_ids_sha256",
            "exact_input_tokens",
            "generation_reserve_tokens",
            "task",
            "length_tokens",
        ):
            _require(dense.get(field) == compressed.get(field), f"Phi paired {field} drifted.")
        difference = float(compressed["effective_score"] - dense["effective_score"])
        differences.append(difference)
        cell = (dense["length_tokens"], dense["task"])
        by_cell[cell].append(difference)
        failures[ARMS[0]] += int(dense["status"] != "scored")
        failures[ARMS[1]] += int(compressed["status"] != "scored")
        if dense["hot_resident_bytes"] > 0 and compressed["hot_resident_bytes"] > 0:
            kv[cell][ARMS[0]] += dense["hot_resident_bytes"]
            kv[cell][ARMS[1]] += compressed["hot_resident_bytes"]
            measurable[cell] += 1

    task_length: list[dict[str, Any]] = [
        {
            "length_tokens": length,
            "task": task,
            "paired_examples": len(values),
            "mean_difference": float(np.mean(values)),
        }
        for (length, task), values in sorted(by_cell.items())
    ]
    by_task = [
        {
            "task": task,
            "paired_examples": sum(
                len(values) for (_length, name), values in by_cell.items() if name == task
            ),
            "mean_difference": float(
                np.mean(
                    [
                        value
                        for (_length, name), values in by_cell.items()
                        if name == task
                        for value in values
                    ]
                )
            ),
        }
        for task in TASKS
    ]
    length_rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for length in LENGTHS:
        effects = [
            float(row["mean_difference"]) for row in task_length if row["length_tokens"] == length
        ]
        exact = exact_task_sign_flip(effects)
        label = str(length)
        p_values[label] = exact["two_sided_p"]
        length_rows.append(
            {
                "length_tokens": length,
                "mean_difference": float(np.mean(effects)),
                **exact,
            }
        )
    adjusted = holm_bonferroni(p_values)
    for row in length_rows:
        row["holm_adjusted_p"] = adjusted[str(row["length_tokens"])]

    kv_rows: list[dict[str, Any]] = []
    for length, task in sorted(by_cell):
        denominator = kv[(length, task)][ARMS[0]]
        ratio = kv[(length, task)][ARMS[1]] / denominator if denominator > 0 else None
        kv_rows.append(
            {
                "length_tokens": length,
                "task": task,
                "measurable_pairs": measurable[(length, task)],
                "native_hot_resident_bytes_sum": kv[(length, task)][ARMS[0]],
                "candidate_hot_resident_bytes_sum": kv[(length, task)][ARMS[1]],
                "candidate_over_native_hot_resident_bytes": ratio,
            }
        )
    bootstrap = paired_bootstrap(differences, seed=bootstrap_seed, resamples=bootstrap_resamples)
    return {
        "overall": bootstrap,
        "task_by_length": task_length,
        "by_task": by_task,
        "by_length_exact_task_sign_flip": length_rows,
        "worst_task_length_regression": min(float(row["mean_difference"]) for row in task_length),
        "failure_rates": {arm: failures[arm] / len(native) for arm in ARMS},
        "failure_rate_increase": (failures[ARMS[1]] - failures[ARMS[0]]) / len(native),
        "accuracy_by_arm": {
            ARMS[0]: float(np.mean([row["effective_score"] for row in native])),
            ARMS[1]: float(np.mean([row["effective_score"] for row in candidate])),
        },
        "hot_resident_bytes_by_arm": {
            arm: {
                "sum_over_measurable_pairs": sum(row[arm] for row in kv.values()),
                "measurable_pairs": sum(measurable.values()),
            }
            for arm in ARMS
        },
        "realized_kv_by_task_length": kv_rows,
        "maximum_realized_kv_fraction": max(
            (
                row["candidate_over_native_hot_resident_bytes"]
                for row in kv_rows
                if row["candidate_over_native_hot_resident_bytes"] is not None
            ),
            default=None,
        ),
        "all_task_length_cells_have_measurable_kv": all(
            row["measurable_pairs"] > 0 for row in kv_rows
        ),
        "paired_record_digest": _canonical_digest(
            [
                {
                    "example_id": dense["example_id"],
                    "native_score": dense["effective_score"],
                    "candidate_score": compressed["effective_score"],
                    "native_status": dense["status"],
                    "candidate_status": compressed["status"],
                }
                for dense, compressed in zip(native, candidate, strict=True)
            ]
        ),
    }


def _group_row(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, label: dict[str, Any]
) -> dict[str, Any]:
    _require(bool(pairs), "Phi statistical cell is empty.")
    native_scores = np.asarray([pair[0]["effective_score"] for pair in pairs])
    candidate_scores = np.asarray([pair[1]["effective_score"] for pair in pairs])
    native_failures = sum(pair[0]["status"] != "scored" for pair in pairs)
    candidate_failures = sum(pair[1]["status"] != "scored" for pair in pairs)
    measurable = [
        (pair[0]["hot_resident_bytes"], pair[1]["hot_resident_bytes"])
        for pair in pairs
        if pair[0]["hot_resident_bytes"] > 0 and pair[1]["hot_resident_bytes"] > 0
    ]
    delta = candidate_scores - native_scores
    return {
        **label,
        "paired_examples": len(pairs),
        "native_accuracy_intention_to_treat": float(native_scores.mean()),
        "qwen_selected_accuracy_intention_to_treat": float(candidate_scores.mean()),
        "mean_difference": float(delta.mean()),
        "mean_difference_percentage_points": float(delta.mean() * 100.0),
        "native_failures": native_failures,
        "qwen_selected_failures": candidate_failures,
        "native_failure_rate": native_failures / len(pairs),
        "qwen_selected_failure_rate": candidate_failures / len(pairs),
        "failure_rate_difference": (candidate_failures - native_failures) / len(pairs),
        "measurable_kv_pairs": len(measurable),
        "realized_kv_fraction": (
            sum(candidate_bytes for _native_bytes, candidate_bytes in measurable)
            / sum(native_bytes for native_bytes, _candidate_bytes in measurable)
            if measurable
            else None
        ),
        "native_hot_resident_bytes_sum": sum(row[0] for row in measurable),
        "qwen_selected_hot_resident_bytes_sum": sum(row[1] for row in measurable),
    }


def summarize_pairs(
    native: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    _require(len(native) == len(candidate) == EXPECTED_EXAMPLES, "Paired record count drifted.")
    analysis = analyze_pairs(
        native,
        candidate,
        bootstrap_seed=manifest["statistics"]["paired_bootstrap_seed"],
        bootstrap_resamples=manifest["statistics"]["paired_bootstrap_resamples"],
    )
    pairs = list(zip(native, candidate, strict=True))
    task_length_rows = [
        _group_row(
            [
                pair
                for pair in pairs
                if pair[0]["length_tokens"] == length and pair[0]["task"] == task
            ],
            label={"length_tokens": length, "task": task},
        )
        for length in LENGTHS
        for task in TASKS
    ]
    length_rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        row = _group_row(
            [pair for pair in pairs if pair[0]["length_tokens"] == length],
            label={"length_tokens": length},
        )
        exact = next(
            value
            for value in analysis["by_length_exact_task_sign_flip"]
            if value["length_tokens"] == length
        )
        row.update(
            exact_task_cluster_sign_flip_two_sided_p=exact["two_sided_p"],
            task_clusters=exact["task_clusters"],
            exact_sign_assignments=exact["exact_assignments"],
            holm_adjusted_task_cluster_p=exact["holm_adjusted_p"],
        )
        length_rows.append(row)
    task_rows = [
        _group_row(
            [pair for pair in pairs if pair[0]["task"] == task],
            label={"task": task},
        )
        for task in TASKS
    ]
    overall = _group_row(pairs, label={})
    overall.update(analysis["overall"])
    worst = min(task_length_rows, key=lambda row: row["mean_difference"])
    measurable_cells = [row for row in task_length_rows if row["realized_kv_fraction"] is not None]
    maximum_kv_fraction = (
        max(float(row["realized_kv_fraction"]) for row in measurable_cells)
        if len(measurable_cells) == len(task_length_rows)
        else None
    )
    gate = manifest["transfer_gate"]
    checks = {
        "overall_mean_accuracy_difference": (
            overall["mean_difference"] >= gate["overall_mean_accuracy_difference_minimum"]
        ),
        "overall_paired_bootstrap_lower_bound": (
            overall["paired_bootstrap_95_ci"][0]
            >= gate["overall_paired_bootstrap_lower_bound_minimum"]
        ),
        "worst_task_length_regression": (
            worst["mean_difference"] >= gate["worst_task_length_regression_minimum"]
        ),
        "failure_rate_increase": (
            overall["failure_rate_difference"] <= gate["maximum_failure_rate_increase"]
        ),
        "maximum_realized_kv_fraction": (
            maximum_kv_fraction is not None
            and maximum_kv_fraction <= gate["maximum_realized_kv_fraction"]
        ),
    }
    return {
        "overall": overall,
        "by_length_with_exact_task_cluster_inference": length_rows,
        "by_task": task_rows,
        "by_task_length": task_length_rows,
        "worst_task_length": worst,
        "operational_failures_by_arm": {
            arm: dict(
                sorted(
                    Counter(
                        row["failure_type"] for row in records if row["status"] != "scored"
                    ).items()
                )
            )
            for arm, records in zip(ARMS, (native, candidate), strict=True)
        },
        "memory": {
            "aggregation": gate["realized_kv_fraction_aggregation"],
            "measurable_task_length_cells": len(measurable_cells),
            "required_task_length_cells": len(task_length_rows),
            "maximum_task_length_realized_kv_fraction": maximum_kv_fraction,
        },
        "transfer_gate": {
            "passed": all(checks.values()),
            "all_required": gate["all_required"],
            "checks": checks,
            "thresholds": gate,
            "interpretation": (
                "bounded cross-family transfer supported"
                if all(checks.values())
                else "negative or bounded transfer result retained without Phi reselection"
            ),
        },
        "paired_record_digest": analysis["paired_record_digest"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen Phi-4-mini RULER transfer.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler/data"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler/results"
        ),
    )
    parser.add_argument(
        "--fixed-selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Cross-family RULER summarization requires a clean source tree.")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    selection = json.loads(args.fixed_selection.read_text())
    manifest_digest = sha256(args.manifest)
    selection_digest = sha256(args.fixed_selection)
    runner_path = Path(__file__).with_name("run_p3_cross_family_ruler.py")
    generator_path = Path(__file__).with_name("prepare_p3_cross_family_ruler_dataset.py")
    dataset_manifests, dataset_digest_set = load_dataset_contracts(
        args.dataset_root,
        manifest_digest=manifest_digest,
        generator_digest=sha256(generator_path),
    )
    answers = _answer_map(dataset_manifests)
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for arm in ARMS:
        loaded[arm] = _load_arm(
            args.result_root,
            arm=arm,
            manifest_digest=manifest_digest,
            runner_digest=sha256(runner_path),
            dataset_digest_set=dataset_digest_set,
            selection_digest=selection_digest,
            expected_config=arm_config(arm, selection, selection_digest),
            answers=answers,
            maximum_context=manifest["model"]["maximum_supported_context_tokens"],
            model_digest_set=manifest["model"]["snapshot_digest_set_sha256"],
        )
    statistics = manifest["statistics"]
    analysis = analyze_pairs(
        loaded[ARMS[0]][1],
        loaded[ARMS[1]][1],
        bootstrap_seed=statistics["paired_bootstrap_seed"],
        bootstrap_resamples=statistics["paired_bootstrap_resamples"],
    )
    gate = manifest["transfer_gate"]
    checks = {
        "overall_mean_accuracy": (
            analysis["overall"]["mean_difference"]
            >= gate["overall_mean_accuracy_difference_minimum"]
        ),
        "overall_bootstrap_lower_bound": (
            analysis["overall"]["paired_bootstrap_95_ci"][0]
            >= gate["overall_paired_bootstrap_lower_bound_minimum"]
        ),
        "worst_task_length_regression": (
            analysis["worst_task_length_regression"] >= gate["worst_task_length_regression_minimum"]
        ),
        "failure_rate_increase": (
            analysis["failure_rate_increase"] <= gate["maximum_failure_rate_increase"]
        ),
        "realized_kv_fraction": (
            analysis["all_task_length_cells_have_measurable_kv"] is True
            and analysis["maximum_realized_kv_fraction"] is not None
            and analysis["maximum_realized_kv_fraction"] <= gate["maximum_realized_kv_fraction"]
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "source": {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "dirty": False,
            "implementation_sha256": sha256(Path(__file__)),
        },
        "experiment_manifest": {"path": str(args.manifest), "sha256": manifest_digest},
        "dataset_manifest_digest_set_sha256": dataset_digest_set,
        "arm_cells": {
            arm: {
                "path": str(args.result_root / arm / "cell.json"),
                "sha256": sha256(args.result_root / arm / "cell.json"),
            }
            for arm in ARMS
        },
        "audit": {
            "required_arms_terminal": True,
            "predictions_per_arm": EXPECTED_EXAMPLES,
            "total_predictions": EXPECTED_EXAMPLES * len(ARMS),
            "paired_examples": EXPECTED_EXAMPLES,
            "all_raw_record_digests_verified": True,
            "all_runtime_kvpress_bindings_verified": True,
            "all_dependency_digests_verified": True,
            "all_record_revisions_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "exact_input_pairing_verified": True,
            "exact_token_contract_verified": True,
            "failure_accounting_complete": True,
            "physical_kv_measurements_verified": True,
            "phi_specific_reselection": False,
            "outcome_dependent_execution": False,
            "task_length_cells": len(LENGTHS) * len(TASKS),
            "exact_task_sign_flip_assignments_per_length": 2 ** len(TASKS),
            "paired_bootstrap_resamples": statistics["paired_bootstrap_resamples"],
            "paired_bootstrap_seed": statistics["paired_bootstrap_seed"],
        },
        "paired_qwen_selected_minus_native": analysis,
        "transfer_gate": {
            "all_required": gate["all_required"],
            "checks": checks,
            "passed": all(checks.values()),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total_predictions": EXPECTED_EXAMPLES * len(ARMS),
                "transfer_gate_passed": payload["transfer_gate"]["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
