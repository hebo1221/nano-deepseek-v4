from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from p3_source_provenance import verify_git_implementation, verify_runtime_kvpress_binding
from prepare_p3_natural_ruler_dataset import LENGTHS, SAMPLES_PER_TASK
from prepare_p3_ruler_dataset import TASKS
from run_p3_mrcr import atomic_json
from run_p3_natural_ruler import (
    ADAPTIVE_QUOTA_ARMS,
    EXPECTED_EXAMPLES,
    compatibility_arm_config,
    load_dataset_contracts,
)
from run_p3_ruler_matrix import example_score, load_dataset
from summarize_p2_core_matrix import holm_bonferroni
from summarize_p3_cross_family_ruler import exact_task_sign_flip, paired_bootstrap
from validate_p3_natural_adaptive_quota_manifest import validate_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-natural-adaptive-quota-ruler-audit-v1"
RUNNER = Path("research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values: Sequence[float | int]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "observations": len(array),
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


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
            _require(example_id not in answers, f"Duplicate RULER example: {example_id}.")
            answers[example_id] = references
    _require(len(answers) == EXPECTED_EXAMPLES, "Adaptive RULER answer grid drifted.")
    return answers


def verify_quota_audit(
    audit: Any,
    *,
    arm: str,
    protected_start: int = 0,
    protected_end: int = 4,
    layer_count: int = 36,
    adaptive_arm: str = "natural-adaptive-quota+pins",
) -> dict[str, Any]:
    _require(isinstance(audit, dict), f"Missing quota audit for {arm}.")
    layers = audit.get("layers")
    _require(
        isinstance(layers, list)
        and len(layers) == layer_count
        and [row.get("layer_index") for row in layers] == list(range(layer_count)),
        f"Incomplete or unordered layer quota audit for {arm}.",
    )
    input_tokens = {row.get("input_tokens") for row in layers}
    _require(
        len(input_tokens) == 1 and all(type(value) is int and value > 0 for value in input_tokens),
        f"Layer input-token contract drifted for {arm}.",
    )
    kept = [row.get("kept_tokens") for row in layers]
    _require(
        all(type(value) is int and value >= protected_end - protected_start for value in kept),
        f"Invalid kept-token quota for {arm}.",
    )
    context_tokens = int(next(iter(input_tokens)))
    fixed_per_layer = int(context_tokens * 0.5)
    target_total = fixed_per_layer * layer_count
    observed_total = sum(int(value) for value in kept)
    if arm == "fixed+pins":
        _require(
            audit.get("same_budget_verified") is True
            and audit.get("protected_start") == protected_start
            and audit.get("protected_end") == protected_end
            and all(value == fixed_per_layer for value in kept),
            "Fixed+pins quota audit drifted.",
        )
    else:
        _require(
            arm == adaptive_arm
            and audit.get("same_global_budget_verified") is True
            and audit.get("causal_layer_order_verified") is True
            and audit.get("compatibility_arm") is True
            and audit.get("synthetic_controller_unchanged_transfer") is False
            and audit.get("protected_start") == protected_start
            and audit.get("protected_end") == protected_end
            and audit.get("fixed_comparator_kept_tokens_per_layer") == fixed_per_layer
            and audit.get("target_total_kept_tokens") == target_total
            and audit.get("observed_total_kept_tokens") == target_total
            and audit.get("max_adjustment_fraction") == 0.25
            and layers[-1].get("remaining_global_budget") == 0,
            "Adaptive+pins global quota audit drifted.",
        )
        for row in layers:
            _require(
                type(row.get("controller_time_ns")) is int
                and row["controller_time_ns"] >= 0
                and isinstance(row.get("score_concentration"), (int, float))
                and math.isfinite(float(row["score_concentration"]))
                and 0.0 <= row["score_concentration"] <= 1.0
                and row["feasible_min"] <= row["kept_tokens"] <= row["feasible_max"],
                "Adaptive controller trace drifted.",
            )
    _require(observed_total == target_total, f"Global kept-token budget mismatch for {arm}.")
    return {
        "context_tokens": context_tokens,
        "fixed_kept_tokens_per_layer": fixed_per_layer,
        "target_total_kept_tokens": target_total,
        "observed_total_kept_tokens": observed_total,
        "controller_time_ns": sum(int(row.get("controller_time_ns", 0)) for row in layers),
        "layer_kept_tokens": [int(value) for value in kept],
        "layer_controller_time_ns": [int(row.get("controller_time_ns", 0)) for row in layers],
        "layer_score_concentration": [
            float(row["score_concentration"])
            for row in layers
            if row.get("score_concentration") is not None
        ],
        "quota_min": min(int(value) for value in kept),
        "quota_max": max(int(value) for value in kept),
    }


def _load_arm(
    root: Path,
    *,
    arm: str,
    natural_manifest_digest: str,
    compatibility_manifest_digest: str,
    dataset_digest_set: str,
    selection: dict[str, Any],
    selection_digest: str,
    answers: dict[str, list[str]],
    expected_revisions: dict[str, str],
    model_digest_set: str,
    maximum_context_tokens: int,
    allowed_failures: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / arm / "cell.json"
    _require(path.is_file(), f"Missing adaptive RULER cell: {path}.")
    cell = json.loads(path.read_text())
    identity = cell.get("run_identity", {})
    expected_config = compatibility_arm_config(arm, selection, selection_digest)
    _require(
        cell.get("experiment_id") == "p3-natural-benchmark-arm-cell-v1"
        and cell.get("benchmark") == "RULER"
        and cell.get("arm") == arm
        and cell.get("cohort") == "adaptive-quota"
        and cell.get("status") == "terminal"
        and identity.get("cohort") == "adaptive-quota",
        f"Adaptive RULER cell identity drifted for {arm}.",
    )
    verified_source = verify_git_implementation(
        cell.get("source"), expected_path=str(RUNNER), label=f"adaptive RULER {arm}"
    )
    _require(
        cell.get("source", {}).get("dirty") is False
        and identity.get("source_commit") == verified_source["commit"]
        and identity.get("implementation_sha256") == verified_source["implementation_sha256"]
        and cell.get("experiment_manifest", {}).get("sha256") == natural_manifest_digest
        and identity.get("manifest_sha256") == natural_manifest_digest
        and cell.get("adaptive_quota_manifest", {}).get("sha256") == compatibility_manifest_digest
        and identity.get("adaptive_quota_manifest_sha256") == compatibility_manifest_digest
        and identity.get("dataset_manifest_digest_set_sha256") == dataset_digest_set
        and cell.get("benchmark_dataset_digest_set_sha256") == dataset_digest_set
        and identity.get("fixed_selection_sha256") == selection_digest
        and identity.get("arm_config") == expected_config
        and identity.get("model_snapshot_digest_set_sha256") == model_digest_set
        and cell.get("model_snapshot_digest_set_sha256") == model_digest_set
        and identity.get("seed") == 42,
        f"Adaptive RULER provenance drifted for {arm}.",
    )
    verify_runtime_kvpress_binding(cell.get("environment", {}).get("kvpress_binding"))
    decision = cell.get("p3_sequence_decision", {})
    dependencies = decision.get("dependencies", {})
    _require(
        set(dependencies)
        == {
            "primary_core",
            "nine_seed_core",
            "primary_causal",
            "nine_seed_causal",
            "fixed_selection",
        },
        f"Adaptive RULER sequence dependencies drifted for {arm}.",
    )
    for label, metadata in dependencies.items():
        dependency = Path(metadata.get("path", ""))
        _require(
            dependency.is_file() and metadata.get("sha256") == sha256(dependency),
            f"Adaptive RULER {label} dependency drifted for {arm}.",
        )
    _require(
        cell.get("causal_gate", {}).get("sha256") == dependencies["primary_causal"]["sha256"],
        f"Adaptive RULER primary causal dependency drifted for {arm}.",
    )
    for label, metadata in (
        ("dataset inventory", cell.get("dataset_inventory")),
        ("fixed selection", cell.get("fixed_baseline_selection")),
    ):
        _require(isinstance(metadata, dict), f"Missing adaptive RULER {label}.")
        dependency = Path(metadata.get("path", ""))
        _require(
            dependency.is_file() and metadata.get("sha256") == sha256(dependency),
            f"Adaptive RULER {label} drifted for {arm}.",
        )
    _require(
        cell["fixed_baseline_selection"]["sha256"] == selection_digest,
        f"Adaptive RULER fixed selection identity drifted for {arm}.",
    )
    raw = cell.get("raw_records", {})
    raw_path = Path(raw.get("path", ""))
    _require(
        raw_path.is_file() and raw.get("sha256") == sha256(raw_path),
        f"Adaptive RULER raw records drifted for {arm}.",
    )
    records = _read_jsonl(raw_path)
    _require(len(records) == EXPECTED_EXAMPLES, f"Adaptive RULER record count drifted for {arm}.")
    expected_ids = list(answers)
    _require(
        [row.get("example_id") for row in records] == expected_ids,
        f"Adaptive RULER order drifted for {arm}.",
    )
    for row in records:
        example_id = row["example_id"]
        length_text, task, _index = example_id.split(":", 2)
        _require(
            row.get("benchmark") == "RULER"
            and row.get("arm") == arm
            and row.get("length_tokens") == int(length_text)
            and row.get("task") == task
            and row.get("arm_config") == expected_config
            and row.get("revisions") == expected_revisions
            and row.get("status") in {"scored", "failure"},
            f"Adaptive RULER coordinates drifted at {example_id}/{arm}.",
        )
        for field in ("raw_prompt_sha256", "input_token_ids_sha256"):
            value = row.get(field)
            _require(isinstance(value, str) and len(value) == 64, f"Invalid {field}.")
        _require(
            type(row.get("exact_input_tokens")) is int
            and row["exact_input_tokens"] > 0
            and type(row.get("generation_reserve_tokens")) is int
            and row["generation_reserve_tokens"] > 0
            and row["exact_input_tokens"] + row["generation_reserve_tokens"] <= row["length_tokens"]
            and row["length_tokens"] <= maximum_context_tokens
            and type(row.get("hot_resident_bytes")) is int
            and row["hot_resident_bytes"] >= 0
            and type(row.get("peak_hbm_bytes")) is int
            and row["peak_hbm_bytes"] >= 0
            and isinstance(row.get("latency_ms"), (int, float))
            and math.isfinite(float(row["latency_ms"]))
            and row["latency_ms"] >= 0,
            f"Adaptive RULER measurements drifted at {example_id}/{arm}.",
        )
        if row["status"] == "scored":
            response = row.get("raw_response")
            _require(isinstance(response, str) and bool(response.strip()), "Missing response.")
            assert isinstance(response, str)
            recomputed = example_score(task, response, answers[example_id])
            _require(
                isinstance(row.get("score"), (int, float))
                and abs(float(row["score"]) - recomputed) <= 1e-12,
                f"Adaptive RULER score drifted at {example_id}/{arm}.",
            )
            row["effective_score"] = recomputed
        else:
            _require(
                row.get("score") is None
                and isinstance(row.get("failure_type"), str)
                and row["failure_type"] in allowed_failures,
                f"Adaptive RULER failure drifted at {example_id}/{arm}.",
            )
            row["effective_score"] = 0.0
        if row.get("quota_physical_audit") is not None:
            row["verified_quota"] = verify_quota_audit(row["quota_physical_audit"], arm=arm)
        else:
            _require(
                row["hot_resident_bytes"] == 0 and row["status"] == "failure",
                f"Successful prefill lacks quota audit at {example_id}/{arm}.",
            )
            row["verified_quota"] = None
    return cell, records


def _measurement_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], metric: str
) -> dict[str, Any]:
    fixed = np.asarray([float(pair[0][metric]) for pair in pairs])
    adaptive = np.asarray([float(pair[1][metric]) for pair in pairs])
    return {
        "paired_examples": len(pairs),
        "fixed_mean": float(fixed.mean()),
        "adaptive_mean": float(adaptive.mean()),
        "mean_paired_difference": float((adaptive - fixed).mean()),
        "ratio_of_means": float(adaptive.mean() / fixed.mean()) if fixed.mean() > 0 else None,
    }


def analyze_pairs(
    fixed: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    arms: tuple[str, str] = ADAPTIVE_QUOTA_ARMS,
    layer_count: int = 36,
    lengths: tuple[int, ...] = LENGTHS,
    tasks: tuple[str, ...] = TASKS,
    expected_examples: int = EXPECTED_EXAMPLES,
) -> dict[str, Any]:
    _require(
        len(fixed) == len(adaptive) == expected_examples,
        "Adaptive pairing is incomplete.",
    )
    pairs = list(zip(fixed, adaptive, strict=True))
    differences: list[float] = []
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    audited_pairs = 0
    hot_relative_differences: list[float] = []
    controller_times: list[int] = []
    quota_by_layer: list[list[int]] = [[] for _ in range(layer_count)]
    concentration_by_layer: list[list[float]] = [[] for _ in range(layer_count)]
    controller_time_by_layer: list[list[int]] = [[] for _ in range(layer_count)]
    for fixed_row, adaptive_row in pairs:
        _require(fixed_row["example_id"] == adaptive_row["example_id"], "Pair identity drifted.")
        for field in (
            "raw_prompt_sha256",
            "input_token_ids_sha256",
            "exact_input_tokens",
            "generation_reserve_tokens",
            "length_tokens",
            "task",
        ):
            _require(fixed_row.get(field) == adaptive_row.get(field), f"Paired {field} drifted.")
        difference = float(adaptive_row["effective_score"] - fixed_row["effective_score"])
        differences.append(difference)
        cells[(fixed_row["length_tokens"], fixed_row["task"])].append(difference)
        fixed_quota = fixed_row["verified_quota"]
        adaptive_quota = adaptive_row["verified_quota"]
        if fixed_quota is not None and adaptive_quota is not None:
            _require(
                fixed_quota["context_tokens"] == adaptive_quota["context_tokens"]
                and fixed_quota["target_total_kept_tokens"]
                == adaptive_quota["target_total_kept_tokens"]
                and fixed_quota["observed_total_kept_tokens"]
                == adaptive_quota["observed_total_kept_tokens"],
                "Paired global quota drifted.",
            )
            audited_pairs += 1
            controller_times.append(adaptive_quota["controller_time_ns"])
            for layer, value in enumerate(adaptive_quota["layer_kept_tokens"]):
                quota_by_layer[layer].append(value)
            for layer, value in enumerate(adaptive_quota["layer_score_concentration"]):
                concentration_by_layer[layer].append(value)
            for layer, value in enumerate(adaptive_quota["layer_controller_time_ns"]):
                controller_time_by_layer[layer].append(value)
            fixed_hot = fixed_row["hot_resident_bytes"]
            adaptive_hot = adaptive_row["hot_resident_bytes"]
            if fixed_hot > 0:
                hot_relative_differences.append(abs(adaptive_hot - fixed_hot) / fixed_hot)

    task_length: list[dict[str, Any]] = [
        {
            "length_tokens": length,
            "task": task,
            "paired_examples": len(cells[(length, task)]),
            "mean_difference": float(np.mean(cells[(length, task)])),
        }
        for length in lengths
        for task in tasks
    ]
    length_rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for length in lengths:
        task_effects = [
            float(row["mean_difference"]) for row in task_length if row["length_tokens"] == length
        ]
        exact = exact_task_sign_flip(task_effects)
        label = str(length)
        raw_p[label] = exact["two_sided_p"]
        length_differences = [
            value
            for pair, value in zip(pairs, differences, strict=True)
            if pair[0]["length_tokens"] == length
        ]
        length_rows.append(
            {
                "length_tokens": length,
                **paired_bootstrap(
                    length_differences,
                    seed=manifest["statistics"]["paired_bootstrap_seed"] + length,
                    resamples=manifest["statistics"]["paired_bootstrap_resamples"],
                ),
                "task_clusters": exact["task_clusters"],
                "exact_sign_assignments": exact["exact_assignments"],
                "exact_task_sign_flip_two_sided_p": exact["two_sided_p"],
            }
        )
    adjusted = holm_bonferroni(raw_p)
    for row in length_rows:
        row["holm_adjusted_task_sign_flip_p"] = adjusted[str(row["length_tokens"])]

    overall = paired_bootstrap(
        differences,
        seed=manifest["statistics"]["paired_bootstrap_seed"],
        resamples=manifest["statistics"]["paired_bootstrap_resamples"],
    )
    failure_rates = {
        arm: sum(row["status"] != "scored" for row in records) / expected_examples
        for arm, records in zip(arms, (fixed, adaptive), strict=True)
    }
    failure_increase = failure_rates[arms[1]] - failure_rates[arms[0]]
    worst = min(task_length, key=lambda row: float(row["mean_difference"]))
    nonnegative_lengths = sum(row["mean_difference"] >= 0.0 for row in length_rows)
    max_hot_difference = max(hot_relative_differences, default=None)
    successful_prefill_pairs = sum(
        pair[0]["hot_resident_bytes"] > 0 and pair[1]["hot_resident_bytes"] > 0 for pair in pairs
    )
    gate = manifest["confirmation_gate"]
    checks = {
        "overall_accuracy_difference": overall["mean_difference"]
        >= gate["overall_accuracy_difference_minimum"],
        "paired_bootstrap_lower_bound": overall["paired_bootstrap_95_ci"][0]
        >= gate["paired_bootstrap_lower_bound_minimum"],
        "nonnegative_length_count": nonnegative_lengths >= gate["nonnegative_length_count_minimum"],
        "worst_task_length_regression": worst["mean_difference"]
        >= gate["worst_task_length_regression_minimum"],
        "failure_rate_increase": failure_increase <= gate["maximum_failure_rate_increase"],
        "all_successful_prefills_quota_audited": audited_pairs == successful_prefill_pairs,
        "global_kept_token_relative_error": True,
        "hot_resident_byte_relative_difference": max_hot_difference is not None
        and max_hot_difference <= gate["maximum_hot_resident_byte_relative_difference"],
    }
    return {
        "overall": overall,
        "accuracy_by_arm": {
            arm: float(np.mean([row["effective_score"] for row in records]))
            for arm, records in zip(arms, (fixed, adaptive), strict=True)
        },
        "by_length": length_rows,
        "by_task_length": task_length,
        "worst_task_length": worst,
        "nonnegative_length_count": nonnegative_lengths,
        "failure_rates": failure_rates,
        "failure_rate_increase": failure_increase,
        "failures_by_arm": {
            arm: dict(
                sorted(
                    Counter(
                        row.get("failure_type") for row in records if row["status"] != "scored"
                    ).items()
                )
            )
            for arm, records in zip(arms, (fixed, adaptive), strict=True)
        },
        "measurements": {
            metric: _measurement_summary(pairs, metric)
            for metric in ("latency_ms", "peak_hbm_bytes", "hot_resident_bytes")
        },
        "quota_audit": {
            "audited_pairs": audited_pairs,
            "successful_prefill_pairs": successful_prefill_pairs,
            "all_successful_prefills_quota_audited": audited_pairs == successful_prefill_pairs,
            "same_global_token_budget_verified": True,
            "maximum_hot_resident_byte_relative_difference": max_hot_difference,
            "adaptive_controller_time_ns": {
                "observations": len(controller_times),
                "mean": float(np.mean(controller_times)) if controller_times else None,
                "p95": float(np.quantile(controller_times, 0.95)) if controller_times else None,
                "p99": float(np.quantile(controller_times, 0.99)) if controller_times else None,
            },
            "adaptive_layer_quota_distribution": _distribution(
                [value for values in quota_by_layer for value in values]
            ),
            "adaptive_score_concentration_distribution": _distribution(
                [value for values in concentration_by_layer for value in values]
            ),
            "adaptive_per_layer_distributions": [
                {
                    "layer_index": layer,
                    "kept_tokens": _distribution(quota_by_layer[layer]),
                    "score_concentration": _distribution(concentration_by_layer[layer]),
                    "controller_time_ns": _distribution(controller_time_by_layer[layer]),
                }
                for layer in range(layer_count)
            ],
        },
        "confirmation_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": gate,
            "classification": "success" if all(checks.values()) else "negative-result",
        },
        "paired_record_digest": _canonical_digest(
            [
                {
                    "example_id": pair[0]["example_id"],
                    "fixed_score": pair[0]["effective_score"],
                    "adaptive_score": pair[1]["effective_score"],
                    "fixed_status": pair[0]["status"],
                    "adaptive_status": pair[1]["status"],
                }
                for pair in pairs
            ]
        ),
    }


def summarize(
    *,
    compatibility_manifest_path: Path,
    natural_manifest_path: Path,
    dataset_root: Path,
    selection_path: Path,
    result_root: Path,
) -> dict[str, Any]:
    compatibility_manifest = json.loads(compatibility_manifest_path.read_text())
    validate_manifest(compatibility_manifest)
    natural_manifest = json.loads(natural_manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    compatibility_digest = sha256(compatibility_manifest_path)
    natural_digest = sha256(natural_manifest_path)
    selection_digest = sha256(selection_path)
    dataset_manifests, dataset_digest_set = load_dataset_contracts(
        dataset_root,
        natural_manifest_digest=natural_digest,
        generator_digest=sha256(Path(__file__).with_name("prepare_p3_natural_ruler_dataset.py")),
    )
    answers = _answer_map(dataset_manifests)
    expected_revisions = {
        "model_revision": natural_manifest["model"]["revision"],
        "dataset_revision": natural_manifest["benchmarks"]["RULER"]["upstream_revision"],
        "code_revision": natural_manifest["benchmarks"]["RULER"]["upstream_revision"],
        "scorer_sha256": sha256(Path(__file__).with_name("run_p3_ruler_matrix.py")),
        "official_scorer_sha256": natural_manifest["benchmarks"]["RULER"]["scorer"]["sha256"],
    }
    model_digest_set = natural_manifest["model"]["snapshot_digest_set_sha256"]
    maximum_context_tokens = natural_manifest["model"]["maximum_supported_context_tokens"]
    allowed_failures = set(natural_manifest["common_protocol"]["failure_accounting"])
    cells: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for arm in ADAPTIVE_QUOTA_ARMS:
        cells[arm], records[arm] = _load_arm(
            result_root,
            arm=arm,
            natural_manifest_digest=natural_digest,
            compatibility_manifest_digest=compatibility_digest,
            dataset_digest_set=dataset_digest_set,
            selection=selection,
            selection_digest=selection_digest,
            answers=answers,
            expected_revisions=expected_revisions,
            model_digest_set=model_digest_set,
            maximum_context_tokens=maximum_context_tokens,
            allowed_failures=allowed_failures,
        )
    analysis = analyze_pairs(
        records[ADAPTIVE_QUOTA_ARMS[0]], records[ADAPTIVE_QUOTA_ARMS[1]], compatibility_manifest
    )
    dependencies = cells[ADAPTIVE_QUOTA_ARMS[0]]["p3_sequence_decision"]["dependencies"]
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "classification": analysis["confirmation_gate"]["classification"],
        "claim_boundary": compatibility_manifest["claim_boundary"],
        "coverage": {
            "arms": list(ADAPTIVE_QUOTA_ARMS),
            "lengths_tokens": list(LENGTHS),
            "tasks": len(TASKS),
            "samples_per_task_length": SAMPLES_PER_TASK,
            "predictions_per_arm": EXPECTED_EXAMPLES,
            "paired_predictions_total": EXPECTED_EXAMPLES * 2,
        },
        "analysis": analysis,
        "audit": {
            "required_arms_terminal": True,
            "terminal_arms": 2,
            "expected_examples_per_arm": EXPECTED_EXAMPLES,
            "predictions_per_arm": EXPECTED_EXAMPLES,
            "total_predictions": EXPECTED_EXAMPLES * 2,
            "paired_examples": EXPECTED_EXAMPLES,
            "raw_scores_recomputed": True,
            "all_scores_recomputed_from_raw_response": True,
            "exact_example_pairing_verified": True,
            "exact_input_pairing_verified": True,
            "all_cell_and_raw_digests_verified": True,
            "all_raw_record_digests_verified": True,
            "all_raw_records_verified": True,
            "sequence_dependencies_verified": True,
            "all_dependency_digests_verified": True,
            "runtime_kvpress_binding_verified": True,
            "all_runtime_kvpress_bindings_verified": True,
            "quota_physical_audits_verified": True,
            "same_global_token_budget_verified": True,
            "causal_layer_order_verified": True,
            "failure_accounting_complete": True,
            "record_revision_provenance_verified": True,
            "model_snapshot_digest_set_verified": True,
            "operational_failure_vocabulary_verified": True,
            "task_length_cells": len(LENGTHS) * len(TASKS),
            "exact_task_sign_flip_assignments_per_length": 2 ** len(TASKS),
            "paired_bootstrap_resamples": compatibility_manifest["statistics"][
                "paired_bootstrap_resamples"
            ],
            "paired_bootstrap_seed": compatibility_manifest["statistics"]["paired_bootstrap_seed"],
            "synthetic_controller_unchanged_transfer": False,
            "outcome_dependent_execution": False,
            "natural_manifest_sha256": natural_digest,
            "compatibility_manifest_sha256": compatibility_digest,
            "dataset_manifest_digest_set_sha256": dataset_digest_set,
            "fixed_selection_sha256": selection_digest,
            "sequence_dependencies": dependencies,
        },
        "arm_cells": {
            arm: {
                "path": str(result_root / arm / "cell.json"),
                "sha256": sha256(result_root / arm / "cell.json"),
            }
            for arm in ADAPTIVE_QUOTA_ARMS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit natural adaptive-quota RULER results.")
    parser.add_argument(
        "--compatibility-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-ruler-v1.json"
        ),
    )
    parser.add_argument(
        "--natural-manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/ruler-qwen3-4b/data"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/ruler-qwen3-4b"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/ruler-qwen3-4b.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Adaptive RULER summarization requires a clean source tree.")
    payload = summarize(
        compatibility_manifest_path=args.compatibility_manifest,
        natural_manifest_path=args.natural_manifest,
        dataset_root=args.dataset_root,
        selection_path=args.selection,
        result_root=args.result_root,
    )
    payload["source"] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": False,
        "implementation_sha256": sha256(Path(__file__)),
    }
    payload["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {"output": str(args.output), "classification": payload["classification"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
