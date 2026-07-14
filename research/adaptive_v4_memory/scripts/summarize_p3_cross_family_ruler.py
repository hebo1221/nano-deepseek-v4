from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from run_p3_cross_family_ruler import (
    ARMS,
    CELL_EXPERIMENT_ID,
    EXPECTED_EXAMPLES,
    LENGTHS,
    MODEL_REVISION,
    SAMPLES_PER_TASK,
    TASKS,
    expected_example_ids,
)
from summarize_p2_core_matrix import holm_bonferroni
from validate_p3_cross_family_ruler_manifest import validate_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-cross-family-ruler-transfer-audit-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _score(record: dict[str, Any]) -> float:
    return float(record["score"]) if record["status"] == "scored" else 0.0


def _valid_record(record: dict[str, Any], *, arm: str) -> bool:
    status = record.get("status")
    score = record.get("score")
    base = bool(
        record.get("arm") == arm
        and record.get("benchmark") == "RULER"
        and record.get("length_tokens") in LENGTHS
        and record.get("task") in TASKS
        and isinstance(record.get("example_id"), str)
        and type(record.get("exact_input_tokens")) is int
        and record["exact_input_tokens"] > 0
        and type(record.get("generation_reserve_tokens")) is int
        and record["generation_reserve_tokens"] > 0
        and record.get("silently_truncated") is False
        and isinstance(record.get("raw_prompt_sha256"), str)
        and len(record["raw_prompt_sha256"]) == 64
        and isinstance(record.get("input_token_ids_sha256"), str)
        and len(record["input_token_ids_sha256"]) == 64
        and type(record.get("hot_resident_bytes")) is int
        and record["hot_resident_bytes"] >= 0
        and type(record.get("peak_hbm_bytes")) is int
        and record["peak_hbm_bytes"] >= 0
        and isinstance(record.get("latency_ms"), (int, float))
        and math.isfinite(float(record["latency_ms"]))
        and record["latency_ms"] >= 0
    )
    if status == "scored":
        return bool(
            base
            and isinstance(score, (int, float))
            and math.isfinite(float(score))
            and 0.0 <= score <= 1.0
            and record.get("failure_type") is None
            and isinstance(record.get("raw_response"), str)
            and bool(record["raw_response"].strip())
        )
    return bool(
        base
        and status == "failure"
        and score is None
        and isinstance(record.get("failure_type"), str)
        and bool(record["failure_type"])
    )


def _load_records(path: Path, *, arm: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    _require(len(records) == EXPECTED_EXAMPLES, f"{arm} record count drifted.")
    _require(
        all(_valid_record(record, arm=arm) for record in records),
        f"{arm} raw record schema drifted.",
    )
    ids = [record["example_id"] for record in records]
    _require(ids == expected_example_ids(), f"{arm} example order drifted.")
    _require(len(set(ids)) == len(ids), f"{arm} example identities are not unique.")
    return records


def _verify_dependency(metadata: dict[str, Any], *, name: str) -> None:
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == sha256(path),
        f"Cross-family {name} dependency drifted: {path}.",
    )


def load_cell(
    path: Path,
    *,
    arm: str,
    manifest_digest: str,
    runner_digest: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(path.is_file(), f"Missing cross-family arm cell: {path}.")
    cell = json.loads(path.read_text())
    identity = cell.get("run_identity", {})
    _require(
        cell.get("experiment_id") == CELL_EXPERIMENT_ID
        and cell.get("benchmark") == "RULER"
        and cell.get("arm") == arm
        and cell.get("status") == "terminal"
        and cell.get("source", {}).get("dirty") is False
        and cell.get("source", {}).get("implementation_sha256") == runner_digest
        and identity.get("implementation_sha256") == runner_digest
        and identity.get("manifest_sha256") == manifest_digest
        and identity.get("model_snapshot_digest_set_sha256")
        == cell.get("model_snapshot_digest_set_sha256")
        and cell.get("experiment_manifest", {}).get("sha256") == manifest_digest,
        f"Cross-family {arm} cell provenance drifted.",
    )
    dependencies = identity.get("sequence_gate_dependencies", {})
    _require(
        set(dependencies)
        == {"primary_core", "primary_causal", "nine_seed_causal", "fixed_selection"},
        f"Cross-family {arm} sequence dependencies drifted.",
    )
    for name, metadata in dependencies.items():
        _verify_dependency(metadata, name=name)
    _require(
        cell.get("sequence_gate", {}).get("dependencies") == dependencies,
        f"Cross-family {arm} gate identity drifted.",
    )
    records_path = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records_path.is_file() and cell["raw_records"].get("sha256") == sha256(records_path),
        f"Cross-family {arm} raw records drifted.",
    )
    return cell, _load_records(records_path, arm=arm)


def paired_bootstrap(differences: np.ndarray, *, manifest: dict[str, Any]) -> dict[str, Any]:
    statistics = manifest["statistics"]
    resamples = int(statistics["paired_bootstrap_resamples"])
    seed = int(statistics["paired_bootstrap_seed"])
    confidence = float(statistics["paired_bootstrap_confidence"])
    _require(differences.ndim == 1 and differences.size > 0, "Paired contrast is empty.")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    probabilities = np.full(differences.size, 1.0 / differences.size)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        counts = rng.multinomial(differences.size, probabilities, size=stop - start)
        means[start:stop] = (counts @ differences) / differences.size
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    return {
        "paired_examples": int(differences.size),
        "mean_difference": float(differences.mean()),
        "mean_difference_percentage_points": float(differences.mean() * 100.0),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "paired_bootstrap_95_ci_percentage_points": [
            float(lower * 100.0),
            float(upper * 100.0),
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "confidence_level": confidence,
    }


def exact_task_sign_flip(task_means: list[float]) -> float:
    values = np.asarray(task_means, dtype=np.float64)
    _require(values.shape == (len(TASKS),), "Exact task sign-flip requires 13 task means.")
    assignments = np.arange(1 << len(TASKS), dtype=np.uint16)[:, None]
    bit_positions = np.arange(len(TASKS), dtype=np.uint16)[None, :]
    signs = 2.0 * ((assignments >> bit_positions) & 1).astype(np.float64) - 1.0
    null_means = (signs @ values) / len(values)
    observed = abs(float(values.mean()))
    return float(np.count_nonzero(np.abs(null_means) >= observed - 1e-15) / len(null_means))


def _group_row(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, label: dict[str, Any]
) -> dict[str, Any]:
    native_scores = np.asarray([_score(native) for native, _fixed in pairs])
    fixed_scores = np.asarray([_score(fixed) for _native, fixed in pairs])
    differences = fixed_scores - native_scores
    native_failures = sum(native["status"] == "failure" for native, _fixed in pairs)
    fixed_failures = sum(fixed["status"] == "failure" for _native, fixed in pairs)
    measurable = [
        (native["hot_resident_bytes"], fixed["hot_resident_bytes"])
        for native, fixed in pairs
        if native["hot_resident_bytes"] > 0 and fixed["hot_resident_bytes"] > 0
    ]
    kv_fraction = (
        sum(fixed for _native, fixed in measurable) / sum(native for native, _fixed in measurable)
        if measurable
        else None
    )
    return {
        **label,
        "paired_examples": len(pairs),
        "native_accuracy_intention_to_treat": float(native_scores.mean()),
        "qwen_selected_accuracy_intention_to_treat": float(fixed_scores.mean()),
        "mean_difference": float(differences.mean()),
        "mean_difference_percentage_points": float(differences.mean() * 100.0),
        "native_failures": native_failures,
        "qwen_selected_failures": fixed_failures,
        "native_failure_rate": native_failures / len(pairs),
        "qwen_selected_failure_rate": fixed_failures / len(pairs),
        "failure_rate_difference": (fixed_failures - native_failures) / len(pairs),
        "measurable_kv_pairs": len(measurable),
        "realized_kv_fraction": kv_fraction,
        "native_hot_resident_bytes_sum": sum(native for native, _fixed in measurable),
        "qwen_selected_hot_resident_bytes_sum": sum(fixed for _native, fixed in measurable),
    }


def summarize_pairs(
    native: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    _require(len(native) == len(fixed) == EXPECTED_EXAMPLES, "Paired record count drifted.")
    pairs = list(zip(native, fixed, strict=True))
    for native_row, fixed_row in pairs:
        _require(
            native_row["example_id"] == fixed_row["example_id"]
            and native_row["length_tokens"] == fixed_row["length_tokens"]
            and native_row["task"] == fixed_row["task"]
            and native_row["raw_prompt_sha256"] == fixed_row["raw_prompt_sha256"]
            and native_row["input_token_ids_sha256"] == fixed_row["input_token_ids_sha256"]
            and native_row["exact_input_tokens"] == fixed_row["exact_input_tokens"]
            and native_row["generation_reserve_tokens"] == fixed_row["generation_reserve_tokens"],
            f"Paired input drifted at {native_row['example_id']}.",
        )

    overall = _group_row(pairs, label={})
    differences = np.asarray(
        [_score(fixed_row) - _score(native_row) for native_row, fixed_row in pairs]
    )
    overall.update(paired_bootstrap(differences, manifest=manifest))

    task_length_rows = []
    for length in LENGTHS:
        for task in TASKS:
            group = [
                pair
                for pair in pairs
                if pair[0]["length_tokens"] == length and pair[0]["task"] == task
            ]
            _require(len(group) == SAMPLES_PER_TASK, "Task-by-length coverage drifted.")
            task_length_rows.append(
                _group_row(group, label={"length_tokens": length, "task": task})
            )

    length_rows = []
    for length in LENGTHS:
        group = [pair for pair in pairs if pair[0]["length_tokens"] == length]
        row = _group_row(group, label={"length_tokens": length})
        task_means = [
            next(
                item["mean_difference"]
                for item in task_length_rows
                if item["length_tokens"] == length and item["task"] == task
            )
            for task in TASKS
        ]
        row["exact_task_cluster_sign_flip_two_sided_p"] = exact_task_sign_flip(task_means)
        row["task_clusters"] = len(TASKS)
        row["exact_sign_assignments"] = 1 << len(TASKS)
        length_rows.append(row)
    adjusted = holm_bonferroni(
        {
            str(row["length_tokens"]): row["exact_task_cluster_sign_flip_two_sided_p"]
            for row in length_rows
        }
    )
    for row in length_rows:
        row["holm_adjusted_task_cluster_p"] = adjusted[str(row["length_tokens"])]

    task_rows = [
        _group_row(
            [pair for pair in pairs if pair[0]["task"] == task],
            label={"task": task},
        )
        for task in TASKS
    ]
    worst = min(task_length_rows, key=lambda row: row["mean_difference"])
    measurable_cells = [row for row in task_length_rows if row["realized_kv_fraction"] is not None]
    maximum_kv_fraction = (
        max(float(row["realized_kv_fraction"]) for row in measurable_cells)
        if len(measurable_cells) == len(task_length_rows)
        else None
    )
    gate_contract = manifest["transfer_gate"]
    checks = {
        "overall_mean_accuracy_difference": overall["mean_difference"]
        >= gate_contract["overall_mean_accuracy_difference_minimum"],
        "overall_paired_bootstrap_lower_bound": overall["paired_bootstrap_95_ci"][0]
        >= gate_contract["overall_paired_bootstrap_lower_bound_minimum"],
        "worst_task_length_regression": worst["mean_difference"]
        >= gate_contract["worst_task_length_regression_minimum"],
        "failure_rate_increase": overall["failure_rate_difference"]
        <= gate_contract["maximum_failure_rate_increase"],
        "maximum_realized_kv_fraction": maximum_kv_fraction is not None
        and maximum_kv_fraction <= gate_contract["maximum_realized_kv_fraction"],
    }
    return {
        "overall": overall,
        "by_length_with_exact_task_cluster_inference": length_rows,
        "by_task": task_rows,
        "by_task_length": task_length_rows,
        "worst_task_length": worst,
        "operational_failures_by_arm": {
            "native-dense": dict(
                sorted(
                    Counter(
                        row["failure_type"] for row in native if row["status"] == "failure"
                    ).items()
                )
            ),
            "qwen-selected-memory-matched": dict(
                sorted(
                    Counter(
                        row["failure_type"] for row in fixed if row["status"] == "failure"
                    ).items()
                )
            ),
        },
        "memory": {
            "aggregation": gate_contract["realized_kv_fraction_aggregation"],
            "measurable_task_length_cells": len(measurable_cells),
            "required_task_length_cells": len(task_length_rows),
            "maximum_task_length_realized_kv_fraction": maximum_kv_fraction,
        },
        "transfer_gate": {
            "passed": all(checks.values()),
            "all_required": True,
            "checks": checks,
            "thresholds": gate_contract,
            "interpretation": (
                "bounded cross-family transfer supported"
                if all(checks.values())
                else "negative or bounded transfer result retained without Phi reselection"
            ),
        },
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
        "--result-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler/results"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler.summary.json"
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    manifest_digest = sha256(args.manifest)
    runner_path = Path(__file__).with_name("run_p3_cross_family_ruler.py")
    runner_digest = sha256(runner_path)
    loaded = {
        arm: load_cell(
            args.result_root / arm / "cell.json",
            arm=arm,
            manifest_digest=manifest_digest,
            runner_digest=runner_digest,
        )
        for arm in ARMS
    }
    native_cell, native = loaded["native-dense"]
    fixed_cell, fixed = loaded["qwen-selected-memory-matched"]
    _require(
        native_cell["benchmark_dataset_digest_set_sha256"]
        == fixed_cell["benchmark_dataset_digest_set_sha256"],
        "Cross-family arm dataset digest sets differ.",
    )
    _require(
        native_cell["run_identity"]["sequence_gate_dependencies"]
        == fixed_cell["run_identity"]["sequence_gate_dependencies"],
        "Cross-family arm sequence gates differ.",
    )
    statistics = summarize_pairs(native, fixed, manifest=manifest)
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Cross-family RULER summarization requires a clean source tree.")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    raw_digest_set = hashlib.sha256(
        "\n".join(
            sorted([native_cell["raw_records"]["sha256"], fixed_cell["raw_records"]["sha256"]])
        ).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "source": {
            "commit": source_commit,
            "dirty": False,
            "implementation_sha256": sha256(Path(__file__)),
        },
        "experiment_manifest": {"path": str(args.manifest), "sha256": manifest_digest},
        "model": {
            "repo_id": manifest["model"]["repo_id"],
            "revision": MODEL_REVISION,
            "family": "Phi-4",
        },
        "audit": {
            "terminal_arms": len(loaded),
            "expected_examples_per_arm": EXPECTED_EXAMPLES,
            "verified_examples_per_arm": {
                arm: len(records) for arm, (_cell, records) in loaded.items()
            },
            "all_raw_records_verified": True,
            "all_dependency_digests_verified": True,
            "all_example_pairs_verified": True,
            "no_silent_truncation_verified": True,
            "qwen_selection_reused_without_phi_tuning": True,
            "raw_record_digest_set_sha256": raw_digest_set,
        },
        "statistics": statistics,
        "claim_boundary": manifest["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_examples": EXPECTED_EXAMPLES,
                "transfer_gate_passed": statistics["transfer_gate"]["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
