from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from run_p3_mrcr import atomic_json
from run_p3_natural_ruler import compatibility_arm_config
from run_p3_scbench import ADAPTIVE_QUOTA_ARMS, load_adaptive_prerequisite
from summarize_p2_core_matrix import holm_bonferroni
from summarize_p3_natural_adaptive_quota_ruler import verify_quota_audit
from summarize_p3_natural_benchmark import (
    _records,
    audit_arm,
    build_score_verifier,
    expected_example_identifiers,
    expected_record_revisions,
)
from validate_p3_natural_adaptive_quota_scbench_manifest import (
    MODES,
    PREDICTIONS_PER_ARM,
    TASKS,
    validate_manifest,
)
from validate_p3_natural_suite_manifest import validate_manifest as validate_natural_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-natural-adaptive-quota-scbench-audit-v1"
RUNNER = Path("research/adaptive_v4_memory/scripts/run_p3_scbench.py")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dependency(metadata: Any, label: str) -> dict[str, str]:
    _require(isinstance(metadata, dict), f"Missing adaptive SCBench {label} dependency.")
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == sha256(path),
        f"Adaptive SCBench {label} dependency drifted.",
    )
    return {"path": str(path), "sha256": metadata["sha256"]}


def _distribution(values: Iterable[float | int]) -> dict[str, Any] | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(array) == 0:
        return None
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


def paired_cluster_bootstrap(
    clusters: dict[str, list[float]], *, seed: int, resamples: int
) -> dict[str, Any]:
    _require(bool(clusters), "Adaptive SCBench paired cluster set is empty.")
    ordered = [clusters[key] for key in sorted(clusters)]
    sums = np.asarray([sum(values) for values in ordered], dtype=np.float64)
    counts = np.asarray([len(values) for values in ordered], dtype=np.float64)
    _require(bool(np.all(counts > 0)), "Adaptive SCBench contains an empty cluster.")
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(ordered), 1.0 / len(ordered), dtype=np.float64)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        sampled = rng.multinomial(len(ordered), probabilities, size=stop - start)
        means[start:stop] = (sampled @ sums) / (sampled @ counts)
    values = np.asarray([value for cluster in ordered for value in cluster], dtype=np.float64)
    lower, upper = np.quantile(means, (0.025, 0.975))
    lower_tail = (np.count_nonzero(means <= 0.0) + 1) / (resamples + 1)
    upper_tail = (np.count_nonzero(means >= 0.0) + 1) / (resamples + 1)
    return {
        "paired_turns": len(values),
        "paired_shared_context_clusters": len(ordered),
        "cluster_unit": "mode/task/shared-context-row",
        "mean_difference": float(values.mean()),
        "mean_difference_percentage_points": float(values.mean() * 100.0),
        "paired_cluster_bootstrap_95_ci": [float(lower), float(upper)],
        "paired_cluster_bootstrap_95_ci_percentage_points": [
            float(lower * 100.0),
            float(upper * 100.0),
        ],
        "two_sided_cluster_bootstrap_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
    }


def _verify_adaptive_cell(
    cell: dict[str, Any],
    *,
    arm: str,
    adaptive_manifest_digest: str,
    selection: dict[str, Any],
    selection_digest: str,
) -> dict[str, Any]:
    identity = cell.get("run_identity", {})
    _require(
        cell.get("cohort") == "adaptive-quota"
        and identity.get("cohort") == "adaptive-quota"
        and cell.get("adaptive_quota_manifest", {}).get("sha256")
        == adaptive_manifest_digest
        and identity.get("adaptive_quota_manifest_sha256") == adaptive_manifest_digest
        and identity.get("arm_config")
        == compatibility_arm_config(arm, selection, selection_digest),
        f"Adaptive SCBench cell identity drifted for {arm}.",
    )
    _dependency(cell.get("adaptive_quota_manifest"), "adaptive manifest")
    prerequisites = cell.get("adaptive_prerequisites")
    _require(
        isinstance(prerequisites, dict)
        and set(prerequisites) == {"adaptive_ruler", "baseline_scbench"},
        f"Adaptive SCBench prerequisite set drifted for {arm}.",
    )
    assert isinstance(prerequisites, dict)
    expected_prerequisites = {
        "adaptive_ruler": (
            "p3-natural-adaptive-quota-ruler-audit-v1",
            65_000,
        ),
        "baseline_scbench": ("p3-natural-scbench-audit-v1", 20_572),
    }
    verified_prerequisites: dict[str, dict[str, str]] = {}
    for name, (experiment_id, predictions) in expected_prerequisites.items():
        metadata = _dependency(prerequisites[name], name)
        verified = load_adaptive_prerequisite(
            Path(metadata["path"]),
            experiment_id=experiment_id,
            predictions=predictions,
            label=name,
        )
        _require(verified == metadata, f"Adaptive SCBench prerequisite drifted: {name}.")
        verified_prerequisites[name] = verified
    _require(
        identity.get("adaptive_prerequisite_sha256")
        == {name: row["sha256"] for name, row in verified_prerequisites.items()},
        f"Adaptive SCBench prerequisite identity drifted for {arm}.",
    )

    decision = cell.get("p3_sequence_decision", {})
    dependencies = decision.get("dependencies", {})
    _require(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {"primary_core", "primary_causal", "nine_seed_causal", "fixed_selection"},
        f"Adaptive SCBench sequence dependency set drifted for {arm}.",
    )
    verified_sequence = {
        name: _dependency(metadata, f"sequence {name}")
        for name, metadata in dependencies.items()
    }
    _require(
        identity.get("sequence_gate_dependency_sha256")
        == {name: row["sha256"] for name, row in verified_sequence.items()}
        and identity.get("causal_gate_sha256")
        == verified_sequence["primary_causal"]["sha256"]
        and identity.get("fixed_selection_sha256")
        == verified_sequence["fixed_selection"]["sha256"],
        f"Adaptive SCBench sequence identity drifted for {arm}.",
    )
    return {
        "prerequisites": verified_prerequisites,
        "sequence_dependencies": verified_sequence,
        "decision": decision,
    }


def _cluster_id(row: dict[str, Any]) -> str:
    return f"{row['mode']}:{row['task']}:{row['row_index']}"


def verify_initial_prefill_audits(
    records: list[dict[str, Any]], *, arm: str, allowed_failures: set[str]
) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in records:
        failure = row.get("failure_type")
        if row["status"] == "failure":
            _require(
                failure in allowed_failures,
                f"Unregistered adaptive SCBench failure at {row['example_id']}/{arm}.",
            )
        else:
            _require(failure is None, f"Scored SCBench turn carries failure at {row['example_id']}.")
        latency = row.get("initial_prefill_latency_ms")
        peak = row.get("initial_prefill_peak_hbm_bytes")
        resident = row.get("initial_prefill_hot_resident_bytes")
        _require(
            isinstance(latency, (int, float))
            and math.isfinite(float(latency))
            and latency >= 0
            and type(peak) is int
            and peak >= 0
            and type(resident) is int
            and resident >= 0,
            f"Adaptive SCBench initial-prefill measurements drifted at {row['example_id']}.",
        )
        assert isinstance(latency, (int, float))
        assert type(peak) is int
        assert type(resident) is int
        audit = row.get("quota_physical_audit")
        verified_quota: dict[str, Any] | None
        if audit is None:
            _require(
                row["status"] == "failure" and resident == 0,
                f"Successful adaptive SCBench prefill lacks quota audit at {row['example_id']}.",
            )
            verified_quota = None
        else:
            verified_quota = verify_quota_audit(
                audit,
                arm=arm,
                layer_count=36,
                adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
            )
            _require(resident > 0, f"Audited SCBench prefill has zero hot bytes: {row['example_id']}.")
        row["verified_quota"] = verified_quota
        cluster = _cluster_id(row)
        signature = {
            "initial_prefill_latency_ms": float(latency),
            "initial_prefill_peak_hbm_bytes": peak,
            "initial_prefill_hot_resident_bytes": resident,
            "quota_physical_audit": audit,
            "verified_quota": verified_quota,
        }
        previous = clusters.setdefault(cluster, signature)
        _require(
            previous == signature,
            f"Adaptive SCBench shared-prefill evidence differs across turns: {cluster}/{arm}.",
        )
    return clusters


def _failure_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = Counter(
        str(row["failure_type"]) for row in records if row["status"] == "failure"
    )
    return {
        "turns": len(records),
        "scored_turns": sum(row["status"] == "scored" for row in records),
        "failures_by_type": dict(sorted(failures.items())),
        "failure_rate": sum(failures.values()) / len(records),
        "mean_score_failures_zero": float(
            np.mean([float(row["score"]) if row["status"] == "scored" else 0.0 for row in records])
        ),
    }


def analyze_pairs(
    fixed: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    expected_examples: int = PREDICTIONS_PER_ARM,
    modes: tuple[str, ...] = MODES,
    tasks: tuple[str, ...] = TASKS,
    expected_clusters_per_mode: int = 922,
    expected_turns_per_mode: int = 5_143,
) -> dict[str, Any]:
    _require(
        len(fixed) == len(adaptive) == expected_examples,
        "Adaptive SCBench paired turn count drifted.",
    )
    fixed_by_id = {row["example_id"]: row for row in fixed}
    adaptive_by_id = {row["example_id"]: row for row in adaptive}
    _require(
        len(fixed_by_id) == len(adaptive_by_id) == expected_examples
        and set(fixed_by_id) == set(adaptive_by_id),
        "Adaptive SCBench paired identities drifted.",
    )
    overall_clusters: dict[str, list[float]] = defaultdict(list)
    mode_clusters: dict[str, dict[str, list[float]]] = {
        mode: defaultdict(list) for mode in modes
    }
    cell_clusters: dict[tuple[str, str], dict[str, list[float]]] = {
        (mode, task): defaultdict(list) for mode in modes for task in tasks
    }
    for identifier in sorted(fixed_by_id):
        reference = fixed_by_id[identifier]
        treatment = adaptive_by_id[identifier]
        _require(
            all(
                reference.get(key) == treatment.get(key)
                for key in (
                    "mode",
                    "task",
                    "row_index",
                    "turn_index",
                    "raw_prompt_sha256",
                    "input_token_ids_sha256",
                    "exact_input_tokens",
                    "generation_reserve_tokens",
                )
            ),
            f"Adaptive SCBench paired input drifted: {identifier}.",
        )
        mode = str(reference["mode"])
        task = str(reference["task"])
        _require(mode in modes and task in tasks, f"Unknown SCBench cell: {mode}/{task}.")
        reference_score = float(reference["score"]) if reference["status"] == "scored" else 0.0
        treatment_score = float(treatment["score"]) if treatment["status"] == "scored" else 0.0
        difference = treatment_score - reference_score
        cluster = _cluster_id(reference)
        overall_clusters[cluster].append(difference)
        mode_clusters[mode][cluster].append(difference)
        cell_clusters[(mode, task)][cluster].append(difference)
    _require(
        all(mode_clusters[mode] for mode in modes)
        and all(cell_clusters[cell] for cell in cell_clusters),
        "Adaptive SCBench mode/task grid is incomplete.",
    )
    for mode in modes:
        _require(
            len(mode_clusters[mode]) == expected_clusters_per_mode
            and sum(len(values) for values in mode_clusters[mode].values())
            == expected_turns_per_mode,
            f"Adaptive SCBench {mode} cluster/turn coverage drifted.",
        )
    statistics = manifest["statistics"]
    seed = statistics["paired_cluster_bootstrap_seed"]
    resamples = statistics["paired_cluster_bootstrap_resamples"]
    overall = paired_cluster_bootstrap(overall_clusters, seed=seed, resamples=resamples)
    by_mode = []
    for index, mode in enumerate(modes, start=1):
        by_mode.append(
            {
                "mode": mode,
                **paired_cluster_bootstrap(
                    mode_clusters[mode], seed=seed + index, resamples=resamples
                ),
            }
        )
    by_mode_task = []
    raw_p: dict[str, float] = {}
    for index, (mode, task) in enumerate(cell_clusters, start=1):
        inference = paired_cluster_bootstrap(
            cell_clusters[(mode, task)], seed=seed + 100 + index, resamples=resamples
        )
        key = f"{mode}/{task}"
        raw_p[key] = inference["two_sided_cluster_bootstrap_p"]
        by_mode_task.append({"mode": mode, "task": task, **inference})
    adjusted = holm_bonferroni(raw_p)
    _require(
        len(raw_p) == statistics["holm_family_size"],
        "Adaptive SCBench Holm family size drifted.",
    )
    for row in by_mode_task:
        row["holm_adjusted_p"] = adjusted[f"{row['mode']}/{row['task']}"]

    fixed_clusters: dict[str, dict[str, Any]] = {}
    adaptive_clusters: dict[str, dict[str, Any]] = {}
    for row in fixed:
        fixed_clusters.setdefault(_cluster_id(row), row)
    for row in adaptive:
        adaptive_clusters.setdefault(_cluster_id(row), row)
    _require(
        set(fixed_clusters) == set(adaptive_clusters),
        "Adaptive SCBench shared-context clusters drifted between arms.",
    )
    quota_errors: list[float] = []
    paired_quota_clusters = 0
    for cluster in fixed_clusters:
        reference_quota = fixed_clusters[cluster].get("verified_quota")
        treatment_quota = adaptive_clusters[cluster].get("verified_quota")
        for quota in (reference_quota, treatment_quota):
            if quota is not None:
                target = int(quota["target_total_kept_tokens"])
                observed = int(quota["observed_total_kept_tokens"])
                quota_errors.append(abs(observed - target) / target if target else 0.0)
        if reference_quota is not None and treatment_quota is not None:
            _require(
                reference_quota["context_tokens"] == treatment_quota["context_tokens"]
                and reference_quota["target_total_kept_tokens"]
                == treatment_quota["target_total_kept_tokens"]
                and reference_quota["observed_total_kept_tokens"]
                == treatment_quota["observed_total_kept_tokens"],
                f"Adaptive SCBench global quota differs between arms: {cluster}.",
            )
            paired_quota_clusters += 1
    fixed_hot = np.asarray(
        [row["initial_prefill_hot_resident_bytes"] for row in fixed_clusters.values()],
        dtype=np.float64,
    )
    adaptive_hot = np.asarray(
        [row["initial_prefill_hot_resident_bytes"] for row in adaptive_clusters.values()],
        dtype=np.float64,
    )
    fixed_hot_mean = float(fixed_hot.mean())
    adaptive_hot_mean = float(adaptive_hot.mean())
    hot_relative_difference = (
        abs(adaptive_hot_mean - fixed_hot_mean) / fixed_hot_mean
        if fixed_hot_mean > 0.0
        else (0.0 if adaptive_hot_mean == 0.0 else math.inf)
    )
    fixed_failure = _failure_summary(fixed)
    adaptive_failure = _failure_summary(adaptive)
    gate_contract = manifest["confirmation_gate"]
    checks = {
        "overall_score_difference": overall["mean_difference"]
        >= gate_contract["overall_score_difference_minimum"],
        "paired_cluster_bootstrap_lower_bound": overall[
            "paired_cluster_bootstrap_95_ci"
        ][0]
        >= gate_contract["paired_cluster_bootstrap_lower_bound_minimum"],
        "nonnegative_mode_count": sum(row["mean_difference"] >= 0.0 for row in by_mode)
        >= gate_contract["nonnegative_mode_count_minimum"],
        "worst_mode_task_regression": min(row["mean_difference"] for row in by_mode_task)
        >= gate_contract["worst_mode_task_regression_minimum"],
        "failure_rate_increase": adaptive_failure["failure_rate"]
        - fixed_failure["failure_rate"]
        <= gate_contract["maximum_failure_rate_increase"],
        "initial_global_kept_token_relative_error": max(quota_errors, default=0.0)
        <= gate_contract["maximum_initial_global_kept_token_relative_error"],
        "initial_hot_resident_byte_relative_difference": hot_relative_difference
        <= gate_contract["maximum_initial_hot_resident_byte_relative_difference"],
        "at_least_one_jointly_successful_prefill": paired_quota_clusters > 0,
    }
    passed = all(checks.values())
    adaptive_quotas = [
        row["verified_quota"]
        for row in adaptive_clusters.values()
        if row.get("verified_quota") is not None
    ]
    return {
        "arms": {
            ADAPTIVE_QUOTA_ARMS[0]: fixed_failure,
            ADAPTIVE_QUOTA_ARMS[1]: adaptive_failure,
        },
        "overall": overall,
        "by_mode": by_mode,
        "by_mode_task": by_mode_task,
        "multiplicity": {
            "method": "Holm-Bonferroni",
            "family_size": len(raw_p),
            "family": "all frozen SCBench mode/task effects",
        },
        "initial_prefill_physical": {
            "shared_context_clusters": len(fixed_clusters),
            "paired_successful_quota_clusters": paired_quota_clusters,
            "maximum_global_kept_token_relative_error": max(quota_errors, default=0.0),
            "fixed_hot_resident_bytes": _distribution(fixed_hot),
            "adaptive_hot_resident_bytes": _distribution(adaptive_hot),
            "absolute_mean_hot_resident_byte_relative_difference": hot_relative_difference,
            "adaptive_controller_time_ns": _distribution(
                quota["controller_time_ns"] for quota in adaptive_quotas
            ),
            "adaptive_layer_quota": _distribution(
                kept for quota in adaptive_quotas for kept in quota["layer_kept_tokens"]
            ),
        },
        "confirmation_gate": {
            "passed": passed,
            "classification": "success" if passed else "bounded-negative-result",
            "checks": checks,
        },
    }


def summarize(
    *,
    adaptive_manifest_path: Path,
    natural_manifest_path: Path,
    result_root: Path,
) -> dict[str, Any]:
    adaptive_manifest = json.loads(adaptive_manifest_path.read_text())
    validate_manifest(adaptive_manifest)
    natural_manifest = json.loads(natural_manifest_path.read_text())
    validate_natural_manifest(natural_manifest)
    adaptive_digest = sha256(adaptive_manifest_path)
    natural_digest = sha256(natural_manifest_path)
    cell_paths = {arm: result_root / arm / "cell.json" for arm in ADAPTIVE_QUOTA_ARMS}
    _require(all(path.is_file() for path in cell_paths.values()), "Missing adaptive SCBench cell.")
    cells = {arm: json.loads(path.read_text()) for arm, path in cell_paths.items()}
    first = cells[ADAPTIVE_QUOTA_ARMS[0]]
    selection_metadata = _dependency(first.get("fixed_baseline_selection"), "fixed selection")
    selection_path = Path(selection_metadata["path"])
    selection = json.loads(selection_path.read_text())
    inventory_metadata = _dependency(first.get("dataset_inventory"), "dataset inventory")
    expected_identifiers = expected_example_identifiers(
        benchmark="SCBench",
        manifest=natural_manifest,
        inventory_path=Path(inventory_metadata["path"]),
    )
    _require(
        len(expected_identifiers) == PREDICTIONS_PER_ARM,
        "Adaptive SCBench frozen identifier count drifted.",
    )
    revisions = expected_record_revisions("SCBench", natural_manifest)
    score_verifier = build_score_verifier(
        benchmark="SCBench",
        manifest=natural_manifest,
        manifest_path=natural_manifest_path,
        cell=first,
    )
    allowed_failures = set(adaptive_manifest["failure_reporting"]["allowed_failure_types"])
    arm_audits: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    adaptive_dependencies: dict[str, dict[str, Any]] = {}
    for arm in ADAPTIVE_QUOTA_ARMS:
        arm_audits[arm], _base_dependencies = audit_arm(
            benchmark="SCBench",
            arm=arm,
            artifact_path=cell_paths[arm],
            expected_examples=PREDICTIONS_PER_ARM,
            manifest_digest=natural_digest,
            allowed_failures=allowed_failures,
            expected_revisions=revisions,
            expected_seed=adaptive_manifest["benchmark"]["generation_seed"],
            expected_identifiers=expected_identifiers,
            score_verifier=score_verifier,
        )
        adaptive_dependencies[arm] = _verify_adaptive_cell(
            cells[arm],
            arm=arm,
            adaptive_manifest_digest=adaptive_digest,
            selection=selection,
            selection_digest=selection_metadata["sha256"],
        )
        raw_path = Path(cells[arm]["raw_records"]["path"])
        records[arm] = _records(raw_path)
        verify_initial_prefill_audits(
            records[arm], arm=arm, allowed_failures=allowed_failures
        )
    _require(
        cells[ADAPTIVE_QUOTA_ARMS[0]]["p3_sequence_decision"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["p3_sequence_decision"]
        and cells[ADAPTIVE_QUOTA_ARMS[0]]["adaptive_prerequisites"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["adaptive_prerequisites"],
        "Adaptive SCBench arms used different frozen prerequisites.",
    )
    analysis = analyze_pairs(
        records[ADAPTIVE_QUOTA_ARMS[0]],
        records[ADAPTIVE_QUOTA_ARMS[1]],
        adaptive_manifest,
    )
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "classification": analysis["confirmation_gate"]["classification"],
        "coverage": {
            "arms": list(ADAPTIVE_QUOTA_ARMS),
            "modes": list(MODES),
            "tasks": list(TASKS),
            "shared_context_rows_per_mode": 922,
            "turn_predictions_per_mode": 5_143,
            "predictions_per_arm": PREDICTIONS_PER_ARM,
            "paired_predictions_total": PREDICTIONS_PER_ARM * 2,
        },
        "audit": {
            "required_arms_terminal": True,
            "terminal_arms": 2,
            "predictions_per_arm": PREDICTIONS_PER_ARM,
            "total_predictions": PREDICTIONS_PER_ARM * 2,
            "paired_turns": PREDICTIONS_PER_ARM,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "all_runtime_kvpress_bindings_verified": True,
            "exact_input_pairing_verified": True,
            "shared_context_cluster_pairing_verified": True,
            "initial_prefill_quota_audits_verified": True,
            "same_initial_global_token_budget_verified": True,
            "causal_layer_order_verified": True,
            "failure_accounting_complete": True,
            "operational_failure_vocabulary_verified": True,
            "record_revision_provenance_verified": True,
            "model_snapshot_digest_set_verified": True,
            "adaptive_allocation_scope": "initial shared-context prefill only",
            "continuous_refresh_claim_available": False,
            "outcome_dependent_execution": False,
            "mode_task_cells": len(MODES) * len(TASKS),
            "paired_cluster_bootstrap_resamples": adaptive_manifest["statistics"][
                "paired_cluster_bootstrap_resamples"
            ],
            "paired_cluster_bootstrap_seed": adaptive_manifest["statistics"][
                "paired_cluster_bootstrap_seed"
            ],
        },
        "arm_audits": arm_audits,
        "analysis": analysis,
        "confirmation_gate": analysis["confirmation_gate"],
        "adaptive_manifest": {
            "path": str(adaptive_manifest_path),
            "sha256": adaptive_digest,
        },
        "natural_manifest": {"path": str(natural_manifest_path), "sha256": natural_digest},
        "adaptive_dependencies": adaptive_dependencies[ADAPTIVE_QUOTA_ARMS[0]],
        "arm_cells": {
            arm: {"path": str(path), "sha256": sha256(path)}
            for arm, path in cell_paths.items()
        },
        "claim_boundary": adaptive_manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit adaptive-quota SCBench results.")
    parser.add_argument(
        "--adaptive-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-scbench-v1.json"
        ),
    )
    parser.add_argument(
        "--natural-manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "scbench-qwen3-4b"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "scbench-qwen3-4b.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Adaptive SCBench summarization requires a clean source tree.")
    payload = summarize(
        adaptive_manifest_path=args.adaptive_manifest,
        natural_manifest_path=args.natural_manifest,
        result_root=args.result_root,
    )
    payload["source"] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": False,
        "implementation_sha256": sha256(Path(__file__)),
        "runner_sha256": sha256(RUNNER),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
