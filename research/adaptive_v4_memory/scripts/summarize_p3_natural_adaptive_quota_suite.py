from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from run_p3_mrcr import atomic_json
from validate_p3_natural_adaptive_quota_suite_manifest import (
    ARMS,
    BENCHMARKS,
    MODEL_REVISION,
    validate_manifest,
)

SUMMARY_EXPERIMENT_ID = "p3-natural-adaptive-quota-suite-audit-v1"
COMMON_AUDIT_CHECKS = (
    "required_arms_terminal",
    "all_raw_records_verified",
    "all_scores_recomputed_from_raw_response",
    "all_dependency_digests_verified",
    "all_runtime_kvpress_bindings_verified",
    "failure_accounting_complete",
    "operational_failure_vocabulary_verified",
    "model_snapshot_digest_set_verified",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _component_gate(summary: dict[str, Any], benchmark: str) -> dict[str, Any]:
    gate = (
        summary.get("analysis", {}).get("confirmation_gate")
        if benchmark == "RULER"
        else summary.get("confirmation_gate")
    )
    if not isinstance(gate, dict):
        raise ValueError(f"Missing {benchmark} confirmation gate.")
    _require(
        type(gate.get("passed")) is bool
        and gate.get("classification") in {"success", "bounded-negative-result"}
        and isinstance(gate.get("checks"), dict),
        f"Malformed {benchmark} confirmation gate.",
    )
    return gate


def _component_ci(overall: dict[str, Any], benchmark: str) -> list[float]:
    field = (
        "paired_cluster_bootstrap_95_ci"
        if benchmark == "SCBench"
        else "paired_bootstrap_95_ci"
    )
    interval = overall.get(field)
    if not (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
        and float(interval[0]) <= float(interval[1])
    ):
        raise ValueError(f"Malformed {benchmark} confidence interval.")
    return [float(interval[0]), float(interval[1])]


def _verify_arm_cells(
    summary: dict[str, Any], *, benchmark: str, root: Path
) -> dict[str, dict[str, str]]:
    cells = summary.get("arm_cells", {})
    _require(tuple(cells) == ARMS, f"{benchmark} arm-cell set drifted.")
    verified: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        metadata = cells[arm]
        _require(isinstance(metadata, dict), f"Malformed {benchmark}/{arm} cell metadata.")
        path = _resolve(str(metadata.get("path", "")), root)
        _require(
            path.is_file() and metadata.get("sha256") == sha256(path),
            f"{benchmark}/{arm} arm-cell digest drifted.",
        )
        verified[arm] = _metadata(path)
    return verified


def _audit_component(
    *,
    benchmark: str,
    contract: dict[str, Any],
    summary_path: Path,
    root: Path,
) -> dict[str, Any]:
    manifest_path = _resolve(contract["manifest"], root)
    _require(
        manifest_path.is_file() and sha256(manifest_path) == contract["manifest_sha256"],
        f"{benchmark} component manifest digest drifted.",
    )
    component_manifest = json.loads(manifest_path.read_text())
    _require(
        component_manifest.get("model", {}).get("revision") == MODEL_REVISION
        and tuple(component_manifest.get("arms", {})) == ARMS
        and component_manifest.get("benchmark", {}).get("predictions_per_arm")
        == contract["predictions_per_arm"],
        f"{benchmark} component manifest contract drifted.",
    )
    _require(summary_path.is_file(), f"Missing {benchmark} adaptive summary.")
    summary = json.loads(summary_path.read_text())
    _require(
        summary.get("experiment_id") == contract["summary_experiment_id"]
        and summary.get("status") == "terminal",
        f"{benchmark} adaptive summary is not terminal.",
    )
    gate = _component_gate(summary, benchmark)
    _require(
        summary.get("classification") == gate["classification"],
        f"{benchmark} classification drifted from its confirmation gate.",
    )
    audit = summary.get("audit", {})
    _require(
        all(audit.get(check) is True for check in COMMON_AUDIT_CHECKS)
        and audit.get("terminal_arms") == len(ARMS)
        and audit.get("predictions_per_arm") == contract["predictions_per_arm"]
        and audit.get("total_predictions") == contract["paired_predictions_total"],
        f"{benchmark} component audit is incomplete.",
    )
    budget_key = (
        "same_global_token_budget_verified"
        if benchmark == "RULER"
        else "same_initial_global_token_budget_verified"
    )
    _require(audit.get(budget_key) is True, f"{benchmark} global budget audit failed.")
    checks = gate["checks"]
    required_checks = tuple(contract["required_gate_checks"])
    _require(
        all(checks.get(name) is True for name in required_checks),
        f"{benchmark} physical or failure gate failed.",
    )
    coverage = summary.get("coverage", {})
    _require(
        tuple(coverage.get("arms", ())) == ARMS
        and coverage.get("predictions_per_arm") == contract["predictions_per_arm"]
        and coverage.get("paired_predictions_total") == contract["paired_predictions_total"],
        f"{benchmark} summary coverage drifted.",
    )
    overall = summary.get("analysis", {}).get("overall", {})
    difference = overall.get("mean_difference")
    _require(
        isinstance(difference, (int, float)),
        f"Missing {benchmark} signed overall effect.",
    )
    source = summary.get("source", {})
    _require(
        source.get("dirty") is False
        and isinstance(source.get("commit"), str)
        and bool(source["commit"]),
        f"{benchmark} source provenance is not clean.",
    )
    p_value = overall.get(
        "two_sided_cluster_bootstrap_p"
        if benchmark == "SCBench"
        else "two_sided_bootstrap_p"
    )
    _require(
        p_value is None
        or (isinstance(p_value, (int, float)) and 0.0 <= float(p_value) <= 1.0),
        f"Malformed {benchmark} component p-value.",
    )
    return {
        "benchmark": benchmark,
        "classification": gate["classification"],
        "confirmation_gate_passed": gate["passed"],
        "mean_difference": float(difference),
        "confidence_interval_95": _component_ci(overall, benchmark),
        "component_two_sided_p": float(p_value) if p_value is not None else None,
        "component_p_value_used_by_suite_gate": False,
        "predictions_per_arm": contract["predictions_per_arm"],
        "paired_predictions_total": contract["paired_predictions_total"],
        "required_physical_and_failure_checks": {
            name: checks[name] for name in required_checks
        },
        "manifest": _metadata(manifest_path),
        "summary": _metadata(summary_path),
        "arm_cells": _verify_arm_cells(summary, benchmark=benchmark, root=root),
        "source": source,
    }


def summarize(
    *,
    suite_manifest_path: Path,
    root: Path = Path("."),
    summary_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    suite_manifest = json.loads(suite_manifest_path.read_text())
    validate_manifest(suite_manifest)
    overrides = summary_overrides or {}
    components = []
    for benchmark in BENCHMARKS:
        contract = suite_manifest["components"][benchmark]
        summary_path = overrides.get(benchmark, _resolve(contract["summary"], root))
        components.append(
            _audit_component(
                benchmark=benchmark,
                contract=contract,
                summary_path=summary_path,
                root=root,
            )
        )
    gate_contract = suite_manifest["confirmation_gate"]
    passed_components = sum(row["confirmation_gate_passed"] for row in components)
    nonnegative_effects = sum(row["mean_difference"] >= 0.0 for row in components)
    physical_and_failure_checks_passed = all(
        all(row["required_physical_and_failure_checks"].values()) for row in components
    )
    checks = {
        "terminal_components": len(components)
        == gate_contract["required_terminal_components"],
        "minimum_component_confirmation_gates_passed": passed_components
        >= gate_contract["minimum_component_confirmation_gates_passed"],
        "minimum_nonnegative_overall_effects": nonnegative_effects
        >= gate_contract["minimum_nonnegative_overall_effects"],
        "all_required_physical_and_failure_checks_passed": physical_and_failure_checks_passed,
        "all_components_same_model_revision": True,
        "all_components_same_arm_pair": True,
        "all_components_same_initial_global_token_budget": True,
        "no_cross_benchmark_score_pooling": True,
        "no_cross_benchmark_p_value_pooling": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "classification": "success" if passed else "bounded-negative-result",
        "interpretation": (
            "broad-initial-quota-transfer-supported"
            if passed
            else "broad-initial-quota-transfer-not-supported"
        ),
        "coverage": suite_manifest["coverage"],
        "components": components,
        "suite_confirmation_gate": {
            "passed": passed,
            "component_gates_passed": passed_components,
            "nonnegative_overall_effects": nonnegative_effects,
            "checks": checks,
        },
        "audit": {
            "terminal_components": len(components),
            "predictions_per_arm": sum(row["predictions_per_arm"] for row in components),
            "total_predictions": sum(row["paired_predictions_total"] for row in components),
            "all_component_manifest_digests_verified": True,
            "all_component_summary_digests_recorded": True,
            "all_component_arm_cell_digests_verified": True,
            "all_component_raw_audits_verified": True,
            "all_component_score_recomputation_verified": True,
            "all_component_dependency_digests_verified": True,
            "all_component_runtime_bindings_verified": True,
            "all_component_failure_accounting_verified": True,
            "same_model_revision_verified": True,
            "same_arm_pair_verified": True,
            "same_initial_global_token_budget_verified": True,
            "no_cross_benchmark_score_pooling": True,
            "no_cross_benchmark_p_value_pooling": True,
            "longmemeval_official_judge_boundary_preserved": True,
            "outcome_dependent_benchmark_selection": False,
        },
        "suite_manifest": _metadata(suite_manifest_path),
        "claim_boundary": suite_manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the adaptive natural benchmark suite.")
    parser.add_argument(
        "--suite-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-suite-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/"
            "natural-adaptive-quota/suite.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Adaptive natural suite summarization requires a clean source tree.")
    payload = summarize(suite_manifest_path=args.suite_manifest)
    payload["source"] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": False,
        "implementation_sha256": sha256(Path(__file__)),
    }
    payload["environment"] = {"python": platform.python_version()}
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {"output": str(args.output), "classification": payload["classification"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
