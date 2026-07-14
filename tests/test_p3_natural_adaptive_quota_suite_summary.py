from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p3_natural_adaptive_quota_suite import summarize  # noqa: E402
from validate_p3_natural_adaptive_quota_suite_manifest import (  # noqa: E402
    ARMS,
    BENCHMARKS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_manifest() -> Path:
    return (
        ROOT
        / "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-suite-v1.json"
    )


def _summary(
    tmp_path: Path,
    *,
    benchmark: str,
    contract: dict,
    gate_passed: bool,
    difference: float,
) -> dict:
    cells = {}
    for arm in ARMS:
        path = tmp_path / f"{benchmark}-{arm}.json"
        path.write_text(json.dumps({"benchmark": benchmark, "arm": arm}))
        cells[arm] = {"path": str(path), "sha256": _sha256(path)}
    checks = {name: True for name in contract["required_gate_checks"]}
    gate = {
        "passed": gate_passed,
        "classification": "success" if gate_passed else "bounded-negative-result",
        "checks": checks,
    }
    overall = {
        "mean_difference": difference,
        (
            "paired_cluster_bootstrap_95_ci"
            if benchmark == "SCBench"
            else "paired_bootstrap_95_ci"
        ): [difference - 0.01, difference + 0.01],
    }
    if benchmark != "RULER":
        overall[
            "two_sided_cluster_bootstrap_p"
            if benchmark == "SCBench"
            else "two_sided_bootstrap_p"
        ] = 0.25
    audit = {
        "required_arms_terminal": True,
        "terminal_arms": 2,
        "predictions_per_arm": contract["predictions_per_arm"],
        "total_predictions": contract["paired_predictions_total"],
        "all_raw_records_verified": True,
        "all_scores_recomputed_from_raw_response": True,
        "all_dependency_digests_verified": True,
        "all_runtime_kvpress_bindings_verified": True,
        "failure_accounting_complete": True,
        "operational_failure_vocabulary_verified": True,
        "model_snapshot_digest_set_verified": True,
        (
            "same_global_token_budget_verified"
            if benchmark == "RULER"
            else "same_initial_global_token_budget_verified"
        ): True,
    }
    payload = {
        "experiment_id": contract["summary_experiment_id"],
        "status": "terminal",
        "classification": gate["classification"],
        "coverage": {
            "arms": list(ARMS),
            "predictions_per_arm": contract["predictions_per_arm"],
            "paired_predictions_total": contract["paired_predictions_total"],
        },
        "audit": audit,
        "analysis": {"overall": overall},
        "arm_cells": cells,
        "source": {"commit": "a" * 40, "dirty": False},
    }
    if benchmark == "RULER":
        payload["analysis"]["confirmation_gate"] = gate
    else:
        payload["confirmation_gate"] = gate
    return payload


def _write_summaries(
    tmp_path: Path, *, effects: dict[str, float] | None = None
) -> dict[str, Path]:
    manifest = json.loads(_suite_manifest().read_text())
    effects = effects or {benchmark: 0.01 for benchmark in BENCHMARKS}
    paths = {}
    for index, benchmark in enumerate(BENCHMARKS):
        payload = _summary(
            tmp_path,
            benchmark=benchmark,
            contract=manifest["components"][benchmark],
            gate_passed=index < 3,
            difference=effects[benchmark],
        )
        path = tmp_path / f"{benchmark}.summary.json"
        path.write_text(json.dumps(payload))
        paths[benchmark] = path
    return paths


def test_adaptive_natural_suite_passes_only_registered_broad_gate(tmp_path: Path) -> None:
    result = summarize(
        suite_manifest_path=_suite_manifest(),
        root=ROOT,
        summary_overrides=_write_summaries(tmp_path),
    )

    assert result["classification"] == "success"
    assert result["suite_confirmation_gate"]["component_gates_passed"] == 3
    assert result["suite_confirmation_gate"]["nonnegative_overall_effects"] == 4
    assert result["audit"]["total_predictions"] == 89_578
    assert all(row["component_p_value_used_by_suite_gate"] is False for row in result["components"])


def test_adaptive_natural_suite_does_not_hide_one_negative_benchmark(tmp_path: Path) -> None:
    effects = {benchmark: 0.01 for benchmark in BENCHMARKS}
    effects["MRCR"] = -0.001

    result = summarize(
        suite_manifest_path=_suite_manifest(),
        root=ROOT,
        summary_overrides=_write_summaries(tmp_path, effects=effects),
    )

    assert result["classification"] == "bounded-negative-result"
    assert result["suite_confirmation_gate"]["nonnegative_overall_effects"] == 3


def test_adaptive_natural_suite_rejects_failed_physical_gate(tmp_path: Path) -> None:
    paths = _write_summaries(tmp_path)
    payload = json.loads(paths["SCBench"].read_text())
    payload["confirmation_gate"]["checks"][
        "initial_global_kept_token_relative_error"
    ] = False
    paths["SCBench"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="physical or failure gate failed"):
        summarize(
            suite_manifest_path=_suite_manifest(), root=ROOT, summary_overrides=paths
        )


def test_adaptive_natural_suite_rejects_arm_cell_digest_drift(tmp_path: Path) -> None:
    paths = _write_summaries(tmp_path)
    payload = json.loads(paths["MRCR"].read_text())
    payload["arm_cells"][ARMS[0]]["sha256"] = "0" * 64
    paths["MRCR"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="arm-cell digest drifted"):
        summarize(
            suite_manifest_path=_suite_manifest(), root=ROOT, summary_overrides=paths
        )


def test_adaptive_natural_suite_rejects_component_manifest_drift(tmp_path: Path) -> None:
    manifest = json.loads(_suite_manifest().read_text())
    manifest["components"]["RULER"]["manifest_sha256"] = "0" * 64
    altered = tmp_path / "suite.json"
    altered.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="component manifest digest drifted"):
        summarize(
            suite_manifest_path=altered,
            root=ROOT,
            summary_overrides=_write_summaries(tmp_path),
        )
