from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_p2_continuous_rank as audit  # noqa: E402
import p2_continuous_rank_contract as contract  # noqa: E402


def _boundary(*, top: list[int], bottom: list[int], identified: bool) -> dict:
    return {
        "top": {
            "identified": identified,
            "point_boundary_candidates": top,
        },
        "bottom": {
            "identified": identified,
            "point_boundary_candidates": bottom,
        },
    }


def test_scientific_boundary_tie_is_terminal_no_go_not_schema_error() -> None:
    result = audit._extreme_layers(  # noqa: SLF001
        _boundary(top=[2, 4], bottom=[6], identified=False)
    )

    assert result == {
        "eligible": False,
        "top_layer": None,
        "bottom_layer": 6,
        "top_candidates": [2, 4],
        "bottom_candidates": [6],
    }


def test_full_forward_gate_requires_exact_reproduction_and_both_boundaries() -> None:
    payload = {
        "scale": "s55",
        "training_seed": 6071401,
        "reproduction": {
            "all_exact": True,
            "budgets": {budget: {"exact": True} for budget in ("1x", "2x", "4x")},
        },
        "budgets": {
            budget: {
                "slice_count": 45,
                "bootstrap_seed": contract.bootstrap_seed("s55", 6071401, budget),
                "boundary_evidence": _boundary(top=[2], bottom=[6], identified=True),
            }
            for budget in contract.BUDGETS
        },
    }

    result = audit._full_forward_cell_result(payload)  # noqa: SLF001

    assert all(item["eligible"] for item in result.values())
    assert result["2x"]["top_layer"] == 2
    assert result["4x"]["bottom_layer"] == 6


def test_exact_path_repeat_or_transfer_failure_is_reported_as_no_go() -> None:
    payload = {
        "scale": "s55",
        "training_seed": 6071401,
        "batch_runs": contract.EXACT_PATH_BATCH_RUNS_PER_CELL,
        "executed_conversations": contract.EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL,
        "budgets": {
            budget: {
                "bootstrap_seed": contract.bootstrap_seed("s55", 6071401, budget),
                "repeat_integrity": {
                    "workload_streams_identical": True,
                    "semantic_query_streams_identical": True,
                    "fp32_score_streams_identical": budget == "2x",
                    "observation_streams_identical": True,
                },
                "path_boundary_evidence": _boundary(top=[2], bottom=[6], identified=True),
            }
            for budget in contract.BUDGETS
        },
    }
    lookup = {
        ("s55", 6071401, "2x"): (2, 6),
        ("s55", 6071401, "4x"): (4, 6),
    }

    result = audit._exact_path_cell_result(payload, full_lookup=lookup)  # noqa: SLF001

    assert result["2x"]["eligible"] is True
    assert result["4x"]["eligible"] is False
    assert result["4x"]["fp32_score_streams_identical"] is False
    assert result["4x"]["path_transfer_passed"] is False


def test_cell_validation_dispatches_to_the_owning_worker_before_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": contract.FULL_FORWARD_EXPERIMENT_ID,
        "status": "terminal",
        "scale": "s55",
        "training_seed": 6_071_401,
        "calibration_seed": 7_071_401,
    }
    payload["payload_sha256"] = contract.payload_digest(payload)
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(payload))

    def reject(_payload: dict) -> None:
        raise ValueError("deep validator sentinel")

    monkeypatch.setattr(
        audit.importlib,
        "import_module",
        lambda name: SimpleNamespace(validate_payload=reject),
    )
    with pytest.raises(ValueError, match="deep validator sentinel"):
        audit._validate_cell(  # noqa: SLF001
            path,
            experiment_id=contract.FULL_FORWARD_EXPERIMENT_ID,
            scale="s55",
            training_seed=6_071_401,
            manifest_path=tmp_path / "manifest.json",
            manifest={},
        )


def test_full_audit_must_equal_deep_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    payload = {
        "schema_version": 1,
        "experiment_id": contract.FULL_FORWARD_AUDIT_ID,
        "status": "terminal",
        "manifest": {
            "path": str(manifest_path),
            "sha256": contract.sha256(manifest_path),
        },
        "exact_path_phase_permitted": True,
        "cells": [],
    }
    payload["payload_sha256"] = contract.payload_digest(payload)
    path = tmp_path / "full.audit.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(audit, "summarize_full_forward", lambda **kwargs: payload)

    assert audit._validate_full_audit(path, manifest_path) == payload  # noqa: SLF001

    mutated = dict(payload)
    mutated["cells"] = [{"fabricated": True}]
    mutated["payload_sha256"] = contract.payload_digest(mutated)
    path.write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="deep recomputation"):
        audit._validate_full_audit(path, manifest_path)  # noqa: SLF001
