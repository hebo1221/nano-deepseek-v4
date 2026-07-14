from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p3_natural_safety_suite import ARMS, summarize  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: str) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    blob = subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout
    return {
        "commit": commit,
        "dirty": False,
        "implementation_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"
    manifest_digest = _digest(manifest)
    long_expected = 3086
    long_raw_cells: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        raw = tmp_path / f"longsafety-{arm}.json"
        raw.write_text("{}")
        long_raw_cells[arm] = {"path": str(raw), "sha256": _digest(raw)}
    longsafety = tmp_path / "longsafety.json"
    longsafety.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-longsafety-generation-audit-v1",
                "source": _source("research/adaptive_v4_memory/scripts/summarize_p3_longsafety.py"),
                "manifest": {"sha256": manifest_digest},
                "audit": {
                    "generation_arms_terminal": True,
                    "input_pairing_verified": True,
                    "generation_failure_accounting_complete": True,
                    "source_implementations_verified": True,
                    "dependency_digests_verified": True,
                    "record_revisions_verified": True,
                    "terminal_measurement_schema_verified": True,
                    "generation_seed_verified": True,
                    "official_judge_status": "blocked",
                    "expected_generations_per_arm": long_expected,
                    "expected_generations_total": long_expected * 2,
                },
                "arms": {
                    arm: {
                        "terminal_generation": True,
                        "expected_generations": long_expected,
                        "generated": long_expected - 1,
                        "failures_by_type": {"oom": 1},
                        "raw_cell": long_raw_cells[arm],
                    }
                    for arm in ARMS
                },
                "official_judge": {
                    "status": "blocked",
                    "safety_scores_reported": False,
                    "raw_generations_preserved": True,
                },
            }
        )
    )
    ifeval = tmp_path / "ifeval.json"
    generation_cells: dict[str, dict[str, str]] = {}
    official_results: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        generation = tmp_path / f"ifeval-generation-{arm}.json"
        generation.write_text("{}")
        generation_cells[arm] = {"path": str(generation), "sha256": _digest(generation)}
        official = tmp_path / f"ifeval-official-{arm}.jsonl"
        official.write_text("{}\n")
        official_results[arm] = {"path": str(official), "sha256": _digest(official)}
    metrics = {
        "expected_prompts": 541,
        "scored_prompts": 540,
        "scorer_failures": 1,
        "generation_failures": 0,
        "instruction_total": 1000,
        "prompt_level_strict_accuracy": 0.5,
        "instruction_level_strict_accuracy": 0.6,
        "prompt_level_loose_accuracy": 0.7,
        "instruction_level_loose_accuracy": 0.8,
    }
    ifeval.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-ifeval-official-audit-v1",
                "source": _source("research/adaptive_v4_memory/scripts/score_p3_ifeval.py"),
                "manifest": {"sha256": manifest_digest},
                "input_pairing_verified": True,
                "audit": {
                    "required_arms_terminal": True,
                    "input_pairing_verified": True,
                    "official_scoring_accounted": True,
                    "source_implementations_verified": True,
                    "generation_dependency_digests_verified": True,
                    "generation_record_revisions_verified": True,
                    "generation_terminal_measurement_schema_verified": True,
                    "generation_seed_verified": True,
                    "official_result_schema_verified": True,
                    "expected_prompts_per_arm": 541,
                },
                "generation_cells": generation_cells,
                "arms": {
                    arm: {
                        "metrics": metrics,
                        "raw_official_results": official_results[arm],
                    }
                    for arm in ARMS
                },
                "paired_fixed_minus_native": {
                    "paired_prompts": 541,
                    "mean_difference": 0.0,
                    "paired_bootstrap_95_ci": [0.0, 0.0],
                    "wins": 0,
                    "ties": 541,
                    "losses": 0,
                    "bootstrap_seed": 9171403,
                    "bootstrap_replicates": 10000,
                },
            }
        )
    )
    return manifest, longsafety, ifeval


def test_natural_safety_suite_preserves_paid_judge_blocker(tmp_path: Path) -> None:
    manifest, longsafety, ifeval = _fixtures(tmp_path)

    result = summarize(manifest_path=manifest, longsafety_path=longsafety, ifeval_path=ifeval)

    assert result["audit"]["longsafety_generation_terminal"] is True
    assert result["audit"]["longsafety_official_judge_status"] == "blocked"
    assert result["audit"]["longsafety_safety_scores_reported"] is False
    assert result["audit"]["longsafety_raw_evidence_verified"] is True
    assert result["audit"]["ifeval_official_terminal"] is True
    assert result["audit"]["ifeval_raw_evidence_verified"] is True
    assert result["audit"]["source_implementations_verified"] is True
    assert result["audit"]["raw_artifact_digests_verified"] is True
    assert result["audit"]["statistical_schema_verified"] is True
    assert result["audit"]["comparative_long_context_safety_claim_available"] is False
    assert result["classification"].endswith("paid-judge-blocker")


def test_natural_safety_suite_rejects_invented_longsafety_scores(tmp_path: Path) -> None:
    manifest, longsafety, ifeval = _fixtures(tmp_path)
    payload = json.loads(longsafety.read_text())
    payload["official_judge"]["safety_scores_reported"] = True
    longsafety.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="judge boundary drifted"):
        summarize(manifest_path=manifest, longsafety_path=longsafety, ifeval_path=ifeval)


def test_natural_safety_suite_rejects_unaccounted_ifeval_audit(tmp_path: Path) -> None:
    manifest, longsafety, ifeval = _fixtures(tmp_path)
    payload = json.loads(ifeval.read_text())
    payload["audit"]["official_scoring_accounted"] = False
    ifeval.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="official audit contract drifted"):
        summarize(manifest_path=manifest, longsafety_path=longsafety, ifeval_path=ifeval)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("boolean-count", "IFEval official accounting is incomplete"),
        ("infinite-rate", "IFEval official accounting is incomplete"),
        ("raw-digest", "official results artifact drifted"),
        ("bootstrap-seed", "paired statistical schema drifted"),
        ("bootstrap-ci", "paired statistical schema drifted"),
    ],
)
def test_natural_safety_suite_rejects_invalid_statistics_and_raw_digests(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, longsafety, ifeval = _fixtures(tmp_path)
    payload = json.loads(ifeval.read_text())
    if mutation == "boolean-count":
        payload["arms"][ARMS[0]]["metrics"]["scored_prompts"] = True
    elif mutation == "infinite-rate":
        payload["arms"][ARMS[0]]["metrics"]["prompt_level_strict_accuracy"] = float(
            "inf"
        )
    elif mutation == "raw-digest":
        payload["arms"][ARMS[0]]["raw_official_results"]["sha256"] = "0" * 64
    elif mutation == "bootstrap-seed":
        payload["paired_fixed_minus_native"]["bootstrap_seed"] = 0
    else:
        payload["paired_fixed_minus_native"]["paired_bootstrap_95_ci"] = [0.5, -0.5]
    ifeval.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        summarize(manifest_path=manifest, longsafety_path=longsafety, ifeval_path=ifeval)
