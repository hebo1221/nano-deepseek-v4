from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p3_natural_safety_suite import ARMS, summarize  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"
    manifest_digest = _digest(manifest)
    long_expected = 3086
    longsafety = tmp_path / "longsafety.json"
    longsafety.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-longsafety-generation-audit-v1",
                "source": {"dirty": False},
                "manifest": {"sha256": manifest_digest},
                "audit": {
                    "generation_arms_terminal": True,
                    "input_pairing_verified": True,
                    "generation_failure_accounting_complete": True,
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
    metrics = {
        "expected_prompts": 541,
        "scored_prompts": 540,
        "scorer_failures": 1,
        "prompt_level_strict_accuracy": 0.5,
        "instruction_level_strict_accuracy": 0.6,
        "prompt_level_loose_accuracy": 0.7,
        "instruction_level_loose_accuracy": 0.8,
    }
    ifeval.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-ifeval-official-audit-v1",
                "source": {"dirty": False},
                "manifest": {"sha256": manifest_digest},
                "input_pairing_verified": True,
                "audit": {
                    "required_arms_terminal": True,
                    "input_pairing_verified": True,
                    "official_scoring_accounted": True,
                    "expected_prompts_per_arm": 541,
                },
                "arms": {arm: {"metrics": metrics} for arm in ARMS},
                "paired_fixed_minus_native": {
                    "paired_prompts": 541,
                    "bootstrap_replicates": 10000,
                },
            }
        )
    )
    return manifest, longsafety, ifeval


def test_natural_safety_suite_preserves_paid_judge_blocker(tmp_path: Path) -> None:
    manifest, longsafety, ifeval = _fixtures(tmp_path)

    result = summarize(
        manifest_path=manifest, longsafety_path=longsafety, ifeval_path=ifeval
    )

    assert result["audit"]["longsafety_generation_terminal"] is True
    assert result["audit"]["longsafety_official_judge_status"] == "blocked"
    assert result["audit"]["longsafety_safety_scores_reported"] is False
    assert result["audit"]["ifeval_official_terminal"] is True
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
