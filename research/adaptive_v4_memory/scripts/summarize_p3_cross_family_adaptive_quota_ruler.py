from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from run_p3_cross_family_ruler import (
    ADAPTIVE_QUOTA_ARMS,
    EXPECTED_EXAMPLES,
    LENGTHS,
    MODEL_REVISION,
    TASKS,
    adaptive_quota_arm_config,
    load_dataset_contracts,
    require_qwen_adaptive_audit,
)
from run_p3_mrcr import atomic_json
from summarize_p3_cross_family_ruler import _answer_map, _load_arm
from summarize_p3_natural_adaptive_quota_ruler import analyze_pairs, verify_quota_audit
from validate_p3_cross_family_adaptive_quota_manifest import validate_manifest
from validate_p3_cross_family_ruler_manifest import validate_manifest as validate_base_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-cross-family-adaptive-quota-ruler-audit-v1"
RUNNER = Path("research/adaptive_v4_memory/scripts/run_p3_cross_family_ruler.py")
GENERATOR = Path("research/adaptive_v4_memory/scripts/prepare_p3_cross_family_ruler_dataset.py")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_adaptive_cell(
    cell: dict[str, Any],
    *,
    arm: str,
    adaptive_manifest_digest: str,
    qwen_audit_digest: str,
) -> None:
    identity = cell.get("run_identity", {})
    _require(
        cell.get("cohort") == "adaptive-quota"
        and identity.get("cohort") == "adaptive-quota"
        and cell.get("adaptive_quota_manifest", {}).get("sha256") == adaptive_manifest_digest
        and identity.get("adaptive_quota_manifest_sha256") == adaptive_manifest_digest
        and cell.get("qwen_adaptive_audit", {}).get("sha256") == qwen_audit_digest
        and identity.get("qwen_adaptive_audit_sha256") == qwen_audit_digest,
        f"Phi adaptive dependency provenance drifted for {arm}.",
    )
    for label in ("adaptive_quota_manifest", "qwen_adaptive_audit"):
        metadata = cell[label]
        path = Path(metadata.get("path", ""))
        _require(
            path.is_file() and metadata.get("sha256") == sha256(path),
            f"Phi adaptive {label} digest drifted for {arm}.",
        )


def _verify_quota_records(
    records: list[dict[str, Any]], *, arm: str, allowed_failures: set[str]
) -> None:
    for row in records:
        if row["status"] == "failure":
            _require(
                row.get("failure_type") in allowed_failures,
                f"Unregistered Phi failure type at {row['example_id']}/{arm}.",
            )
        else:
            _require(
                row.get("failure_type") is None,
                f"Scored Phi record carries a failure type at {row['example_id']}/{arm}.",
            )
        audit = row.get("quota_physical_audit")
        if audit is None:
            _require(
                row["status"] == "failure" and row["hot_resident_bytes"] == 0,
                f"Successful Phi prefill lacks quota audit at {row['example_id']}/{arm}.",
            )
            row["verified_quota"] = None
            continue
        row["verified_quota"] = verify_quota_audit(
            audit,
            arm=arm,
            layer_count=32,
            adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
        )


def summarize(
    *,
    adaptive_manifest_path: Path,
    base_manifest_path: Path,
    dataset_root: Path,
    selection_path: Path,
    qwen_audit_path: Path,
    result_root: Path,
) -> dict[str, Any]:
    adaptive_manifest = json.loads(adaptive_manifest_path.read_text())
    validate_manifest(adaptive_manifest)
    base_manifest = json.loads(base_manifest_path.read_text())
    validate_base_manifest(base_manifest)
    selection = json.loads(selection_path.read_text())
    require_qwen_adaptive_audit(qwen_audit_path)
    adaptive_digest = sha256(adaptive_manifest_path)
    base_digest = sha256(base_manifest_path)
    selection_digest = sha256(selection_path)
    qwen_digest = sha256(qwen_audit_path)
    dataset_manifests, dataset_digest_set = load_dataset_contracts(
        dataset_root,
        manifest_digest=base_digest,
        generator_digest=sha256(GENERATOR),
    )
    answers = _answer_map(dataset_manifests)
    allowed_failures = set(adaptive_manifest["failure_reporting"]["allowed_failure_types"])
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for arm in ADAPTIVE_QUOTA_ARMS:
        loaded[arm] = _load_arm(
            result_root,
            arm=arm,
            manifest_digest=base_digest,
            runner_digest=sha256(RUNNER),
            dataset_digest_set=dataset_digest_set,
            selection_digest=selection_digest,
            expected_config=adaptive_quota_arm_config(arm, selection, selection_digest),
            answers=answers,
            maximum_context=base_manifest["model"]["maximum_supported_context_tokens"],
            model_digest_set=base_manifest["model"]["snapshot_digest_set_sha256"],
        )
        cell, records = loaded[arm]
        _verify_adaptive_cell(
            cell,
            arm=arm,
            adaptive_manifest_digest=adaptive_digest,
            qwen_audit_digest=qwen_digest,
        )
        _verify_quota_records(records, arm=arm, allowed_failures=allowed_failures)
    fixed_cell, fixed = loaded[ADAPTIVE_QUOTA_ARMS[0]]
    adaptive_cell, adaptive = loaded[ADAPTIVE_QUOTA_ARMS[1]]
    _require(
        fixed_cell["run_identity"]["sequence_gate_dependencies"]
        == adaptive_cell["run_identity"]["sequence_gate_dependencies"],
        "Phi adaptive arm sequence gates differ.",
    )
    analysis = analyze_pairs(
        fixed,
        adaptive,
        adaptive_manifest,
        arms=ADAPTIVE_QUOTA_ARMS,
        layer_count=32,
        lengths=LENGTHS,
        tasks=TASKS,
        expected_examples=EXPECTED_EXAMPLES,
    )
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "classification": analysis["confirmation_gate"]["classification"],
        "model": {
            "repo_id": adaptive_manifest["model"]["repo_id"],
            "revision": MODEL_REVISION,
            "family": "Phi-4",
        },
        "audit": {
            "required_arms_terminal": True,
            "terminal_arms": 2,
            "predictions_per_arm": EXPECTED_EXAMPLES,
            "total_predictions": EXPECTED_EXAMPLES * 2,
            "paired_examples": EXPECTED_EXAMPLES,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "operational_failure_vocabulary_verified": True,
            "exact_input_pairing_verified": True,
            "quota_physical_audits_verified": True,
            "same_global_token_budget_verified": True,
            "causal_layer_order_verified": True,
            "phi_specific_reselection": False,
            "qwen_selection_reused_without_phi_tuning": True,
            "pooled_with_qwen": False,
            "outcome_dependent_execution": False,
            "task_length_cells": len(LENGTHS) * len(TASKS),
            "exact_task_sign_flip_assignments_per_length": 2 ** len(TASKS),
            "paired_bootstrap_resamples": adaptive_manifest["statistics"][
                "paired_bootstrap_resamples"
            ],
            "paired_bootstrap_seed": adaptive_manifest["statistics"]["paired_bootstrap_seed"],
            "natural_compatibility_controller": True,
            "synthetic_controller_unchanged_transfer": False,
        },
        "analysis": analysis,
        "confirmation_gate": analysis["confirmation_gate"],
        "arm_cells": {
            arm: {
                "path": str(result_root / arm / "cell.json"),
                "sha256": sha256(result_root / arm / "cell.json"),
            }
            for arm in ADAPTIVE_QUOTA_ARMS
        },
        "claim_boundary": adaptive_manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen Phi adaptive-quota cohort.")
    parser.add_argument(
        "--adaptive-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-cross-family-adaptive-quota-ruler-v1.json"
        ),
    )
    parser.add_argument(
        "--base-manifest",
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
        "--fixed-selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--qwen-adaptive-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "ruler-qwen3-4b.summary.json"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/"
            "phi4-mini-adaptive-quota-ruler/results"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/"
            "phi4-mini-adaptive-quota-ruler.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Phi adaptive summarization requires a clean source tree.")
    payload = summarize(
        adaptive_manifest_path=args.adaptive_manifest,
        base_manifest_path=args.base_manifest,
        dataset_root=args.dataset_root,
        selection_path=args.fixed_selection,
        qwen_audit_path=args.qwen_adaptive_audit,
        result_root=args.result_root,
    )
    payload["source"] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": False,
        "implementation_sha256": sha256(Path(__file__)),
    }
    payload["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
