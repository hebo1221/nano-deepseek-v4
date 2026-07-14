from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from run_p3_cross_family_ruler import (
    ADAPTIVE_QUOTA_ARMS,
    adaptive_quota_arm_config,
)
from run_p3_longbench_v2 import load_adaptive_prerequisite
from run_p3_mrcr import atomic_json
from summarize_p3_natural_adaptive_quota_longbench_v2 import (
    analyze_pairs,
    verify_quota_records,
)
from summarize_p3_natural_benchmark import (
    _records,
    audit_arm,
    build_score_verifier,
    expected_example_identifiers,
    expected_record_revisions,
)
from validate_p3_cross_family_adaptive_quota_longbench_v2_manifest import (
    CATEGORIES,
    PREDICTIONS_PER_ARM,
    validate_manifest,
)
from validate_p3_natural_suite_manifest import validate_manifest as validate_natural_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-cross-family-adaptive-quota-longbench-v2-audit-v1"
RUNNER = Path("research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dependency(metadata: Any, label: str) -> dict[str, str]:
    _require(isinstance(metadata, dict), f"Missing Phi LongBench-v2 {label} dependency.")
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == sha256(path),
        f"Phi LongBench-v2 {label} dependency drifted.",
    )
    return {"path": str(path), "sha256": metadata["sha256"]}


def _verify_transfer_cell(
    cell: dict[str, Any],
    *,
    arm: str,
    adaptive_manifest_digest: str,
    model_digest: str,
    selection: dict[str, Any],
    selection_digest: str,
) -> dict[str, Any]:
    identity = cell.get("run_identity", {})
    _require(
        cell.get("cohort") == "cross-family-adaptive-quota"
        and identity.get("cohort") == "cross-family-adaptive-quota"
        and cell.get("adaptive_quota_manifest", {}).get("sha256")
        == adaptive_manifest_digest
        and identity.get("adaptive_quota_manifest_sha256")
        == adaptive_manifest_digest
        and identity.get("model_snapshot_digest_set_sha256") == model_digest
        and cell.get("model_snapshot_digest_set_sha256") == model_digest
        and identity.get("arm_config")
        == adaptive_quota_arm_config(arm, selection, selection_digest),
        f"Phi LongBench-v2 cell identity drifted for {arm}.",
    )
    _dependency(cell.get("adaptive_quota_manifest"), "adaptive manifest")
    prerequisites = cell.get("adaptive_prerequisites")
    _require(
        isinstance(prerequisites, dict)
        and set(prerequisites)
        == {"phi_adaptive_ruler", "qwen_adaptive_longbench_v2"},
        f"Phi LongBench-v2 prerequisite set drifted for {arm}.",
    )
    assert isinstance(prerequisites, dict)
    expected = {
        "phi_adaptive_ruler": (
            "p3-cross-family-adaptive-quota-ruler-audit-v1",
            7_800,
        ),
        "qwen_adaptive_longbench_v2": (
            "p3-natural-adaptive-quota-longbench-v2-audit-v1",
            1_006,
        ),
    }
    verified_prerequisites: dict[str, dict[str, str]] = {}
    for name, (experiment_id, predictions) in expected.items():
        metadata = _dependency(prerequisites[name], name)
        verified = load_adaptive_prerequisite(
            Path(metadata["path"]),
            experiment_id=experiment_id,
            predictions=predictions,
            label=name,
        )
        _require(verified == metadata, f"Phi LongBench-v2 prerequisite drifted: {name}.")
        verified_prerequisites[name] = verified
    _require(
        identity.get("adaptive_prerequisite_sha256")
        == {name: row["sha256"] for name, row in verified_prerequisites.items()},
        f"Phi LongBench-v2 prerequisite identity drifted for {arm}.",
    )
    decision = cell.get("p3_sequence_decision", {})
    dependencies = decision.get("dependencies", {})
    _require(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {"primary_core", "primary_causal", "nine_seed_causal", "fixed_selection"},
        f"Phi LongBench-v2 sequence dependency set drifted for {arm}.",
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
        f"Phi LongBench-v2 sequence identity drifted for {arm}.",
    )
    return {
        "prerequisites": verified_prerequisites,
        "sequence_dependencies": verified_sequence,
        "decision": decision,
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
    model_digest = adaptive_manifest["model"]["snapshot_digest_set_sha256"]
    cell_paths = {arm: result_root / arm / "cell.json" for arm in ADAPTIVE_QUOTA_ARMS}
    _require(all(path.is_file() for path in cell_paths.values()), "Missing Phi cells.")
    cells = {arm: json.loads(path.read_text()) for arm, path in cell_paths.items()}
    first = cells[ADAPTIVE_QUOTA_ARMS[0]]
    selection_metadata = _dependency(first.get("fixed_baseline_selection"), "selection")
    selection = json.loads(Path(selection_metadata["path"]).read_text())
    inventory_metadata = _dependency(first.get("dataset_inventory"), "dataset inventory")
    expected_identifiers = expected_example_identifiers(
        benchmark="LongBench-v2",
        manifest=natural_manifest,
        inventory_path=Path(inventory_metadata["path"]),
    )
    _require(
        len(expected_identifiers) == PREDICTIONS_PER_ARM,
        "Phi LongBench-v2 identifier count drifted.",
    )
    revisions = expected_record_revisions("LongBench-v2", natural_manifest)
    revisions["model_revision"] = adaptive_manifest["model"]["revision"]
    score_verifier = build_score_verifier(
        benchmark="LongBench-v2",
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
            benchmark="LongBench-v2",
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
        adaptive_dependencies[arm] = _verify_transfer_cell(
            cells[arm],
            arm=arm,
            adaptive_manifest_digest=adaptive_digest,
            model_digest=model_digest,
            selection=selection,
            selection_digest=selection_metadata["sha256"],
        )
        records[arm] = _records(Path(cells[arm]["raw_records"]["path"]))
        verify_quota_records(
            records[arm],
            arm=arm,
            allowed_failures=allowed_failures,
            layer_count=32,
            adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
        )
    _require(
        cells[ADAPTIVE_QUOTA_ARMS[0]]["p3_sequence_decision"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["p3_sequence_decision"]
        and cells[ADAPTIVE_QUOTA_ARMS[0]]["adaptive_prerequisites"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["adaptive_prerequisites"],
        "Phi LongBench-v2 arms used different frozen prerequisites.",
    )
    analysis = analyze_pairs(
        records[ADAPTIVE_QUOTA_ARMS[0]],
        records[ADAPTIVE_QUOTA_ARMS[1]],
        adaptive_manifest,
        arms=ADAPTIVE_QUOTA_ARMS,
    )
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "classification": analysis["confirmation_gate"]["classification"],
        "coverage": {
            "arms": list(ADAPTIVE_QUOTA_ARMS),
            "categories": list(CATEGORIES),
            "predictions_per_arm": PREDICTIONS_PER_ARM,
            "paired_predictions_total": PREDICTIONS_PER_ARM * 2,
            "transfer_model_family": "Phi-4",
            "maximum_supported_context_tokens": 131_072,
        },
        "audit": {
            "required_arms_terminal": True,
            "terminal_arms": 2,
            "predictions_per_arm": PREDICTIONS_PER_ARM,
            "total_predictions": PREDICTIONS_PER_ARM * 2,
            "paired_examples": PREDICTIONS_PER_ARM,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "all_runtime_kvpress_bindings_verified": True,
            "exact_input_pairing_verified": True,
            "exact_token_id_pairing_verified": True,
            "quota_physical_audits_verified": True,
            "same_initial_global_token_budget_verified": True,
            "causal_layer_order_verified": True,
            "failure_accounting_complete": True,
            "operational_failure_vocabulary_verified": True,
            "record_revision_provenance_verified": True,
            "model_snapshot_digest_set_verified": True,
            "phi_specific_reselection": False,
            "qwen_selection_reused_without_phi_tuning": True,
            "category_cells": len(CATEGORIES),
            "holm_family_size": len(CATEGORIES),
            "paired_bootstrap_resamples": adaptive_manifest["statistics"][
                "paired_bootstrap_resamples"
            ],
            "paired_bootstrap_seed": adaptive_manifest["statistics"][
                "paired_bootstrap_seed"
            ],
            "adaptive_allocation_scope": "initial context prefill only",
            "continuous_refresh_claim_available": False,
            "secondary_slices_are_descriptive": True,
            "outcome_dependent_execution": False,
        },
        "arm_audits": arm_audits,
        "analysis": analysis,
        "confirmation_gate": analysis["confirmation_gate"],
        "adaptive_manifest": {"path": str(adaptive_manifest_path), "sha256": adaptive_digest},
        "natural_manifest": {"path": str(natural_manifest_path), "sha256": natural_digest},
        "adaptive_dependencies": adaptive_dependencies[ADAPTIVE_QUOTA_ARMS[0]],
        "arm_cells": {
            arm: {"path": str(path), "sha256": sha256(path)}
            for arm, path in cell_paths.items()
        },
        "claim_boundary": adaptive_manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phi cross-family adaptive-quota LongBench-v2 results."
    )
    parser.add_argument(
        "--adaptive-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-cross-family-adaptive-quota-longbench-v2-v1.json"
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
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/"
            "phi4-mini-adaptive-quota-longbench-v2"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/"
            "phi4-mini-adaptive-quota-longbench-v2.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Phi LongBench-v2 summarization requires a clean source tree.")
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
