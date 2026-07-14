from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from run_p3_longmemeval import ADAPTIVE_QUOTA_ARMS, load_adaptive_prerequisite
from run_p3_mrcr import atomic_json
from run_p3_natural_ruler import compatibility_arm_config
from summarize_p3_natural_adaptive_quota_ruler import verify_quota_audit
from summarize_p3_natural_benchmark import (
    _records,
    audit_arm,
    build_score_verifier,
    expected_example_identifiers,
    expected_record_revisions,
)
from validate_p3_natural_adaptive_quota_longmemeval_manifest import validate_manifest
from validate_p3_natural_suite_manifest import validate_manifest as validate_natural_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-natural-adaptive-quota-longmemeval-audit-v1"
RUNNER = Path("research/adaptive_v4_memory/scripts/run_p3_longmemeval.py")
EXPECTED_EXAMPLES = 500
JUDGE_BLOCKED = "judge-blocked"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dependency(metadata: Any, label: str) -> dict[str, str]:
    _require(isinstance(metadata, dict), f"Missing adaptive LongMemEval {label} dependency.")
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == sha256(path),
        f"Adaptive LongMemEval {label} dependency drifted.",
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


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        and identity.get("judge_mode") == "blocked"
        and cell.get("adaptive_quota_manifest", {}).get("sha256")
        == adaptive_manifest_digest
        and identity.get("adaptive_quota_manifest_sha256") == adaptive_manifest_digest
        and identity.get("arm_config")
        == compatibility_arm_config(arm, selection, selection_digest),
        f"Adaptive LongMemEval cell identity drifted for {arm}.",
    )
    _dependency(cell.get("adaptive_quota_manifest"), "adaptive manifest")
    prerequisites = cell.get("adaptive_prerequisites")
    _require(
        isinstance(prerequisites, dict)
        and set(prerequisites) == {"adaptive_ruler", "baseline_longmemeval"},
        f"Adaptive LongMemEval prerequisite set drifted for {arm}.",
    )
    assert isinstance(prerequisites, dict)
    contracts = {
        "adaptive_ruler": ("p3-natural-adaptive-quota-ruler-audit-v1", 65_000),
        "baseline_longmemeval": ("p3-natural-longmemeval-audit-v1", 1_000),
    }
    verified_prerequisites: dict[str, dict[str, str]] = {}
    for name, (experiment_id, predictions) in contracts.items():
        metadata = _dependency(prerequisites[name], name)
        verified = load_adaptive_prerequisite(
            Path(metadata["path"]),
            experiment_id=experiment_id,
            predictions=predictions,
            label=name,
        )
        _require(verified == metadata, f"Adaptive LongMemEval prerequisite drifted: {name}.")
        verified_prerequisites[name] = verified
    _require(
        identity.get("adaptive_prerequisite_sha256")
        == {name: row["sha256"] for name, row in verified_prerequisites.items()},
        f"Adaptive LongMemEval prerequisite identity drifted for {arm}.",
    )
    decision = cell.get("p3_sequence_decision", {})
    dependencies = decision.get("dependencies", {})
    _require(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {
            "primary_core",
            "nine_seed_core",
            "primary_causal",
            "nine_seed_causal",
            "fixed_selection",
        },
        f"Adaptive LongMemEval sequence dependency set drifted for {arm}.",
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
        f"Adaptive LongMemEval sequence identity drifted for {arm}.",
    )
    return {
        "prerequisites": verified_prerequisites,
        "sequence_dependencies": verified_sequence,
        "decision": decision,
    }


def verify_generation_records(
    records: list[dict[str, Any]], *, arm: str, allowed_failures: set[str]
) -> dict[str, Any]:
    response_generations = 0
    operational_failures: Counter[str] = Counter()
    quota_audits = 0
    for row in records:
        identifier = row.get("example_id")
        _require(
            row.get("status") == "failure"
            and row.get("failure_type") in allowed_failures
            and row.get("score") is None,
            f"Adaptive LongMemEval frozen judge-blocked contract drifted: {identifier}/{arm}.",
        )
        token_digest = row.get("input_token_ids_sha256")
        _require(
            isinstance(token_digest, str) and len(token_digest) == 64,
            f"Missing exact LongMemEval token digest: {identifier}/{arm}.",
        )
        _require(
            isinstance(row.get("question_type"), str) and bool(row["question_type"]),
            f"Missing LongMemEval question type: {identifier}/{arm}.",
        )
        failure = str(row["failure_type"])
        if failure == JUDGE_BLOCKED:
            judge = row.get("judge")
            _require(
                isinstance(judge, dict)
                and judge.get("model") == "gpt-4o-2024-08-06"
                and judge.get("status") == "blocked"
                and isinstance(judge.get("reason"), str)
                and bool(judge["reason"])
                and isinstance(row.get("raw_response"), str)
                and bool(row["raw_response"].strip())
                and row.get("parsed_response") == row["raw_response"]
                and type(row.get("generated_tokens_observed")) is int
                and row["generated_tokens_observed"] > 0
                and row.get("stop_reason") in {"max-new-tokens", "eos-or-special-token"},
                f"Incomplete judge-blocked generation evidence: {identifier}/{arm}.",
            )
            response_generations += 1
        else:
            operational_failures[failure] += 1
        audit = row.get("quota_physical_audit")
        if audit is None:
            _require(
                failure != JUDGE_BLOCKED and row.get("hot_resident_bytes") == 0,
                f"Successful LongMemEval prefill lacks quota audit: {identifier}/{arm}.",
            )
            row["verified_quota"] = None
        else:
            row["verified_quota"] = verify_quota_audit(
                audit,
                arm=arm,
                layer_count=36,
                adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
            )
            _require(
                row.get("hot_resident_bytes", 0) > 0,
                f"Audited LongMemEval prefill has zero hot bytes: {identifier}/{arm}.",
            )
            quota_audits += 1
    _require(
        response_generations + sum(operational_failures.values()) == len(records),
        f"Adaptive LongMemEval failure accounting does not close for {arm}.",
    )
    return {
        "accounted_examples": len(records),
        "response_generations_completed": response_generations,
        "response_generation_rate": response_generations / len(records),
        "judge_blocked_responses_retained": response_generations,
        "officially_scored_examples": 0,
        "operational_failures_by_type": dict(sorted(operational_failures.items())),
        "operational_failure_rate": sum(operational_failures.values()) / len(records),
        "verified_quota_audits": quota_audits,
        "quality_status": "unverified",
    }


def _arm_measurements(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row["failure_type"] == JUDGE_BLOCKED]
    return {
        "all_terminal_attempts": {
            key: _distribution(row[key] for row in records)
            for key in ("exact_input_tokens", "latency_ms", "peak_hbm_bytes", "hot_resident_bytes")
        },
        "completed_response_generations": {
            key: _distribution(row[key] for row in completed)
            for key in (
                "generated_tokens_observed",
                "latency_ms",
                "peak_hbm_bytes",
                "hot_resident_bytes",
            )
        },
    }


def analyze_generation_pairs(
    fixed: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
    *,
    expected_examples: int = EXPECTED_EXAMPLES,
) -> dict[str, Any]:
    _require(
        len(fixed) == len(adaptive) == expected_examples,
        "Adaptive LongMemEval paired example count drifted.",
    )
    fixed_by_id = {row["example_id"]: row for row in fixed}
    adaptive_by_id = {row["example_id"]: row for row in adaptive}
    _require(
        len(fixed_by_id) == len(adaptive_by_id) == expected_examples
        and set(fixed_by_id) == set(adaptive_by_id),
        "Adaptive LongMemEval paired identities drifted.",
    )
    pairing: Counter[str] = Counter()
    paired_quota_examples = 0
    quota_errors: list[float] = []
    fixed_hot: list[float] = []
    adaptive_hot: list[float] = []
    controller_times: list[int] = []
    layer_quotas: list[int] = []
    digest_rows: list[dict[str, Any]] = []
    question_types: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {arm: Counter() for arm in ADAPTIVE_QUOTA_ARMS}
    )
    for identifier in sorted(fixed_by_id):
        reference = fixed_by_id[identifier]
        treatment = adaptive_by_id[identifier]
        paired_fields = (
            "raw_prompt_sha256",
            "input_token_ids_sha256",
            "exact_input_tokens",
            "generation_reserve_tokens",
            "question_type",
            "abstention",
        )
        _require(
            all(reference.get(key) == treatment.get(key) for key in paired_fields),
            f"Adaptive LongMemEval paired input drifted: {identifier}.",
        )
        fixed_complete = reference["failure_type"] == JUDGE_BLOCKED
        adaptive_complete = treatment["failure_type"] == JUDGE_BLOCKED
        pairing[
            "both_completed"
            if fixed_complete and adaptive_complete
            else "adaptive_only_completed"
            if adaptive_complete
            else "fixed_only_completed"
            if fixed_complete
            else "both_operationally_failed"
        ] += 1
        question_type = str(reference["question_type"])
        for arm, row in zip(ADAPTIVE_QUOTA_ARMS, (reference, treatment), strict=True):
            question_types[question_type][arm][
                "response_completed" if row["failure_type"] == JUDGE_BLOCKED else "operational_failure"
            ] += 1
        fixed_quota = reference.get("verified_quota")
        adaptive_quota = treatment.get("verified_quota")
        for quota in (fixed_quota, adaptive_quota):
            if quota is not None:
                target = int(quota["target_total_kept_tokens"])
                observed = int(quota["observed_total_kept_tokens"])
                quota_errors.append(abs(observed - target) / target if target else 0.0)
        if fixed_quota is not None and adaptive_quota is not None:
            _require(
                fixed_quota["context_tokens"] == adaptive_quota["context_tokens"]
                and fixed_quota["target_total_kept_tokens"]
                == adaptive_quota["target_total_kept_tokens"]
                and fixed_quota["observed_total_kept_tokens"]
                == adaptive_quota["observed_total_kept_tokens"],
                f"Adaptive LongMemEval global quota differs: {identifier}.",
            )
            paired_quota_examples += 1
            fixed_hot.append(float(reference["hot_resident_bytes"]))
            adaptive_hot.append(float(treatment["hot_resident_bytes"]))
            controller_times.append(int(adaptive_quota["controller_time_ns"]))
            layer_quotas.extend(int(value) for value in adaptive_quota["layer_kept_tokens"])
        digest_rows.append(
            {
                "example_id": identifier,
                "input_token_ids_sha256": reference["input_token_ids_sha256"],
                "fixed_failure_type": reference["failure_type"],
                "adaptive_failure_type": treatment["failure_type"],
            }
        )
    arm_rows = {
        arm: records for arm, records in zip(ADAPTIVE_QUOTA_ARMS, (fixed, adaptive), strict=True)
    }
    completion_counts = {
        arm: sum(row["failure_type"] == JUDGE_BLOCKED for row in rows)
        for arm, rows in arm_rows.items()
    }
    return {
        "quality": {
            "status": "unverified",
            "official_metric_status": "blocked",
            "officially_scored_examples": 0,
            "proxy_metric_substitution": False,
            "judge_blocked_interpreted_as_quality_failure": False,
        },
        "response_generation": {
            "completed_by_arm": completion_counts,
            "completion_rate_by_arm": {
                arm: count / expected_examples for arm, count in completion_counts.items()
            },
            "adaptive_minus_fixed_completion_rate": (
                completion_counts[ADAPTIVE_QUOTA_ARMS[1]]
                - completion_counts[ADAPTIVE_QUOTA_ARMS[0]]
            )
            / expected_examples,
            "paired_completion_outcomes": dict(sorted(pairing.items())),
        },
        "measurements_by_arm": {arm: _arm_measurements(rows) for arm, rows in arm_rows.items()},
        "by_question_type": {
            question_type: {arm: dict(counter) for arm, counter in by_arm.items()}
            for question_type, by_arm in sorted(question_types.items())
        },
        "initial_prefill_physical": {
            "paired_successful_quota_examples": paired_quota_examples,
            "maximum_global_kept_token_relative_error": max(quota_errors, default=0.0),
            "same_initial_global_token_budget_verified": True,
            "fixed_hot_resident_bytes": _distribution(fixed_hot),
            "adaptive_hot_resident_bytes": _distribution(adaptive_hot),
            "adaptive_controller_time_ns": _distribution(controller_times),
            "adaptive_layer_quota": _distribution(layer_quotas),
        },
        "paired_record_digest": _canonical_digest(digest_rows),
        "confirmation_gate": {
            "available": False,
            "passed": None,
            "classification": "unverified",
            "reason": "official GPT-4o LongMemEval judge was not executed",
        },
    }


def summarize(
    *, adaptive_manifest_path: Path, natural_manifest_path: Path, result_root: Path
) -> dict[str, Any]:
    adaptive_manifest = json.loads(adaptive_manifest_path.read_text())
    validate_manifest(adaptive_manifest)
    natural_manifest = json.loads(natural_manifest_path.read_text())
    validate_natural_manifest(natural_manifest)
    adaptive_digest = sha256(adaptive_manifest_path)
    natural_digest = sha256(natural_manifest_path)
    cell_paths = {arm: result_root / arm / "cell.json" for arm in ADAPTIVE_QUOTA_ARMS}
    _require(all(path.is_file() for path in cell_paths.values()), "Missing adaptive cells.")
    cells = {arm: json.loads(path.read_text()) for arm, path in cell_paths.items()}
    first = cells[ADAPTIVE_QUOTA_ARMS[0]]
    selection_metadata = _dependency(first.get("fixed_baseline_selection"), "fixed selection")
    selection = json.loads(Path(selection_metadata["path"]).read_text())
    inventory_metadata = _dependency(first.get("dataset_inventory"), "dataset inventory")
    expected_identifiers = expected_example_identifiers(
        benchmark="LongMemEval",
        manifest=natural_manifest,
        inventory_path=Path(inventory_metadata["path"]),
    )
    _require(
        len(expected_identifiers) == EXPECTED_EXAMPLES,
        "Adaptive LongMemEval identifier count drifted.",
    )
    revisions = expected_record_revisions("LongMemEval", natural_manifest)
    score_verifier = build_score_verifier(
        benchmark="LongMemEval",
        manifest=natural_manifest,
        manifest_path=natural_manifest_path,
        cell=first,
    )
    allowed_failures = set(adaptive_manifest["failure_reporting"]["allowed_failure_types"])
    validation_evidence: dict[str, dict[str, Any]] = {}
    generation_audits: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    adaptive_dependencies: dict[str, dict[str, Any]] = {}
    for arm in ADAPTIVE_QUOTA_ARMS:
        base_audit, _base_dependencies = audit_arm(
            benchmark="LongMemEval",
            arm=arm,
            artifact_path=cell_paths[arm],
            expected_examples=EXPECTED_EXAMPLES,
            manifest_digest=natural_digest,
            allowed_failures=allowed_failures,
            expected_revisions=revisions,
            expected_seed=adaptive_manifest["benchmark"]["generation_seed"],
            expected_identifiers=expected_identifiers,
            score_verifier=score_verifier,
        )
        validation_evidence[arm] = {
            key: base_audit[key]
            for key in (
                "terminal",
                "expected_examples",
                "accounted_examples",
                "raw_cell",
                "raw_record_digest_set_sha256",
                "source_implementation",
                "runtime_kvpress_binding",
                "run_identity_verified",
                "terminal_measurement_schema_verified",
                "dataset_example_identities_verified",
            )
        }
        adaptive_dependencies[arm] = _verify_adaptive_cell(
            cells[arm],
            arm=arm,
            adaptive_manifest_digest=adaptive_digest,
            selection=selection,
            selection_digest=selection_metadata["sha256"],
        )
        records[arm] = _records(Path(cells[arm]["raw_records"]["path"]))
        generation_audits[arm] = verify_generation_records(
            records[arm], arm=arm, allowed_failures=allowed_failures
        )
    _require(
        cells[ADAPTIVE_QUOTA_ARMS[0]]["p3_sequence_decision"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["p3_sequence_decision"]
        and cells[ADAPTIVE_QUOTA_ARMS[0]]["adaptive_prerequisites"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["adaptive_prerequisites"],
        "Adaptive LongMemEval arms used different frozen prerequisites.",
    )
    analysis = analyze_generation_pairs(
        records[ADAPTIVE_QUOTA_ARMS[0]], records[ADAPTIVE_QUOTA_ARMS[1]]
    )
    return {
        "schema_version": 1,
        "experiment_id": SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "classification": "unverified",
        "coverage": {
            "arms": list(ADAPTIVE_QUOTA_ARMS),
            "predictions_per_arm": EXPECTED_EXAMPLES,
            "paired_predictions_total": EXPECTED_EXAMPLES * 2,
        },
        "audit": {
            "required_arms_terminal": True,
            "terminal_arms": 2,
            "predictions_per_arm": EXPECTED_EXAMPLES,
            "total_predictions": EXPECTED_EXAMPLES * 2,
            "paired_examples": EXPECTED_EXAMPLES,
            "all_raw_records_verified": True,
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
            "successful_response_generation_retained": True,
            "judge_blocked_not_scored_as_zero": True,
            "official_scores_verified": False,
            "proxy_metric_substitution": False,
            "adaptive_allocation_scope": "initial context prefill only",
            "continuous_refresh_claim_available": False,
            "outcome_dependent_execution": False,
        },
        "generation_audits": generation_audits,
        "validation_evidence": validation_evidence,
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
        description="Audit judge-blocked adaptive-quota LongMemEval generation results."
    )
    parser.add_argument(
        "--adaptive-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longmemeval-v1.json"
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
            "longmemeval-qwen3-4b"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "longmemeval-qwen3-4b.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Adaptive LongMemEval summarization requires a clean source tree.")
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
