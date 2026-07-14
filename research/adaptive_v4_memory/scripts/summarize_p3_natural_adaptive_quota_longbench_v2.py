from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from run_p3_longbench_v2 import (
    ADAPTIVE_QUOTA_ARMS,
    load_adaptive_prerequisite,
)
from run_p3_mrcr import atomic_json
from run_p3_natural_ruler import compatibility_arm_config
from summarize_p2_core_matrix import holm_bonferroni
from summarize_p3_natural_adaptive_quota_ruler import verify_quota_audit
from summarize_p3_natural_benchmark import (
    _records,
    audit_arm,
    build_score_verifier,
    expected_example_identifiers,
    expected_record_revisions,
)
from validate_p3_natural_adaptive_quota_longbench_v2_manifest import (
    CATEGORIES,
    PREDICTIONS_PER_ARM,
    validate_manifest,
)
from validate_p3_natural_suite_manifest import validate_manifest as validate_natural_manifest
from verify_p3_natural_model import sha256

SUMMARY_EXPERIMENT_ID = "p3-natural-adaptive-quota-longbench-v2-audit-v1"
RUNNER = Path("research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dependency(metadata: Any, label: str) -> dict[str, str]:
    _require(isinstance(metadata, dict), f"Missing adaptive LongBench-v2 {label} dependency.")
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == sha256(path),
        f"Adaptive LongBench-v2 {label} dependency drifted.",
    )
    return {"path": str(path), "sha256": metadata["sha256"]}


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def paired_bootstrap(
    differences: Iterable[float], *, seed: int, resamples: int
) -> dict[str, Any]:
    values = np.asarray(tuple(differences), dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 0, "Paired bootstrap requires examples.")
    unique, counts = np.unique(values, return_counts=True)
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(len(values), counts / len(values), size=resamples)
    means = sampled @ unique / len(values)
    lower, upper = np.quantile(means, (0.025, 0.975))
    lower_tail = (np.count_nonzero(means <= 0.0) + 1) / (resamples + 1)
    upper_tail = (np.count_nonzero(means >= 0.0) + 1) / (resamples + 1)
    return {
        "paired_examples": len(values),
        "mean_difference": float(values.mean()),
        "mean_difference_percentage_points": float(values.mean() * 100.0),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "paired_bootstrap_95_ci_percentage_points": [
            float(lower * 100.0),
            float(upper * 100.0),
        ],
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
    }


def canonical_category(value: Any) -> str:
    _require(isinstance(value, str) and bool(value.strip()), "Missing LongBench-v2 domain.")
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    _require(normalized in CATEGORIES, f"Unknown LongBench-v2 category: {value}.")
    return normalized


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
        f"Adaptive LongBench-v2 cell identity drifted for {arm}.",
    )
    _dependency(cell.get("adaptive_quota_manifest"), "adaptive manifest")
    prerequisites = cell.get("adaptive_prerequisites")
    _require(
        isinstance(prerequisites, dict)
        and set(prerequisites) == {"adaptive_ruler", "baseline_longbench_v2"},
        f"Adaptive LongBench-v2 prerequisite set drifted for {arm}.",
    )
    assert isinstance(prerequisites, dict)
    expected = {
        "adaptive_ruler": ("p3-natural-adaptive-quota-ruler-audit-v1", 65_000),
        "baseline_longbench_v2": ("p3-natural-longbench-v2-audit-v1", 1_006),
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
        _require(verified == metadata, f"Adaptive LongBench-v2 prerequisite drifted: {name}.")
        verified_prerequisites[name] = verified
    _require(
        identity.get("adaptive_prerequisite_sha256")
        == {name: row["sha256"] for name, row in verified_prerequisites.items()},
        f"Adaptive LongBench-v2 prerequisite identity drifted for {arm}.",
    )
    decision = cell.get("p3_sequence_decision", {})
    dependencies = decision.get("dependencies", {})
    _require(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {"primary_core", "primary_causal", "nine_seed_causal", "fixed_selection"},
        f"Adaptive LongBench-v2 sequence dependency set drifted for {arm}.",
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
        f"Adaptive LongBench-v2 sequence identity drifted for {arm}.",
    )
    return {
        "prerequisites": verified_prerequisites,
        "sequence_dependencies": verified_sequence,
        "decision": decision,
    }


def verify_quota_records(
    records: list[dict[str, Any]],
    *,
    arm: str,
    allowed_failures: set[str],
    layer_count: int = 36,
    adaptive_arm: str = ADAPTIVE_QUOTA_ARMS[1],
) -> None:
    for row in records:
        failure = row.get("failure_type")
        if row["status"] == "failure":
            _require(
                failure in allowed_failures,
                f"Unregistered adaptive LongBench-v2 failure: {row['example_id']}/{arm}.",
            )
        else:
            _require(
                failure is None,
                f"Scored adaptive LongBench-v2 example carries a failure: {row['example_id']}.",
            )
        token_digest = row.get("input_token_ids_sha256")
        _require(
            isinstance(token_digest, str) and len(token_digest) == 64,
            f"Missing exact token digest: {row['example_id']}.",
        )
        audit = row.get("quota_physical_audit")
        if audit is None:
            _require(
                row["status"] == "failure" and row["hot_resident_bytes"] == 0,
                f"Successful context prefill lacks quota audit: {row['example_id']}/{arm}.",
            )
            row["verified_quota"] = None
        else:
            row["verified_quota"] = verify_quota_audit(
                audit,
                arm=arm,
                layer_count=layer_count,
                adaptive_arm=adaptive_arm,
            )
            _require(
                row["hot_resident_bytes"] > 0,
                f"Audited context prefill has zero hot bytes: {row['example_id']}/{arm}.",
            )
        row["canonical_category"] = canonical_category(row.get("domain"))


def _failure_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = Counter(
        str(row["failure_type"]) for row in records if row["status"] == "failure"
    )
    return {
        "examples": len(records),
        "scored_examples": sum(row["status"] == "scored" for row in records),
        "failures_by_type": dict(sorted(failures.items())),
        "failure_rate": sum(failures.values()) / len(records),
        "mean_accuracy_failures_zero": float(
            np.mean(
                [float(row["score"]) if row["status"] == "scored" else 0.0 for row in records]
            )
        ),
    }


def _descriptive_slices(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], field: str
) -> list[dict[str, Any]]:
    by_value: dict[str, list[float]] = defaultdict(list)
    for fixed, adaptive in pairs:
        value = str(fixed.get(field))
        fixed_score = float(fixed["score"]) if fixed["status"] == "scored" else 0.0
        adaptive_score = float(adaptive["score"]) if adaptive["status"] == "scored" else 0.0
        by_value[value].append(adaptive_score - fixed_score)
    return [
        {
            "slice": value,
            "paired_examples": len(values),
            "mean_difference": float(np.mean(values)),
            "confirmation_gate_role": False,
        }
        for value, values in sorted(by_value.items())
        if len(values) >= 10
    ]


def analyze_pairs(
    fixed: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    expected_examples: int = PREDICTIONS_PER_ARM,
    categories: tuple[str, ...] = CATEGORIES,
    arms: tuple[str, str] = ADAPTIVE_QUOTA_ARMS,
) -> dict[str, Any]:
    _require(
        len(fixed) == len(adaptive) == expected_examples,
        "Adaptive LongBench-v2 paired example count drifted.",
    )
    fixed_by_id = {row["example_id"]: row for row in fixed}
    adaptive_by_id = {row["example_id"]: row for row in adaptive}
    _require(
        len(fixed_by_id) == len(adaptive_by_id) == expected_examples
        and set(fixed_by_id) == set(adaptive_by_id),
        "Adaptive LongBench-v2 paired identities drifted.",
    )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    differences: list[float] = []
    by_category: dict[str, list[float]] = {category: [] for category in categories}
    paired_digest_rows: list[dict[str, Any]] = []
    for identifier in sorted(fixed_by_id):
        reference = fixed_by_id[identifier]
        treatment = adaptive_by_id[identifier]
        _require(
            all(
                reference.get(key) == treatment.get(key)
                for key in (
                    "raw_prompt_sha256",
                    "input_token_ids_sha256",
                    "exact_input_tokens",
                    "generation_reserve_tokens",
                    "domain",
                    "sub_domain",
                    "difficulty",
                    "length_stratum",
                )
            ),
            f"Adaptive LongBench-v2 paired input drifted: {identifier}.",
        )
        category = canonical_category(reference.get("domain"))
        _require(category == reference["canonical_category"], "Category audit drifted.")
        reference_score = float(reference["score"]) if reference["status"] == "scored" else 0.0
        treatment_score = float(treatment["score"]) if treatment["status"] == "scored" else 0.0
        difference = treatment_score - reference_score
        differences.append(difference)
        by_category[category].append(difference)
        pairs.append((reference, treatment))
        paired_digest_rows.append(
            {
                "example_id": identifier,
                "input_token_ids_sha256": reference["input_token_ids_sha256"],
                "fixed_score": reference_score,
                "adaptive_score": treatment_score,
            }
        )
    _require(
        all(by_category[category] for category in categories),
        "Adaptive LongBench-v2 category grid is incomplete.",
    )
    statistics = manifest["statistics"]
    seed = int(statistics["paired_bootstrap_seed"])
    resamples = int(statistics["paired_bootstrap_resamples"])
    overall = paired_bootstrap(differences, seed=seed, resamples=resamples)
    category_rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for index, category in enumerate(categories, start=1):
        estimate = paired_bootstrap(
            by_category[category], seed=seed + index, resamples=resamples
        )
        raw_p[category] = float(estimate["two_sided_bootstrap_p"])
        category_rows.append({"category": category, **estimate})
    _require(
        len(raw_p) == statistics["holm_family_size"],
        "Adaptive LongBench-v2 Holm family size drifted.",
    )
    adjusted = holm_bonferroni(raw_p)
    for row in category_rows:
        row["holm_adjusted_p"] = adjusted[row["category"]]
        row["holm_family_size"] = len(raw_p)
    quota_errors: list[float] = []
    paired_quota_examples = 0
    fixed_hot: list[float] = []
    adaptive_hot: list[float] = []
    adaptive_controller_times: list[int] = []
    adaptive_layer_quotas: list[int] = []
    for reference, treatment in pairs:
        reference_quota = reference.get("verified_quota")
        treatment_quota = treatment.get("verified_quota")
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
                f"Adaptive LongBench-v2 global quota differs: {reference['example_id']}.",
            )
            paired_quota_examples += 1
            fixed_hot.append(float(reference["hot_resident_bytes"]))
            adaptive_hot.append(float(treatment["hot_resident_bytes"]))
            adaptive_controller_times.append(int(treatment_quota["controller_time_ns"]))
            adaptive_layer_quotas.extend(treatment_quota["layer_kept_tokens"])
    fixed_hot_mean = float(np.mean(fixed_hot)) if fixed_hot else 0.0
    adaptive_hot_mean = float(np.mean(adaptive_hot)) if adaptive_hot else 0.0
    hot_relative_difference = (
        abs(adaptive_hot_mean - fixed_hot_mean) / fixed_hot_mean
        if fixed_hot_mean > 0.0
        else (0.0 if adaptive_hot_mean == 0.0 else math.inf)
    )
    fixed_failure = _failure_summary(fixed)
    adaptive_failure = _failure_summary(adaptive)
    gate = manifest["confirmation_gate"]
    checks = {
        "overall_accuracy_difference": overall["mean_difference"]
        >= gate["overall_accuracy_difference_minimum"],
        "paired_bootstrap_lower_bound": overall["paired_bootstrap_95_ci"][0]
        >= gate["paired_bootstrap_lower_bound_minimum"],
        "nonnegative_category_count": sum(
            row["mean_difference"] >= 0.0 for row in category_rows
        )
        >= gate["nonnegative_category_count_minimum"],
        "worst_category_regression": min(row["mean_difference"] for row in category_rows)
        >= gate["worst_category_regression_minimum"],
        "failure_rate_increase": adaptive_failure["failure_rate"]
        - fixed_failure["failure_rate"]
        <= gate["maximum_failure_rate_increase"],
        "initial_global_kept_token_relative_error": max(quota_errors, default=0.0)
        <= gate["maximum_initial_global_kept_token_relative_error"],
        "initial_hot_resident_byte_relative_difference": hot_relative_difference
        <= gate["maximum_initial_hot_resident_byte_relative_difference"],
        "at_least_one_jointly_successful_prefill": paired_quota_examples > 0,
    }
    passed = all(checks.values())
    return {
        "arms": {
            arms[0]: fixed_failure,
            arms[1]: adaptive_failure,
        },
        "overall": overall,
        "by_category": category_rows,
        "descriptive_slices": {
            "sub_domain": _descriptive_slices(pairs, "sub_domain"),
            "difficulty": _descriptive_slices(pairs, "difficulty"),
            "length_stratum": _descriptive_slices(pairs, "length_stratum"),
        },
        "multiplicity": {
            "method": "Holm-Bonferroni",
            "family_size": len(raw_p),
            "family": "six frozen LongBench-v2 capability categories",
        },
        "initial_prefill_physical": {
            "paired_successful_quota_examples": paired_quota_examples,
            "maximum_global_kept_token_relative_error": max(quota_errors, default=0.0),
            "fixed_hot_resident_bytes": _distribution(fixed_hot),
            "adaptive_hot_resident_bytes": _distribution(adaptive_hot),
            "absolute_mean_hot_resident_byte_relative_difference": hot_relative_difference,
            "adaptive_controller_time_ns": _distribution(adaptive_controller_times),
            "adaptive_layer_quota": _distribution(adaptive_layer_quotas),
        },
        "paired_record_digest": _canonical_digest(paired_digest_rows),
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
    _require(all(path.is_file() for path in cell_paths.values()), "Missing adaptive cells.")
    cells = {arm: json.loads(path.read_text()) for arm, path in cell_paths.items()}
    first = cells[ADAPTIVE_QUOTA_ARMS[0]]
    selection_metadata = _dependency(first.get("fixed_baseline_selection"), "fixed selection")
    selection = json.loads(Path(selection_metadata["path"]).read_text())
    inventory_metadata = _dependency(first.get("dataset_inventory"), "dataset inventory")
    expected_identifiers = expected_example_identifiers(
        benchmark="LongBench-v2",
        manifest=natural_manifest,
        inventory_path=Path(inventory_metadata["path"]),
    )
    _require(
        len(expected_identifiers) == PREDICTIONS_PER_ARM,
        "Adaptive LongBench-v2 identifier count drifted.",
    )
    revisions = expected_record_revisions("LongBench-v2", natural_manifest)
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
        adaptive_dependencies[arm] = _verify_adaptive_cell(
            cells[arm],
            arm=arm,
            adaptive_manifest_digest=adaptive_digest,
            selection=selection,
            selection_digest=selection_metadata["sha256"],
        )
        records[arm] = _records(Path(cells[arm]["raw_records"]["path"]))
        verify_quota_records(records[arm], arm=arm, allowed_failures=allowed_failures)
    _require(
        cells[ADAPTIVE_QUOTA_ARMS[0]]["p3_sequence_decision"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["p3_sequence_decision"]
        and cells[ADAPTIVE_QUOTA_ARMS[0]]["adaptive_prerequisites"]
        == cells[ADAPTIVE_QUOTA_ARMS[1]]["adaptive_prerequisites"],
        "Adaptive LongBench-v2 arms used different frozen prerequisites.",
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
            "categories": list(CATEGORIES),
            "predictions_per_arm": PREDICTIONS_PER_ARM,
            "paired_predictions_total": PREDICTIONS_PER_ARM * 2,
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
    parser = argparse.ArgumentParser(description="Audit adaptive-quota LongBench-v2 results.")
    parser.add_argument(
        "--adaptive-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longbench-v2-v1.json"
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
            "longbench-v2-qwen3-4b"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "longbench-v2-qwen3-4b.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Adaptive LongBench-v2 summarization requires a clean source tree.")
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
