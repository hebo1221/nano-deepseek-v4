from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

BENCHMARK_IDS = {
    "RULER": "p3-natural-ruler-audit-v1",
    "SCBench": "p3-natural-scbench-audit-v1",
    "LongBench-v2": "p3-natural-longbench-v2-audit-v1",
    "LongMemEval": "p3-natural-longmemeval-audit-v1",
    "MRCR": "p3-natural-mrcr-audit-v1",
}
DEPENDENCY_IDS = {
    "causal_gate": "p2-causal-ablation-audit-v1",
    "dataset_inventory": "p3-natural-dataset-inventory-v1",
    "fixed_baseline_selection": "p3-fixed-baseline-selection-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Invalid {label} digest.")
    int(value, 16)


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and (
        type(value) is int or math.isfinite(value)
    )


def _finite_nonnegative_number(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def _unit_interval_number(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= value <= 1.0


def _signed_unit_interval_number(value: Any) -> bool:
    return _finite_number(value) and -1.0 <= value <= 1.0


def _close(left: Any, right: Any) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
    )


def _validate_distribution(value: Any, *, observations: int, label: str) -> None:
    _require(isinstance(value, dict), f"Missing {label} distribution.")
    _require(
        type(value.get("observations")) is int
        and value["observations"] == observations,
        f"{label} observations drifted.",
    )
    for field in (
        "mean",
        "sample_standard_deviation",
        "p50",
        "p95",
        "p99",
        "minimum",
        "maximum",
    ):
        item = value.get(field)
        _require(
            _finite_nonnegative_number(item),
            f"{label} {field} drifted.",
        )
    _require(
        value["minimum"]
        <= value["p50"]
        <= value["p95"]
        <= value["p99"]
        <= value["maximum"]
        and value["minimum"] <= value["mean"] <= value["maximum"],
        f"{label} order statistics drifted.",
    )


def audit_benchmark(
    *,
    name: str,
    path: Path,
    expected_examples: int,
    required_arms: tuple[str, ...],
    allowed_failures: set[str],
    manifest_digest: str,
    model_snapshot_digest: str,
    expected_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file(), f"Missing natural benchmark summary: {path}")
    payload = json.loads(path.read_text())
    _require(payload.get("experiment_id") == BENCHMARK_IDS[name], f"Wrong {name} audit id.")
    _require(payload.get("benchmark") == name, f"Wrong benchmark label for {name}.")
    _require(
        payload.get("generation_seed") == expected_seed,
        f"{name} generation seed drifted.",
    )
    _require(payload.get("source", {}).get("dirty") is False, f"Dirty {name} evidence.")
    _require(
        payload.get("experiment_manifest", {}).get("sha256") == manifest_digest,
        f"{name} manifest dependency drifted.",
    )
    audit = payload.get("audit", {})
    _require(audit.get("all_raw_artifacts_verified") is True, f"{name} raw audit failed.")
    _require(
        audit.get("all_source_implementations_verified") is True,
        f"{name} source implementation audit failed.",
    )
    _require(
        audit.get("all_record_revisions_verified") is True,
        f"{name} record revision audit failed.",
    )
    _require(
        audit.get("all_run_identities_verified") is True,
        f"{name} run identity audit failed.",
    )
    _require(
        audit.get("all_terminal_measurement_schema_verified") is True,
        f"{name} terminal measurement audit failed.",
    )
    _require(
        audit.get("all_failure_accounting_complete") is True,
        f"{name} failure accounting is incomplete.",
    )
    _require(
        audit.get("all_required_arms_input_paired") is True,
        f"{name} required arms are not input-paired.",
    )
    _sha256_value(audit.get("raw_record_digest_set_sha256"), f"{name} raw-record set")
    arms = payload.get("arms", {})
    _require(set(required_arms).issubset(arms), f"{name} is missing a required baseline arm.")
    arm_rows: dict[str, Any] = {}
    for arm in required_arms:
        row = arms[arm]
        _require(
            row.get("run_identity_verified") is True
            and row.get("terminal_measurement_schema_verified") is True,
            f"{name}/{arm} execution evidence audit failed.",
        )
        failures = row.get("failures_by_type", {})
        _require(isinstance(failures, dict), f"{name}/{arm} failure map invalid.")
        _require(
            set(failures).issubset(allowed_failures),
            f"{name}/{arm} contains an unregistered failure type.",
        )
        _require(
            all(_nonnegative_integer(count) for count in failures.values()),
            f"{name}/{arm} has an invalid failure count.",
        )
        scored = row.get("scored_examples")
        accounted = row.get("accounted_examples")
        _require(_nonnegative_integer(scored), f"{name}/{arm} score count invalid.")
        _require(_nonnegative_integer(accounted), f"{name}/{arm} accounted count invalid.")
        _require(
            accounted == scored + sum(failures.values()),
            f"{name}/{arm} score and failure counts do not close.",
        )
        _require(
            row.get("expected_examples") == expected_examples
            and accounted == expected_examples
            and row.get("terminal") is True,
            f"{name}/{arm} does not account for every frozen example.",
        )
        conservative_mean = row.get("mean_score_over_all_expected_failures_zero")
        _require(
            _unit_interval_number(conservative_mean),
            f"{name}/{arm} conservative quality is missing.",
        )
        scored_mean = row.get("mean_score_over_scored")
        _require(
            _unit_interval_number(scored_mean) if scored > 0 else scored_mean is None,
            f"{name}/{arm} scored-only quality drifted.",
        )
        failure_rate = row.get("failure_rate")
        _require(
            _unit_interval_number(failure_rate)
            and _close(failure_rate, sum(failures.values()) / expected_examples),
            f"{name}/{arm} failure rate drifted.",
        )
        measurements = row.get("measurements", {})
        for metric in (
            "exact_input_tokens",
            "latency_ms",
            "peak_hbm_bytes",
            "hot_resident_bytes",
        ):
            _validate_distribution(
                measurements.get("all_terminal_attempts", {}).get(metric),
                observations=expected_examples,
                label=f"{name}/{arm} all-terminal {metric}",
            )
            if scored > 0:
                _validate_distribution(
                    measurements.get("scored_only", {}).get(metric),
                    observations=scored,
                    label=f"{name}/{arm} scored-only {metric}",
                )
            else:
                _require(
                    measurements.get("scored_only", {}).get(metric) is None,
                    f"{name}/{arm} empty scored-only {metric} drifted.",
                )
        arm_rows[arm] = {
            "terminal": True,
            "expected_examples": expected_examples,
            "scored_examples": scored,
            "failed_examples": sum(failures.values()),
            "failures_by_type": failures,
            "mean_score_over_scored": scored_mean,
            "mean_score_over_all_expected_failures_zero": conservative_mean,
            "failure_rate": failure_rate,
            "measurements": measurements,
        }
    contrast = payload.get("paired_quality_contrast", {})
    expected_clusters = 1_844 if name == "SCBench" else expected_examples
    failure_pairing = contrast.get("failure_pairing", {})
    expected_failure_pairing = {
        "both_scored",
        "candidate_only_failed",
        "comparator_only_failed",
        "both_failed",
    }
    _require(
        type(contrast.get("paired_examples")) is int
        and contrast["paired_examples"] == expected_examples
        and type(contrast.get("paired_clusters")) is int
        and contrast["paired_clusters"] == expected_clusters
        and contrast.get("cluster_unit")
        == ("shared-context-row" if name == "SCBench" else "example")
        and contrast.get("candidate") == required_arms[1]
        and contrast.get("comparator") == required_arms[0]
        and contrast.get("failure_as_zero") is True
        and contrast.get("bootstrap_resamples") == 10_000
        and contrast.get("confidence_level") == 0.95
        and contrast.get("bootstrap_seed")
        == int.from_bytes(
            hashlib.sha256(f"p3-natural:{name}:quality".encode()).digest()[:8],
            "big",
        ),
        f"{name} paired quality contrast is incomplete.",
    )
    _require(
        isinstance(failure_pairing, dict)
        and set(failure_pairing) == expected_failure_pairing
        and all(_nonnegative_integer(value) for value in failure_pairing.values())
        and _nonnegative_integer(contrast.get("jointly_scored_examples"))
        and contrast["jointly_scored_examples"] == failure_pairing["both_scored"]
        and sum(failure_pairing.values()) == expected_examples,
        f"{name} paired failure accounting drifted.",
    )
    mean_difference = contrast.get("mean_difference")
    mean_percentage_points = contrast.get("mean_difference_percentage_points")
    confidence_interval = contrast.get("paired_bootstrap_95_ci")
    confidence_interval_pp = contrast.get(
        "paired_bootstrap_95_ci_percentage_points"
    )
    _require(
        _finite_number(mean_difference)
        and -1.0 <= mean_difference <= 1.0
        and _close(mean_percentage_points, mean_difference * 100.0)
        and isinstance(confidence_interval, list)
        and len(confidence_interval) == 2
        and all(
            _finite_number(value) and -1.0 <= value <= 1.0
            for value in confidence_interval
        )
        and confidence_interval[0] <= confidence_interval[1]
        and isinstance(confidence_interval_pp, list)
        and len(confidence_interval_pp) == 2
        and all(
            _close(value, confidence_interval[index] * 100.0)
            for index, value in enumerate(confidence_interval_pp)
        )
        and _unit_interval_number(contrast.get("two_sided_bootstrap_p"))
        and _finite_nonnegative_number(
            contrast.get("cluster_mean_sample_standard_deviation")
        ),
        f"{name} paired quality statistics drifted.",
    )
    measurement_contrasts = payload.get("paired_measurement_contrasts", {})
    _require(
        isinstance(measurement_contrasts, dict)
        and set(measurement_contrasts)
        == {"latency_ms", "peak_hbm_bytes", "hot_resident_bytes"}
        and all(
            row.get("paired_examples") == expected_examples
            and row.get("candidate") == required_arms[1]
            and row.get("comparator") == required_arms[0]
            and row.get("includes_terminal_failures") is True
            for row in measurement_contrasts.values()
        ),
        f"{name} paired measurement contrasts are incomplete.",
    )
    for metric, row in measurement_contrasts.items():
        candidate_mean = row.get("candidate_mean")
        comparator_mean = row.get("comparator_mean")
        mean_difference = row.get("mean_paired_difference")
        ratio = row.get("ratio_of_means")
        _require(
            _finite_nonnegative_number(candidate_mean)
            and _finite_nonnegative_number(comparator_mean)
            and _finite_number(mean_difference)
            and _close(mean_difference, candidate_mean - comparator_mean)
            and (
                _close(ratio, candidate_mean / comparator_mean)
                if comparator_mean > 0.0
                else ratio is None
            ),
            f"{name} paired {metric} statistics drifted.",
        )
    conditional = payload.get("conditional_arms", {})
    _require(
        all(
            row.get("status") in {"complete", "incompatible", "withheld-by-causal-gate"}
            for row in conditional.values()
        ),
        f"{name} conditional-arm disposition is incomplete.",
    )
    result = {
        "terminal": True,
        "native_and_fixed_terminal": all(arm_rows[arm]["terminal"] for arm in required_arms),
        "expected_examples_per_required_arm": expected_examples,
        "required_arms": arm_rows,
        "paired_quality_contrast": contrast,
        "paired_measurement_contrasts": measurement_contrasts,
        "conditional_arms": conditional,
        "summary": {"path": str(path), "sha256": sha256(path)},
    }
    dependencies = {
        "causal_gate": payload.get("causal_gate"),
        "dataset_inventory": payload.get("dataset_inventory"),
        "fixed_baseline_selection": payload.get("fixed_baseline_selection"),
        "model_snapshot_digest_set_sha256": payload.get("model_snapshot_digest_set_sha256"),
        "benchmark_dataset_digest_set_sha256": payload.get("benchmark_dataset_digest_set_sha256"),
    }
    _sha256_value(dependencies["model_snapshot_digest_set_sha256"], f"{name} model set")
    _require(
        dependencies["model_snapshot_digest_set_sha256"] == model_snapshot_digest,
        f"{name} model snapshot does not match the frozen manifest.",
    )
    if name == "RULER":
        _sha256_value(
            dependencies["benchmark_dataset_digest_set_sha256"],
            "RULER dataset manifest set",
        )
    for dependency_name in (
        "causal_gate",
        "dataset_inventory",
        "fixed_baseline_selection",
    ):
        metadata = dependencies[dependency_name]
        _require(isinstance(metadata, dict), f"Missing {name} {dependency_name} dependency.")
        dependency_path = Path(metadata.get("path", ""))
        _require(dependency_path.is_file(), f"Missing {name} dependency: {dependency_path}")
        _require(
            metadata.get("sha256") == sha256(dependency_path),
            f"{name} {dependency_name} digest drifted.",
        )
        dependency = json.loads(dependency_path.read_text())
        _require(
            dependency.get("experiment_id") == DEPENDENCY_IDS[dependency_name],
            f"{name} {dependency_name} has the wrong experiment id.",
        )
        _require(
            dependency.get("source", {}).get("dirty") is False,
            f"{name} {dependency_name} was produced from a dirty source tree.",
        )
    return result, dependencies


def audit_safety_stress(
    *,
    path: Path,
    manifest: dict[str, Any],
    natural_manifest_digest: str,
    model_snapshot_digest: str,
) -> dict[str, Any]:
    _require(path.is_file(), f"Missing safety stress summary: {path}")
    payload = json.loads(path.read_text())
    _require(
        payload.get("experiment_id") == "p3-safety-stress-audit-v1"
        and payload.get("source", {}).get("dirty") is False,
        "Safety stress audit is missing, dirty, or has the wrong id.",
    )
    contract = manifest["suite_audit"]["safety_stress"]
    safety_manifest_path = Path(contract["manifest"])
    _require(safety_manifest_path.is_file(), "Missing frozen safety stress manifest.")
    safety_manifest = json.loads(safety_manifest_path.read_text())
    _require(
        payload.get("manifest", {}).get("path") == str(safety_manifest_path)
        and payload.get("manifest", {}).get("sha256") == sha256(safety_manifest_path),
        "Safety stress manifest dependency drifted.",
    )
    expected_examples = contract["examples_per_required_arm"]
    expected_families = tuple(safety_manifest.get("families", {}))
    expected_contexts = tuple(safety_manifest.get("context_targets_tokens", ()))
    expected_per_slice = safety_manifest.get("examples_per_family_context")
    allowed_failures = set(safety_manifest.get("failure_accounting", ()))
    _require(
        safety_manifest.get("expected_examples_per_arm") == expected_examples
        and len(expected_families) == contract["families"]
        and len(expected_contexts) == contract["contexts"]
        and type(expected_per_slice) is int
        and expected_per_slice > 0
        and expected_examples
        == len(expected_families) * len(expected_contexts) * expected_per_slice,
        "Safety stress frozen Cartesian contract drifted.",
    )
    audit = payload.get("audit", {})
    expected_audit = {
        "required_arms_terminal": True,
        "failure_accounting_complete": True,
        "input_pairing_verified": True,
        "source_implementations_verified": True,
        "coordinate_grid_verified": True,
        "record_revisions_verified": True,
        "terminal_measurement_schema_verified": True,
        "target_and_canary_pairing_verified": True,
        "protected_prefix_physical_budget_verified": True,
        "raw_artifact_digests_verified": True,
        "statistical_schema_verified": True,
        "examples_accounted_per_arm": expected_examples,
        "families_terminal": contract["families"],
        "contexts_terminal": contract["contexts"],
    }
    _require(
        all(audit.get(key) == value for key, value in expected_audit.items()),
        "Safety stress coverage, pairing, or failure accounting is incomplete.",
    )
    dependencies = payload.get("dependencies", {})
    _require(
        dependencies.get("natural_manifest") == natural_manifest_digest
        and dependencies.get("model_snapshot") == model_snapshot_digest,
        "Safety stress natural-manifest or model dependency drifted.",
    )
    arms = payload.get("arms", {})
    safety_arms = tuple(contract["required_arms"])
    _require(set(arms) == set(safety_arms), "Safety stress arm set drifted.")
    contrast = payload.get("protected_prefix_causal_contrast", {})
    interval = contrast.get("paired_bootstrap_95_ci")
    _require(
        type(contrast.get("paired_examples")) is int
        and contrast["paired_examples"] == expected_examples
        and isinstance(contrast.get("estimand"), str)
        and "operational failures score zero" in contrast["estimand"]
        and _signed_unit_interval_number(
            contrast.get("mean_success_rate_difference")
        )
        and isinstance(interval, list)
        and len(interval) == 2
        and all(_signed_unit_interval_number(value) for value in interval)
        and interval[0] <= interval[1]
        and all(
            _nonnegative_integer(contrast.get(field))
            for field in ("wins", "ties", "losses")
        )
        and contrast["wins"] + contrast["ties"] + contrast["losses"]
        == expected_examples
        and _close(
            contrast["mean_success_rate_difference"],
            (contrast["wins"] - contrast["losses"]) / expected_examples,
        )
        and _unit_interval_number(contrast.get("exact_two_sided_paired_pvalue"))
        and _nonnegative_integer(
            contrast.get("physically_comparable_scored_pairs")
        )
        and contrast["physically_comparable_scored_pairs"] <= expected_examples
        and contrast.get("resident_bytes_equal_for_comparable_pairs") is True
        and contrast.get("bootstrap_seed") == safety_manifest.get("seed")
        and contrast.get("bootstrap_replicates") == 10_000,
        "Safety protected-prefix causal contrast is incomplete or not memory matched.",
    )
    source_implementations: set[str] = set()
    for arm in safety_arms:
        row = arms[arm]
        failures = row.get("failures_by_type")
        scored = row.get("scored_examples")
        slices = row.get("slices")
        _require(
            row.get("terminal") is True
            and row.get("expected_examples") == expected_examples
            and _nonnegative_integer(scored)
            and isinstance(failures, dict)
            and set(failures).issubset(allowed_failures)
            and all(_nonnegative_integer(value) for value in failures.values())
            and scored + sum(failures.values()) == expected_examples
            and _unit_interval_number(row.get("macro_success_rate_failures_zero"))
            and _nonnegative_integer(row.get("total_leakage_events"))
            and row["total_leakage_events"] <= expected_examples
            and isinstance(slices, list)
            and len(slices) == len(expected_families) * len(expected_contexts),
            f"Safety stress arm is incomplete: {arm}.",
        )
        raw_cell = row.get("raw_cell", {})
        raw_cell_path = Path(raw_cell.get("path", ""))
        _require(
            raw_cell_path.is_file()
            and raw_cell.get("sha256") == sha256(raw_cell_path),
            f"Safety stress raw cell drifted: {arm}.",
        )
        _sha256_value(
            row.get("paired_prompt_digest_set_sha256"), f"{arm} safety pairing"
        )
        source = row.get("source_implementation", {})
        _require(
            isinstance(source, dict)
            and isinstance(source.get("commit"), str)
            and bool(source["commit"]),
            f"Safety stress source implementation is incomplete: {arm}.",
        )
        _sha256_value(source.get("implementation_sha256"), f"{arm} safety runner")
        _sha256_value(source.get("workload_sha256"), f"{arm} safety workload")
        source_implementations.add(json.dumps(source, sort_keys=True))
        seen_slices: set[tuple[str, int]] = set()
        for slice_row in slices:
            family = slice_row.get("family")
            context = slice_row.get("context_target")
            slice_scored = slice_row.get("scored_examples")
            slice_failures = slice_row.get("failures")
            leakage_events = slice_row.get("leakage_events")
            _require(
                family in expected_families
                and context in expected_contexts
                and (family, context) not in seen_slices
                and slice_row.get("expected_examples") == expected_per_slice
                and _nonnegative_integer(slice_scored)
                and _nonnegative_integer(slice_failures)
                and slice_scored + slice_failures == expected_per_slice
                and _unit_interval_number(
                    slice_row.get("success_rate_failures_zero")
                )
                and _nonnegative_integer(leakage_events)
                and leakage_events <= expected_per_slice
                and _unit_interval_number(
                    slice_row.get("leakage_rate_all_expected")
                )
                and _close(
                    slice_row["leakage_rate_all_expected"],
                    leakage_events / expected_per_slice,
                ),
                f"Safety stress slice statistics drifted: {arm}.",
            )
            seen_slices.add((family, context))
        _require(
            seen_slices
            == {
                (family, context)
                for family in expected_families
                for context in expected_contexts
            }
            and _close(
                row["macro_success_rate_failures_zero"],
                sum(
                    slice_row["success_rate_failures_zero"]
                    for slice_row in slices
                )
                / len(slices),
            )
            and row["total_leakage_events"]
            == sum(slice_row["leakage_events"] for slice_row in slices)
            and row.get("worst_slice")
            == min(slices, key=lambda item: item["success_rate_failures_zero"]),
            f"Safety stress arm aggregates drifted: {arm}.",
        )
    _require(
        len(source_implementations) == 1,
        "Safety stress arms used different source implementations.",
    )
    return {
        "terminal": True,
        "examples_per_required_arm": contract["examples_per_required_arm"],
        "families": contract["families"],
        "contexts": contract["contexts"],
        "required_arms": list(safety_arms),
        "protected_prefix_physical_budget_verified": True,
        "source_implementations_verified": True,
        "raw_artifact_digests_verified": True,
        "statistical_schema_verified": True,
        "protected_prefix_causal_contrast": contrast,
        "summary": {"path": str(path), "sha256": sha256(path)},
        "arms": arms,
        "claim_boundary": payload.get("claim_boundary"),
    }


def audit_natural_safety(
    *, path: Path, manifest: dict[str, Any], model_snapshot_digest: str
) -> dict[str, Any]:
    _require(path.is_file(), f"Missing natural safety summary: {path}")
    payload = json.loads(path.read_text())
    _require(
        payload.get("experiment_id") == "p3-natural-safety-suite-audit-v1"
        and payload.get("source", {}).get("dirty") is False,
        "Natural safety suite audit is missing, dirty, or has the wrong id.",
    )
    contract = manifest["suite_audit"]["natural_safety"]
    safety_manifest_path = Path(contract["manifest"])
    _require(safety_manifest_path.is_file(), "Missing frozen natural safety manifest.")
    safety_manifest = json.loads(safety_manifest_path.read_text())
    _require(
        safety_manifest.get("model", {}).get("snapshot_digest_set_sha256")
        == model_snapshot_digest
        and payload.get("manifest", {}).get("path") == str(safety_manifest_path)
        and payload.get("manifest", {}).get("sha256") == sha256(safety_manifest_path),
        "Natural safety manifest or model dependency drifted.",
    )
    audit = payload.get("audit", {})
    _require(
        audit.get("required_arms") == contract["required_arms"]
        and audit.get("longsafety_generation_terminal") is True
        and audit.get("longsafety_input_pairing_verified") is True
        and audit.get("longsafety_expected_generations_per_arm")
        == contract["longsafety_generations_per_arm"]
        and audit.get("longsafety_official_judge_status")
        == contract["longsafety_official_judge_status"]
        and audit.get("longsafety_safety_scores_reported") is False
        and audit.get("ifeval_official_terminal") is True
        and audit.get("ifeval_input_pairing_verified") is True
        and audit.get("ifeval_expected_prompts_per_arm")
        == contract["ifeval_prompts_per_arm"]
        and audit.get("failure_accounting_complete") is True
        and audit.get("source_implementations_verified") is True
        and audit.get("comparative_long_context_safety_claim_available")
        is contract["comparative_long_context_safety_claim_available"],
        "Natural safety coverage or claim boundary is incomplete.",
    )
    for section in ("longsafety", "ifeval"):
        artifact = payload.get(section, {}).get("summary", {})
        artifact_path = Path(artifact.get("path", ""))
        _require(
            artifact_path.is_file() and artifact.get("sha256") == sha256(artifact_path),
            f"Natural safety child summary drifted: {section}.",
        )
    return {
        "terminal": True,
        "required_arms": contract["required_arms"],
        "longsafety_generations_per_arm": contract["longsafety_generations_per_arm"],
        "ifeval_prompts_per_arm": contract["ifeval_prompts_per_arm"],
        "longsafety_official_judge_status": audit["longsafety_official_judge_status"],
        "comparative_long_context_safety_claim_available": False,
        "source_implementations_verified": True,
        "summary": {"path": str(path), "sha256": sha256(path)},
        "classification": payload.get("classification"),
        "claim_boundary": payload.get("claim_boundary"),
    }


def audit_provenance_inventories(
    *,
    manifest: dict[str, Any],
    manifest_digest: str,
    dataset_inventory_path: Path,
    source_inventory_path: Path,
) -> dict[str, Any]:
    dataset_inventory = json.loads(dataset_inventory_path.read_text())
    source_inventory = json.loads(source_inventory_path.read_text())
    _require(
        dataset_inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and dataset_inventory.get("source", {}).get("dirty") is False
        and dataset_inventory.get("manifest", {}).get("sha256") == manifest_digest,
        "Natural dataset provenance inventory is invalid.",
    )
    _require(
        source_inventory.get("experiment_id") == "p3-natural-source-inventory-v1"
        and source_inventory.get("status") == "verified"
        and source_inventory.get("source", {}).get("dirty") is False
        and source_inventory.get("manifest", {}).get("sha256") == manifest_digest,
        "Natural upstream-source provenance inventory is invalid.",
    )
    expected_datasets = {"SCBench", "LongBench-v2", "LongMemEval", "MRCR"}
    expected_sources = {"SCBench", "LongBench-v2", "LongMemEval"}
    observed_datasets = dataset_inventory.get("benchmarks", {})
    observed_sources = source_inventory.get("benchmarks", {})
    _require(set(observed_datasets) == expected_datasets, "Dataset inventory coverage drifted.")
    _require(set(observed_sources) == expected_sources, "Source inventory coverage drifted.")
    for benchmark in sorted(expected_datasets):
        expected = manifest["benchmarks"][benchmark]["dataset"]
        observed = observed_datasets[benchmark]
        _require(
            observed.get("repo_id") == expected["repo_id"]
            and observed.get("revision") == expected["revision"]
            and observed.get("license") == expected["license"],
            f"{benchmark} dataset license or revision drifted.",
        )
        observed_files = observed.get("files", [])
        _require(
            len(observed_files) == len(expected["files"]),
            f"{benchmark} dataset file coverage drifted.",
        )
        for registered in expected["files"]:
            matches = [
                row
                for row in observed_files
                if Path(row.get("path", "")).as_posix().endswith(registered["path"])
            ]
            _require(
                len(matches) == 1
                and matches[0].get("bytes") == registered["bytes"]
                and matches[0].get("sha256") == registered["sha256"]
                and matches[0].get("rows") == registered["rows"],
                f"{benchmark} dataset file provenance drifted: {registered['path']}",
            )
    for benchmark in sorted(expected_sources):
        expected = manifest["benchmarks"][benchmark]["upstream_code"]
        observed = observed_sources[benchmark]
        _require(
            observed.get("repository") == expected["repository"]
            and observed.get("revision") == expected["revision"]
            and observed.get("license") == expected["license"]
            and observed.get("license_sha256") == expected["license_sha256"],
            f"{benchmark} upstream source license or revision drifted.",
        )
        observed_files = {row.get("path"): row.get("sha256") for row in observed["files"]}
        _require(
            observed_files == expected["files_sha256"],
            f"{benchmark} upstream source file provenance drifted.",
        )
    model = manifest["model"]
    ruler = manifest["benchmarks"]["RULER"]
    ruler_revision = ruler.get("upstream_revision")
    ruler_scorer = ruler.get("scorer", {})
    _require(
        isinstance(ruler.get("upstream_repository"), str)
        and ruler["upstream_repository"].startswith("https://github.com/")
        and isinstance(ruler_revision, str)
        and len(ruler_revision) == 40
        and set(ruler_revision) <= set("0123456789abcdef")
        and ruler.get("license") == "apache-2.0"
        and isinstance(ruler_scorer.get("path"), str),
        "RULER upstream license or revision contract drifted.",
    )
    _sha256_value(ruler_scorer.get("sha256"), "RULER scorer")
    model_revision = model.get("revision")
    _require(
        isinstance(model_revision, str)
        and len(model_revision) == 40
        and set(model_revision) <= set("0123456789abcdef")
        and model.get("license") == "apache-2.0"
        and isinstance(model.get("repo_id"), str),
        "Natural model license or revision contract drifted.",
    )
    _sha256_value(model.get("snapshot_digest_set_sha256"), "natural model snapshot set")
    return {
        "dataset_inventory": {
            "path": str(dataset_inventory_path),
            "sha256": sha256(dataset_inventory_path),
            "benchmarks_verified": len(expected_datasets),
        },
        "source_inventory": {
            "path": str(source_inventory_path),
            "sha256": sha256(source_inventory_path),
            "benchmarks_verified": len(expected_sources),
        },
        "model_license": model["license"],
        "model_revision": model["revision"],
        "ruler_upstream_license": ruler["license"],
        "ruler_upstream_revision": ruler["upstream_revision"],
    }


def summarize(
    manifest_path: Path,
    summary_paths: dict[str, Path],
    safety_summary_path: Path,
    natural_safety_summary_path: Path,
    dataset_inventory_path: Path,
    source_inventory_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    _require(
        manifest.get("experiment_id") == "p3-natural-language-suite-v1",
        "Wrong natural-suite manifest.",
    )
    manifest_digest = sha256(manifest_path)
    required_arms = tuple(manifest["common_protocol"]["p4_gate_baseline_arms"])
    expected = manifest["suite_audit"]["per_arm_minimum_accounted_examples"]
    allowed_failures = set(manifest["common_protocol"]["failure_accounting"])
    model_snapshot_digest = manifest["model"]["snapshot_digest_set_sha256"]
    _sha256_value(model_snapshot_digest, "frozen model snapshot set")
    _require(set(summary_paths) == set(BENCHMARK_IDS), "Natural benchmark summary set drifted.")
    benchmarks: dict[str, Any] = {}
    dependency_sets: list[dict[str, Any]] = []
    for name in manifest["execution_order"]:
        result, dependencies = audit_benchmark(
            name=name,
            path=summary_paths[name],
            expected_examples=expected[name],
            required_arms=required_arms,
            allowed_failures=allowed_failures,
            manifest_digest=manifest_digest,
            model_snapshot_digest=model_snapshot_digest,
            expected_seed=manifest["benchmarks"][name]["generation_seed"],
        )
        benchmarks[name] = result
        dependency_sets.append(dependencies)
    causal_digests = {row["causal_gate"]["sha256"] for row in dependency_sets}
    inventory_digests = {row["dataset_inventory"]["sha256"] for row in dependency_sets}
    fixed_selection_digests = {row["fixed_baseline_selection"]["sha256"] for row in dependency_sets}
    model_digests = {row["model_snapshot_digest_set_sha256"] for row in dependency_sets}
    ruler_dataset_digests = {
        row["benchmark_dataset_digest_set_sha256"]
        for row in dependency_sets
        if row["benchmark_dataset_digest_set_sha256"] is not None
    }
    _require(len(causal_digests) == 1, "Natural benchmarks used different causal gates.")
    _require(len(inventory_digests) == 1, "Natural benchmarks used different datasets.")
    _require(
        len(fixed_selection_digests) == 1,
        "Natural benchmarks used different fixed baseline selections.",
    )
    _require(len(model_digests) == 1, "Natural benchmarks used different model snapshots.")
    _require(len(ruler_dataset_digests) == 1, "RULER dataset manifest set is missing.")
    provenance = audit_provenance_inventories(
        manifest=manifest,
        manifest_digest=manifest_digest,
        dataset_inventory_path=dataset_inventory_path,
        source_inventory_path=source_inventory_path,
    )
    _require(
        provenance["dataset_inventory"]["sha256"] == next(iter(inventory_digests)),
        "Natural benchmark cells do not bind the verified dataset inventory.",
    )
    safety = audit_safety_stress(
        path=safety_summary_path,
        manifest=manifest,
        natural_manifest_digest=manifest_digest,
        model_snapshot_digest=model_snapshot_digest,
    )
    natural_safety = audit_natural_safety(
        path=natural_safety_summary_path,
        manifest=manifest,
        model_snapshot_digest=model_snapshot_digest,
    )
    totals = {
        arm: sum(
            benchmarks[name]["required_arms"][arm]["expected_examples"]
            for name in manifest["execution_order"]
        )
        for arm in required_arms
    }
    minimum = manifest["execution_totals"]["minimum_predictions_per_arm"]
    _require(all(total == minimum for total in totals.values()), "Natural arm total drifted.")
    conservative_quality = {
        arm: sum(
            benchmarks[name]["required_arms"][arm]["mean_score_over_all_expected_failures_zero"]
            * benchmarks[name]["required_arms"][arm]["expected_examples"]
            for name in manifest["execution_order"]
        )
        / minimum
        for arm in required_arms
    }
    return {
        "schema_version": 1,
        "experiment_id": "p3-natural-language-suite-audit-v1",
        "experiment_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
        },
        "audit": {
            "all_required_artifacts_verified": True,
            "all_required_baseline_cells_terminal": True,
            "all_failure_accounting_complete": True,
            "all_source_implementations_verified": True,
            "all_record_revisions_verified": True,
            "all_run_identities_verified": True,
            "all_terminal_measurement_schema_verified": True,
            "all_paired_quality_contrasts_verified": True,
            "dataset_license_revision_inventory_verified": True,
            "upstream_code_license_revision_inventory_verified": True,
            "ruler_license_revision_manifest_verified": True,
            "model_license_revision_manifest_verified": True,
            "safety_stress_terminal": True,
            "natural_safety_terminal": True,
            "benchmarks_terminal": len(benchmarks),
            "minimum_protocol_examples_accounted_per_arm": minimum,
            "accounted_examples_by_required_arm": totals,
            "weighted_conservative_quality_by_required_arm": conservative_quality,
            "causal_gate_sha256": next(iter(causal_digests)),
            "dataset_inventory_sha256": next(iter(inventory_digests)),
            "fixed_baseline_selection_sha256": next(iter(fixed_selection_digests)),
            "model_snapshot_digest_set_sha256": next(iter(model_digests)),
            "ruler_dataset_manifest_digest_set_sha256": next(iter(ruler_dataset_digests)),
            "generation_seed_by_benchmark": {
                name: manifest["benchmarks"][name]["generation_seed"]
                for name in manifest["execution_order"]
            },
        },
        "benchmarks": benchmarks,
        "provenance": provenance,
        "supplemental_safety": safety,
        "supplemental_natural_safety": natural_safety,
        "claim_boundary": (
            "Five-benchmark and synthetic safety-retention evidence on one pinned compatible "
            "Qwen3 model. The primary "
            "conservative quality aggregate scores every operational failure as zero; this "
            "is not official DeepSeek-V4 evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the five-benchmark P3 natural suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-suite.summary.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary_paths = {
        name: Path(path) for name, path in manifest["suite_audit"]["benchmark_summaries"].items()
    }
    safety_summary_path = Path(manifest["suite_audit"]["safety_stress"]["summary"])
    natural_safety_summary_path = Path(
        manifest["suite_audit"]["natural_safety"]["summary"]
    )
    dataset_inventory_path = Path(manifest["suite_audit"]["dataset_inventory"])
    source_inventory_path = Path(manifest["suite_audit"]["source_inventory"])
    payload = summarize(
        args.manifest,
        summary_paths,
        safety_summary_path,
        natural_safety_summary_path,
        dataset_inventory_path,
        source_inventory_path,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Natural-suite summarization requires a clean source tree.")
    payload["source"] = {"commit": commit, "dirty": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
