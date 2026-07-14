from __future__ import annotations

import argparse
import hashlib
import json
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


def _validate_distribution(value: Any, *, observations: int, label: str) -> None:
    _require(isinstance(value, dict), f"Missing {label} distribution.")
    _require(value.get("observations") == observations, f"{label} observations drifted.")
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
            isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 0,
            f"{label} {field} drifted.",
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file(), f"Missing natural benchmark summary: {path}")
    payload = json.loads(path.read_text())
    _require(payload.get("experiment_id") == BENCHMARK_IDS[name], f"Wrong {name} audit id.")
    _require(payload.get("benchmark") == name, f"Wrong benchmark label for {name}.")
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
        failures = row.get("failures_by_type", {})
        _require(
            set(failures).issubset(allowed_failures),
            f"{name}/{arm} contains an unregistered failure type.",
        )
        _require(
            all(isinstance(count, int) and count >= 0 for count in failures.values()),
            f"{name}/{arm} has an invalid failure count.",
        )
        scored = row.get("scored_examples")
        accounted = row.get("accounted_examples")
        _require(isinstance(scored, int) and scored >= 0, f"{name}/{arm} score count invalid.")
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
            isinstance(conservative_mean, (int, float)) and 0.0 <= conservative_mean <= 1.0,
            f"{name}/{arm} conservative quality is missing.",
        )
        failure_rate = row.get("failure_rate")
        _require(
            isinstance(failure_rate, (int, float))
            and not isinstance(failure_rate, bool)
            and failure_rate == sum(failures.values()) / expected_examples,
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
            "mean_score_over_scored": row.get("mean_score_over_scored"),
            "mean_score_over_all_expected_failures_zero": conservative_mean,
            "failure_rate": failure_rate,
            "measurements": measurements,
        }
    contrast = payload.get("paired_quality_contrast", {})
    _require(
        contrast.get("paired_examples") == expected_examples
        and contrast.get("candidate") == required_arms[1]
        and contrast.get("comparator") == required_arms[0]
        and contrast.get("failure_as_zero") is True
        and contrast.get("bootstrap_resamples") == 10_000
        and contrast.get("confidence_level") == 0.95
        and isinstance(contrast.get("paired_bootstrap_95_ci"), list)
        and len(contrast["paired_bootstrap_95_ci"]) == 2
        and sum(contrast.get("failure_pairing", {}).values()) == expected_examples,
        f"{name} paired quality contrast is incomplete.",
    )
    measurement_contrasts = payload.get("paired_measurement_contrasts", {})
    _require(
        set(measurement_contrasts)
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
    _require(
        payload.get("manifest", {}).get("path") == str(safety_manifest_path)
        and payload.get("manifest", {}).get("sha256") == sha256(safety_manifest_path),
        "Safety stress manifest dependency drifted.",
    )
    audit = payload.get("audit", {})
    expected_audit = {
        "required_arms_terminal": True,
        "failure_accounting_complete": True,
        "input_pairing_verified": True,
        "source_implementations_verified": True,
        "protected_prefix_physical_budget_verified": True,
        "examples_accounted_per_arm": contract["examples_per_required_arm"],
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
    _require(
        contrast.get("paired_examples") == contract["examples_per_required_arm"]
        and contrast.get("resident_bytes_equal_for_comparable_pairs") is True,
        "Safety protected-prefix causal contrast is incomplete or not memory matched.",
    )
    for arm in safety_arms:
        row = arms[arm]
        _require(
            row.get("terminal") is True
            and row.get("expected_examples") == contract["examples_per_required_arm"]
            and row.get("scored_examples", 0)
            + sum(row.get("failures_by_type", {}).values())
            == contract["examples_per_required_arm"]
            and len(row.get("slices", [])) == contract["families"] * contract["contexts"],
            f"Safety stress arm is incomplete: {arm}.",
        )
    return {
        "terminal": True,
        "examples_per_required_arm": contract["examples_per_required_arm"],
        "families": contract["families"],
        "contexts": contract["contexts"],
        "required_arms": list(safety_arms),
        "protected_prefix_physical_budget_verified": True,
        "source_implementations_verified": True,
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


def summarize(
    manifest_path: Path,
    summary_paths: dict[str, Path],
    safety_summary_path: Path,
    natural_safety_summary_path: Path,
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
            "all_paired_quality_contrasts_verified": True,
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
        },
        "benchmarks": benchmarks,
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
    payload = summarize(
        args.manifest,
        summary_paths,
        safety_summary_path,
        natural_safety_summary_path,
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
