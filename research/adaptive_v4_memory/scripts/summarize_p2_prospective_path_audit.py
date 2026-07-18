from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import evaluate_p2_causal_factorial_shard as causal
import run_p2_prospective_path_audit as runner

AUDITOR_IMPLEMENTATION_PATHS = (
    *runner.IMPLEMENTATION_PATHS,
    "research/adaptive_v4_memory/scripts/summarize_p2_prospective_path_audit.py",
)


def auditor_implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *AUDITOR_IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_paths = {line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in AUDITOR_IMPLEMENTATION_PATHS
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if not tracked_tree or missing:
        raise RuntimeError(f"Untracked prospective-auditor implementation paths: {missing}")
    return hashlib.sha256(tracked_tree.encode()).hexdigest()


def _expected_phase_paths(phase: str) -> list[tuple[dict[str, Any], Path]]:
    return [
        (
            runner._coordinate_identity(scale, training_seed, budget, family, context),
            runner.coordinate_path(
                phase,
                scale,
                training_seed,
                budget,
                family,
                context,
            ),
        )
        for scale in runner.SCALES
        for training_seed in runner.TRAINING_SEEDS
        for budget in runner.BUDGETS
        for family in runner.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in runner.CONTEXTS
    ]


def _validate_exact_file_set(phase: str, expected_paths: list[Path]) -> None:
    phase_root = runner.OUTPUT_ROOT / phase
    observed = set(phase_root.rglob("*.json")) if phase_root.exists() else set()
    expected = set(expected_paths)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise FileNotFoundError(
            f"{phase} coordinate set is non-terminal: missing={len(missing)}, "
            f"unexpected={len(unexpected)}."
        )


def _validate_receipts(
    phase: str,
    *,
    expected_files: dict[Path, str],
    implementation_sha256: str,
) -> list[dict[str, Any]]:
    receipt_root = runner.OUTPUT_ROOT / "receipts" / phase
    expected_receipt_paths = {
        runner.worker_receipt_path(phase, scale, training_seed)
        for scale in runner.SCALES
        for training_seed in runner.TRAINING_SEEDS
    }
    observed_receipt_paths = set(receipt_root.rglob("*.json")) if receipt_root.exists() else set()
    if observed_receipt_paths != expected_receipt_paths:
        raise FileNotFoundError(
            f"{phase} worker receipt set is non-terminal or contains extras: "
            f"expected={len(expected_receipt_paths)}, observed={len(observed_receipt_paths)}."
        )
    receipts: list[dict[str, Any]] = []
    attributed: dict[Path, str] = {}
    for scale in runner.SCALES:
        for training_seed in runner.TRAINING_SEEDS:
            path = runner.worker_receipt_path(phase, scale, training_seed)
            payload = json.loads(path.read_text())
            artifacts = payload.get("coordinate_artifacts", [])
            if (
                payload.get("schema_version") != 1
                or payload.get("experiment_id") != runner.EXPERIMENT_ID
                or payload.get("artifact_kind") != "terminal_worker_receipt"
                or payload.get("status") != "terminal"
                or payload.get("phase") != phase
                or payload.get("scale") != scale
                or payload.get("training_seed") != training_seed
                or payload.get("expected_coordinate_artifacts")
                != runner.EXPECTED_WORKER_COORDINATES
                or payload.get("observed_coordinate_artifacts")
                != runner.EXPECTED_WORKER_COORDINATES
                or payload.get("expected_path_runs") != runner.EXPECTED_WORKER_RUNS[phase]
                or payload.get("observed_path_runs") != runner.EXPECTED_WORKER_RUNS[phase]
                or len(artifacts) != runner.EXPECTED_WORKER_COORDINATES
                or payload.get("coordinate_artifact_set_sha256") != runner.json_digest(artifacts)
                or payload.get("provenance", {})
                .get("start", {})
                .get("source", {})
                .get("implementation_digest")
                != implementation_sha256
                or payload.get("provenance", {})
                .get("end", {})
                .get("source", {})
                .get("implementation_digest")
                != implementation_sha256
                or payload.get("provenance", {}).get("start", {}).get("source", {}).get("dirty")
                is not False
                or payload.get("provenance", {}).get("end", {}).get("source", {}).get("dirty")
                is not False
                or payload.get("provenance", {}).get("start")
                != payload.get("provenance", {}).get("end")
                or payload.get("payload_sha256") != runner.payload_digest(payload)
            ):
                raise ValueError(f"Invalid terminal worker receipt: {path}")
            for artifact in artifacts:
                artifact_path = Path(str(artifact.get("path", "")))
                digest = str(artifact.get("sha256", ""))
                if artifact_path in attributed:
                    raise ValueError(f"Coordinate appears in multiple receipts: {artifact_path}")
                attributed[artifact_path] = digest
            receipts.append(payload)
    if attributed != expected_files:
        raise ValueError("Terminal receipts do not bind the exact coordinate artifact set.")
    return receipts


def _comparison_failure_counts(comparisons: list[dict[str, Any]]) -> dict[str, int]:
    fields = sorted(
        {field for comparison in comparisons for field in comparison.get("gating_fields", ())}
    )
    return {
        field: sum(comparison.get(field) is not True for comparison in comparisons)
        for field in fields
    }


def _bounded_failure_details(values: list[dict[str, Any]], *, limit: int = 20) -> dict[str, Any]:
    return {
        "count": len(values),
        "all_records_sha256": runner.json_digest(values),
        "sample_limit": limit,
        "samples": values[:limit],
        "all_raw_records_retained_in_coordinate_shards": True,
    }


def _collect_integrity(
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    exact_controls = []
    pin_exposure_groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "coordinates_with_defined_protected_positions": 0,
            "primary_actions_with_nonempty_pins": 0,
            "primary_pinned_position_occurrences": 0,
        }
    )
    protected_coordinates_by_family: Counter[str] = Counter()
    coordinate_pin_exposure_failures: list[dict[str, Any]] = []
    for payload in payloads:
        exact_controls.append(payload["stage_a_exact_controls"])
        family = str(payload["coordinate"]["family"])
        protected_positions_defined = bool(payload["workload"]["protected_end_positions"])
        if protected_positions_defined:
            protected_coordinates_by_family[family] += 1
        for arm in runner.STAGE_A_ARMS:
            arm_payload = payload["arms"][arm]
            arm_observations = arm_payload["observations"]
            if (
                arm_payload.get("path_order")
                != [runner.INTEGRITY_PATH_NAME, runner.INTEGRITY_PATH_NAME]
                or len(arm_observations) != 2
                or [item.get("observation_role") for item in arm_observations]
                != ["primary", "integrity-replay"]
            ):
                raise ValueError("Integrity coordinate contains an invalid replay schedule.")
            observations.extend(arm_observations)
            comparisons.append(arm_payload["sequential_tiered_repeat_comparison"])
            if protected_positions_defined:
                config_sha256 = payload["configurations"][arm]["config_sha256"]
                group = pin_exposure_groups[(family, arm, config_sha256)]
                group["coordinates_with_defined_protected_positions"] += 1
                exposure = arm_observations[0].get("controller", {}).get("pin_exposure", {})
                group["primary_actions_with_nonempty_pins"] += int(
                    exposure.get("actions_with_nonempty_pins", 0)
                )
                group["primary_pinned_position_occurrences"] += int(
                    exposure.get("pinned_position_occurrences", 0)
                )
                if int(exposure.get("actions_with_nonempty_pins", 0)) <= 0:
                    coordinate_pin_exposure_failures.append(
                        {
                            "coordinate": payload["coordinate"],
                            "arm": arm,
                            "config_sha256": config_sha256,
                            "primary_actions_with_nonempty_pins": int(
                                exposure.get("actions_with_nonempty_pins", 0)
                            ),
                        }
                    )
    errors = [item for item in observations if item.get("status") != "success"]
    budget_violations = sum(int(item.get("budget_violations", 0)) for item in observations)
    failed_comparisons = [item for item in comparisons if item.get("passed") is not True]
    exact_control_failures = sum(item.get("passed") is not True for item in exact_controls)
    pin_exposure_failures = [
        {
            "family": family,
            "arm": arm,
            "config_sha256": config_sha256,
            **counts,
        }
        for (family, arm, config_sha256), counts in sorted(pin_exposure_groups.items())
        if counts["primary_actions_with_nonempty_pins"] <= 0
    ]
    pin_exposure_by_family: dict[str, dict[str, Any]] = {}
    for family in runner.PAPER_GRADE_WORKLOAD_FAMILIES:
        group_records: list[dict[str, Any]] = [
            {
                "arm": arm,
                "config_sha256": config_sha256,
                **counts,
                "passed": counts["primary_actions_with_nonempty_pins"] > 0,
            }
            for (group_family, arm, config_sha256), counts in sorted(pin_exposure_groups.items())
            if group_family == family
        ]
        required = protected_coordinates_by_family[family] > 0
        pin_exposure_by_family[family] = {
            "protected_position_coordinates": protected_coordinates_by_family[family],
            "pin_exposure_required": required,
            "distinct_arm_configurations_requiring_exposure": len(group_records),
            "configurations_with_nonzero_pin_actions": sum(
                int(record["primary_actions_with_nonempty_pins"]) > 0 for record in group_records
            ),
            "primary_actions_with_nonempty_pins": sum(
                int(record["primary_actions_with_nonempty_pins"]) for record in group_records
            ),
            "primary_pinned_position_occurrences": sum(
                int(record["primary_pinned_position_occurrences"]) for record in group_records
            ),
            "arm_configuration_exposure": group_records,
            "arm_configuration_exposure_sha256": runner.json_digest(group_records),
            "passed": (not required)
            or all(
                int(record["primary_actions_with_nonempty_pins"]) > 0 for record in group_records
            ),
            "evidence_interpretation": (
                "observed pin exposure"
                if required
                and group_records
                and all(
                    int(record["primary_actions_with_nonempty_pins"]) > 0
                    for record in group_records
                )
                else "not applicable: workload defines no protected positions"
                if not required
                else "no pin-integrity evidence: at least one required configuration had zero exposure"
            ),
        }
    pin_exposure_passed = not pin_exposure_failures and not coordinate_pin_exposure_failures
    gate_passed = (
        len(comparisons) == runner.EXPECTED_ARM_COORDINATES
        and not failed_comparisons
        and not errors
        and budget_violations == 0
        and exact_control_failures == 0
        and pin_exposure_passed
    )
    summary = {
        "expected_repeat_comparisons": runner.EXPECTED_ARM_COORDINATES,
        "observed_repeat_comparisons": len(comparisons),
        "passing_repeat_comparisons": len(comparisons) - len(failed_comparisons),
        "failing_repeat_comparisons": len(failed_comparisons),
        "error_observations": len(errors),
        "budget_violations": budget_violations,
        "exact_control_failures": exact_control_failures,
        "pin_exposure": {
            "requirement": (
                "Every coordinate/arm with workload-defined protected positions must observe "
                "at least one nonempty pin action on the sequential-tiered primary; grouped "
                "family/arm/configuration coverage is reported in addition."
            ),
            "passed": pin_exposure_passed,
            "failing_family_arm_configurations": len(pin_exposure_failures),
            "failing_coordinate_arms": len(coordinate_pin_exposure_failures),
            "by_family": pin_exposure_by_family,
        },
        "comparison_failure_counts": _comparison_failure_counts(comparisons),
        "sequential_tiered_eligibility_passed": gate_passed,
        "cross_corner_results_used_as_blocker": False,
    }
    details = {
        "failed_comparisons": _bounded_failure_details(failed_comparisons),
        "errors": _bounded_failure_details(errors),
        "pin_exposure_failures": _bounded_failure_details(pin_exposure_failures),
        "coordinate_pin_exposure_failures": _bounded_failure_details(
            coordinate_pin_exposure_failures
        ),
    }
    return summary, details


def _collect_cross_corner(
    cross_payloads: list[dict[str, Any]],
    integrity_by_identity: dict[str, tuple[dict[str, Any], Path]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    positional_counts: dict[int, Counter[str]] = defaultdict(Counter)
    integrity_reference_failures = 0
    for payload in cross_payloads:
        identity_digest = runner.json_digest(payload["coordinate"])
        integrity_payload, integrity_path = integrity_by_identity[identity_digest]
        expected_reference = payload.get("integrity_artifact", {})
        if (
            expected_reference.get("path") != str(integrity_path)
            or expected_reference.get("sha256") != causal.sha256(integrity_path)
            or payload["workload"] != integrity_payload["workload"]
            or payload["configurations"] != integrity_payload["configurations"]
        ):
            integrity_reference_failures += 1
        for arm in runner.STAGE_A_ARMS:
            arm_payload = payload["arms"][arm]
            arm_observations = arm_payload["observations"]
            arm_comparisons = arm_payload["comparisons_to_sequential_tiered_primary"]
            if (
                len(arm_observations) != len(runner.CROSS_CORNER_PATH_NAMES)
                or len(arm_comparisons) != len(runner.CROSS_CORNER_PATH_NAMES)
                or set(arm_payload.get("path_order", ())) != set(runner.CROSS_CORNER_PATH_NAMES)
            ):
                raise ValueError("Cross-corner coordinate has an invalid path schedule.")
            for position, path_name in enumerate(arm_payload["path_order"]):
                positional_counts[position][path_name] += 1
            observations.extend(arm_observations)
            comparisons.extend(arm_comparisons)
    errors = [item for item in observations if item.get("status") != "success"]
    tiered_chunk2_errors = [item for item in errors if item.get("path") == "tiered-chunk2"]
    budget_violations = sum(int(item.get("budget_violations", 0)) for item in observations)
    failed_comparisons = [item for item in comparisons if item.get("passed") is not True]
    expected_per_position = runner.EXPECTED_ARM_COORDINATES // len(runner.CROSS_CORNER_PATH_NAMES)
    schedule_balanced = all(
        set(counts) == set(runner.CROSS_CORNER_PATH_NAMES)
        and set(counts.values()) == {expected_per_position}
        for counts in positional_counts.values()
    )
    gate_passed = (
        len(comparisons) == runner.EXPECTED_ARM_COORDINATES * len(runner.CROSS_CORNER_PATH_NAMES)
        and not failed_comparisons
        and not errors
        and budget_violations == 0
        and integrity_reference_failures == 0
        and schedule_balanced
    )
    summary = {
        "expected_comparisons_to_sequential_tiered": runner.EXPECTED_ARM_COORDINATES
        * len(runner.CROSS_CORNER_PATH_NAMES),
        "observed_comparisons_to_sequential_tiered": len(comparisons),
        "passing_comparisons": len(comparisons) - len(failed_comparisons),
        "failing_comparisons": len(failed_comparisons),
        "error_observations": len(errors),
        "tiered_chunk2_error_observations": len(tiered_chunk2_errors),
        "budget_violations": budget_violations,
        "integrity_reference_failures": integrity_reference_failures,
        "comparison_failure_counts": _comparison_failure_counts(comparisons),
        "corner_order_position_counts": {
            str(position): dict(sorted(counts.items()))
            for position, counts in sorted(positional_counts.items())
        },
        "corner_order_globally_balanced": schedule_balanced,
        "cross_corner_interchangeability_passed": gate_passed,
        "targeted_stage_a_blocked_by_cross_corner": False,
    }
    details = {
        "failed_comparisons": _bounded_failure_details(failed_comparisons),
        "errors": _bounded_failure_details(errors),
    }
    return summary, details


def _validate_content_grid(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed_coordinate: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    digest_counts: Counter[str] = Counter()
    generation_seeds: dict[tuple[Any, ...], int] = {}
    for payload in payloads:
        identity = payload["coordinate"]
        key = (
            identity["scale"],
            identity["training_seed"],
            identity["family"],
            identity["context"],
        )
        digest = payload["workload"]["content_sha256"]
        by_seed_coordinate[key].add(digest)
        digest_counts[digest] += 1
        generation_seeds[key] = int(payload["generation_seed"])
    expected_unique = (
        len(runner.SCALES)
        * len(runner.TRAINING_SEEDS)
        * len(runner.PAPER_GRADE_WORKLOAD_FAMILIES)
        * len(runner.CONTEXTS)
    )
    passed = (
        len(by_seed_coordinate) == expected_unique
        and all(len(values) == 1 for values in by_seed_coordinate.values())
        and len(digest_counts) == expected_unique
        and set(digest_counts.values()) == {len(runner.BUDGETS)}
        and len(set(generation_seeds.values())) == expected_unique
        and all(payload["localization_collision_check"]["passed"] is True for payload in payloads)
    )
    return {
        "expected_unique_seed_workloads": expected_unique,
        "observed_unique_seed_workloads": len(by_seed_coordinate),
        "observed_unique_content_digests": len(digest_counts),
        "each_content_digest_repeated_once_per_budget": set(digest_counts.values())
        == {len(runner.BUDGETS)},
        "generation_seeds_unique": len(set(generation_seeds.values())) == expected_unique,
        "localization_collision_count": sum(
            payload["localization_collision_check"]["collision"] is True for payload in payloads
        ),
        "passed": passed,
    }


def summarize(phase: str) -> dict[str, Any]:
    if phase not in {"integrity", "full"}:
        raise ValueError(f"Unregistered summary phase: {phase}")
    manifest = runner._validate_frozen_contract()
    source = runner.source_state()
    if source["dirty"]:
        raise RuntimeError("Prospective audit summarization requires a clean source tree.")
    implementation_sha256 = runner.implementation_digest()
    manifest_sha256 = causal.sha256(runner.FROZEN_MANIFEST_PATH)
    if phase == "full":
        runner._validate_integrity_summary(
            runner.integrity_summary_path(),
            implementation_sha256=implementation_sha256,
            manifest_sha256=manifest_sha256,
        )

    phases = ["integrity"] if phase == "integrity" else ["integrity", "cross-corner"]
    phase_payloads: dict[str, list[dict[str, Any]]] = {}
    phase_files: dict[str, dict[Path, str]] = {}
    phase_receipts: dict[str, list[dict[str, Any]]] = {}
    for execution_phase in phases:
        expected = _expected_phase_paths(execution_phase)
        paths = [path for _identity, path in expected]
        _validate_exact_file_set(execution_phase, paths)
        payloads: list[dict[str, Any]] = []
        files: dict[Path, str] = {}
        for identity, path in expected:
            payload = json.loads(path.read_text())
            runner._validate_coordinate_payload(
                payload,
                phase=execution_phase,
                expected_identity=identity,
                expected_implementation_digest=implementation_sha256,
                expected_manifest_sha256=manifest_sha256,
            )
            payloads.append(payload)
            files[path] = causal.sha256(path)
        phase_payloads[execution_phase] = payloads
        phase_files[execution_phase] = files
        phase_receipts[execution_phase] = _validate_receipts(
            execution_phase,
            expected_files=files,
            implementation_sha256=implementation_sha256,
        )

    integrity_summary, integrity_details = _collect_integrity(phase_payloads["integrity"])
    content_grid = _validate_content_grid(phase_payloads["integrity"])
    if not content_grid["passed"]:
        raise ValueError("Prospective workload seed/content/collision grid failed audit.")

    cross_summary = None
    cross_details = None
    if phase == "full":
        integrity_by_identity = {
            runner.json_digest(payload["coordinate"]): (payload, path)
            for payload, (_identity, path) in zip(
                phase_payloads["integrity"],
                _expected_phase_paths("integrity"),
                strict=True,
            )
        }
        cross_summary, cross_details = _collect_cross_corner(
            phase_payloads["cross-corner"], integrity_by_identity
        )

    observed_runs = sum(
        payload["observed_path_runs"]
        for payloads in phase_payloads.values()
        for payload in payloads
    )
    expected_runs = (
        runner.EXPECTED_PHASE_RUNS["integrity"]
        if phase == "integrity"
        else runner.EXPECTED_TOTAL_RUNS
    )
    if observed_runs != expected_runs:
        raise ValueError("Terminal path-run count drifted from the frozen contract.")
    artifact_sets = {
        execution_phase: {
            "coordinate_artifacts": len(phase_files[execution_phase]),
            "coordinate_artifact_set_sha256": runner.json_digest(
                [
                    {"path": str(path), "sha256": digest}
                    for path, digest in sorted(phase_files[execution_phase].items())
                ]
            ),
            "worker_receipts": len(phase_receipts[execution_phase]),
            "worker_receipt_set_sha256": runner.json_digest(
                [receipt["payload_sha256"] for receipt in phase_receipts[execution_phase]]
            ),
        }
        for execution_phase in phases
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": runner.SUMMARY_EXPERIMENT_ID,
        "status": "terminal",
        "phase": phase,
        "execution_schedule": {
            "phase_1": {
                "name": "integrity",
                "path_runs": runner.EXPECTED_PHASE_RUNS["integrity"],
                "purpose": "Stage-A sequential-tiered eligibility is audited first.",
            },
            "phase_2": {
                "name": "cross-corner",
                "path_runs": runner.EXPECTED_PHASE_RUNS["cross-corner"],
                "purpose": (
                    "Complete the remaining three corners without making their result a "
                    "Stage-A blocker."
                ),
                "included_in_this_summary": phase == "full",
            },
            "frozen_total_path_runs_preserved": runner.EXPECTED_TOTAL_RUNS,
        },
        "expected_coordinate_artifacts": runner.EXPECTED_COORDINATES * len(phases),
        "observed_coordinate_artifacts": sum(len(values) for values in phase_payloads.values()),
        "expected_path_runs": expected_runs,
        "observed_path_runs": observed_runs,
        "implementation_digest": implementation_sha256,
        "auditor_implementation_digest": auditor_implementation_digest(),
        "source": source,
        "frozen_manifest": {
            "path": str(runner.FROZEN_MANIFEST_PATH),
            "sha256": manifest_sha256,
            "experiment_id": manifest["experiment_id"],
            "frozen_at": manifest["frozen_at"],
        },
        "artifact_sets": artifact_sets,
        "workload_grid": content_grid,
        "sequential_tiered_eligibility": integrity_summary,
        "cross_corner_interchangeability": cross_summary,
        "targeted_stage_a_eligible": integrity_summary["sequential_tiered_eligibility_passed"],
        "cross_corner_results_used_as_stage_a_blocker": False,
        "failure_details": {
            "integrity": integrity_details,
            "cross_corner": cross_details,
        },
        "nonclaims": [
            "no quality or accuracy comparison",
            "no causal effect estimate",
            "no production latency, memory-capacity, or throughput claim",
            "no replacement of the paused 16-arm causal factorial",
        ],
        "claim_boundary": (
            "The integrity summary decides only same-path sequential-tiered execution "
            "eligibility. The full summary additionally reports four-corner computational "
            "interchangeability, but cross-corner failure does not block Stage A."
        ),
    }
    summary["payload_sha256"] = runner.payload_digest(summary)
    return summary


def _write_summary_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if not path.resolve().is_relative_to((runner.OUTPUT_ROOT / "audits").resolve()):
        raise ValueError("Prospective summary must remain under the staged audit root.")
    if path.exists():
        raise FileExistsError(f"Prospective summary already exists: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Prospective summary already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the frozen Stage-A prospective path-panel shards."
    )
    parser.add_argument("--phase", choices=("integrity", "full"), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = summarize(args.phase)
    path = (
        runner.integrity_summary_path()
        if args.phase == "integrity"
        else runner.OUTPUT_ROOT / "audits/full.summary.json"
    )
    _write_summary_exclusive(path, payload)
    print(
        json.dumps(
            {
                "event": "prospective_path_audit_terminal",
                "phase": args.phase,
                "output": str(path),
                "expected_path_runs": payload["expected_path_runs"],
                "targeted_stage_a_eligible": payload["targeted_stage_a_eligible"],
                "cross_corner_interchangeability": payload["cross_corner_interchangeability"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
