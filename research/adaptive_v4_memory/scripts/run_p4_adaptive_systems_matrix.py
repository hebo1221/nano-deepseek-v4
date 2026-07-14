from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as causal
import run_p4_systems_matrix as reference
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from freeze_p2_causal_factorial_arms import BuiltCausalArm, build_arm_configs

from nano_deepseek_v4 import DeepSeekV4ForCausalLM, SameTokenControllerConfig

SCALES = reference.SCALES
BUDGETS = causal.BUDGET_LABELS
CONTEXTS = reference.CONTEXTS
GENERATIONS = reference.GENERATIONS
LOAD_PROFILES = reference.LOAD_PROFILES
POLICIES = ("fixed+pins", "calibrated+pins")
PROTECTED_END_POSITIONS = (3,)
WARMUPS = reference.WARMUPS
MEASURED_REPETITIONS = reference.MEASURED_REPETITIONS
INPUT_SEED_BASE = 9_171_400
CELL_TIMEOUT_SECONDS = reference.CELL_TIMEOUT_SECONDS
EXPECTED_CELLS = len(SCALES) * len(BUDGETS) * len(CONTEXTS) * len(GENERATIONS) * len(LOAD_PROFILES)
REQUIRED_CAUSAL_TRUE_AUDITS = (
    "exact_seed_randomization_verified",
    "physical_controller_budget_verified",
    "exact_statistical_cell_coverage_verified",
    "family_holm_bonferroni_verified",
    "contrast_holm_bonferroni_verified",
    "primary_four_cell_bonferroni_verified",
    "required_scale_seed_completion_verified",
)
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4",
    "research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json",
    "research/adaptive_v4_memory/manifests/p4-adaptive-systems-matrix-v1.json",
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/evaluate_p1_heldout_policy_pilot.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_causal_factorial_shard.py",
    "research/adaptive_v4_memory/scripts/freeze_p2_causal_factorial_arms.py",
    "research/adaptive_v4_memory/scripts/run_p4_systems_matrix.py",
    "research/adaptive_v4_memory/scripts/run_p4_adaptive_systems_matrix.py",
    "research/adaptive_v4_memory/scripts/summarize_p4_adaptive_systems_matrix.py",
)

Cell = tuple[str, str, int, int, str, int, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def implementation_digest() -> str:
    tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = {line.split("\t", 1)[1] for line in tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in IMPLEMENTATION_PATHS
        if path not in paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked adaptive P4 implementation paths: {missing}")
    return hashlib.sha256(tree.encode()).hexdigest()


def frozen_cells() -> tuple[Cell, ...]:
    return tuple(
        (scale, budget, context, generation, name, batch, active_requests)
        for scale, budget, context, generation, (name, batch, active_requests) in product(
            SCALES, BUDGETS, CONTEXTS, GENERATIONS, LOAD_PROFILES
        )
    )


def cell_path(root: Path, cell: Cell) -> Path:
    scale, budget, context, generation, profile, _batch, _active_requests = cell
    return (
        root
        / scale
        / f"budget-{budget}"
        / f"context-{context}"
        / f"generation-{generation}"
        / profile
    )


def require_nine_seed_causal_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    gate = payload.get("primary_causal_gate", {})
    cells_payload = gate.get("cells", [])
    cells = cells_payload if isinstance(cells_payload, list) else []
    raw_matrix = payload.get("raw_matrix", {})
    raw_matrix_path = Path(raw_matrix.get("path", ""))
    pooling = payload.get("pooling_audit", {})
    inference = payload.get("confirmatory_inference", {})
    identities = {
        (cell.get("scale"), cell.get("budget")) for cell in cells if isinstance(cell, dict)
    }
    cell_contracts_are_consistent = all(
        isinstance(cell, dict)
        and all(
            type(cell.get(name)) is bool
            for name in (
                "pooled_effect_positive",
                "four_cell_corrected_lower_bound_positive",
                "all_seed_effects_positive",
                "all_seed_memory_cells_within_one_percent",
            )
        )
        and cell.get("passed")
        is all(
            cell[name]
            for name in (
                "pooled_effect_positive",
                "four_cell_corrected_lower_bound_positive",
                "all_seed_effects_positive",
                "all_seed_memory_cells_within_one_percent",
            )
        )
        for cell in cells
    )
    _require(
        payload.get("experiment_id") == "p2-nine-seed-causal-ablation-audit-v1"
        and payload.get("source", {}).get("dirty") is False
        and raw_matrix_path.is_file()
        and raw_matrix.get("sha256") == reference.sha256(raw_matrix_path)
        and audit.get("unique_shards") == 16_200
        and audit.get("independent_seed_clusters_per_cell") == 9
        and audit.get("all_raw_shards_verified") is True
        and audit.get("all_dependency_digests_verified") is True
        and audit.get("all_record_digests_verified") is True
        and audit.get("no_budget_violations") is True
        and audit.get("all_physical_predictions_identical") is True
        and audit.get("exact_config_reuse_verified") is True
        and audit.get("outcome_dependent_early_stopping") is False
        and audit.get("seed_p_values_used_as_success_gate") is False
        and all(audit.get(name) is True for name in REQUIRED_CAUSAL_TRUE_AUDITS)
        and pooling.get("identical_frozen_contracts") is True
        and pooling.get("identical_base_implementation") is True
        and pooling.get("disjoint_training_seeds") is True
        and pooling.get("cohorts_independently_audited") is True
        and pooling.get("outcome_dependent_early_stopping") is False
        and inference.get("independent_training_seeds_per_scale") == 9
        and inference.get("exact_sign_assignments") == 512
        and inference.get("outcome_dependent_early_stopping") is False
        and gate.get("candidate") == "calibrated+pins"
        and gate.get("comparator") == "fixed+pins"
        and tuple(gate.get("scales", ())) == SCALES
        and tuple(gate.get("budgets", ())) == BUDGETS
        and gate.get("seeds_per_scale") == 9
        and gate.get("required_cells") == 4
        and isinstance(cells_payload, list)
        and len(cells) == 4
        and identities == set(product(SCALES, BUDGETS))
        and cell_contracts_are_consistent
        and gate.get("passed") is all(cell["passed"] for cell in cells),
        "Adaptive P4 requires the complete nine-seed causal audit.",
    )
    return payload


def schedule_index(cell: Cell, repetition: int) -> int:
    if not 0 <= repetition < WARMUPS + MEASURED_REPETITIONS:
        raise ValueError("Adaptive P4 repetition is outside the frozen schedule.")
    return frozen_cells().index(cell) * (WARMUPS + MEASURED_REPETITIONS) + repetition


def _config_sha256(config: SameTokenControllerConfig) -> str:
    return causal.config_digest(config)


def policy_config(
    arms: dict[str, BuiltCausalArm], *, policy: str, cell: Cell, repetition: int
) -> tuple[SameTokenControllerConfig, str]:
    arm = arms[policy]
    index = schedule_index(cell, repetition)
    config = arm.config_for_batch(index)
    variant = "single"
    if len(arm.configs) == 2:
        variant = "low" if config is arm.configs[0] else "high"
    return config, variant


def load_policy_arms(
    *,
    scale: str,
    budget: str,
    checkpoint: Path,
    calibration_path: Path,
    memory_match_path: Path,
) -> tuple[dict[str, BuiltCausalArm], dict[str, Any]]:
    calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
    memory_match = causal._memory_match(
        memory_match_path,
        scale=scale,
        training_seed=6071401,
        calibration_path=calibration_path,
    )
    arms, metadata = build_arm_configs(calibration, budget, fixed_match=memory_match)
    return {policy: arms[policy] for policy in POLICIES}, metadata


def _valid_adaptive_controller(run: dict[str, Any], *, config: SameTokenControllerConfig) -> bool:
    controller = run.get("adaptive_controller")
    configured = dict(config.layer_budgets)
    if not isinstance(controller, dict):
        return False
    try:
        recorded_configured = {
            int(layer): value
            for layer, value in controller["configured_blocks_per_sequence_by_layer"].items()
        }
        physical = {
            int(layer): value
            for layer, value in controller["configured_physical_hot_blocks_by_layer"].items()
        }
        observed = {
            int(layer): value for layer, value in controller["observed_hot_blocks_by_layer"].items()
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    expected_physical = {
        layer: blocks * int(run["batch"]) * int(run["requests"])
        for layer, blocks in configured.items()
    }
    counters = (
        "selected_queries",
        "finalized_control_points",
        "fallback_control_points",
        "telemetry_time_ns",
        "controller_time_ns",
    )
    return (
        controller.get("enabled") is True
        and controller.get("protected_end_positions") == list(PROTECTED_END_POSITIONS)
        and controller.get("config_sha256") == _config_sha256(config)
        and recorded_configured == configured
        and physical == expected_physical
        and set(observed) == set(configured)
        and all(
            type(value) is int and 0 <= value <= expected_physical[layer]
            for layer, value in observed.items()
        )
        and all(type(controller.get(name)) is int and controller[name] >= 0 for name in counters)
        and controller["controller_time_ns"] == run.get("controller_time_ns")
    )


def _valid_repetition(
    row: Any,
    *,
    cell: Cell,
    repetition: int,
    arms: dict[str, BuiltCausalArm],
) -> bool:
    if not isinstance(row, dict):
        return False
    policies = row.get("policies", {})
    failures = row.get("policy_failures", {})
    order = POLICIES if repetition % 2 == 0 else tuple(reversed(POLICIES))
    reference_cell = (cell[0], cell[2], cell[3], cell[4], cell[5], cell[6])
    if not (
        row.get("repetition") == repetition
        and row.get("input_seed") == INPUT_SEED_BASE + WARMUPS + repetition
        and tuple(row.get("execution_order", ())) == order
        and isinstance(policies, dict)
        and set(policies).issubset(POLICIES)
        and isinstance(failures, dict)
        and set(failures).issubset(POLICIES)
        and not (set(policies) & set(failures))
        and all(
            reference._valid_failure(failure, phases={"measured"})
            and failure.get("repetition") == repetition + WARMUPS
            for failure in failures.values()
        )
    ):
        return False
    input_digest = row.get("input_digest")
    if not isinstance(input_digest, str):
        return False
    for policy, run in policies.items():
        config, variant = policy_config(
            arms,
            policy=policy,
            cell=cell,
            repetition=repetition + WARMUPS,
        )
        if (
            not reference._valid_policy_run(
                run,
                policy=policy,
                input_digest=input_digest,
                cell=reference_cell,
            )
            or not _valid_adaptive_controller(run, config=config)
            or row.get("policy_configs", {}).get(policy)
            != {"sha256": _config_sha256(config), "variant": variant}
        ):
            return False
    if set(policies) == set(POLICIES):
        equal = (
            policies[POLICIES[0]]["prediction_digest"] == policies[POLICIES[1]]["prediction_digest"]
        )
        return row.get("prediction_digests_equal") is equal
    return row.get("prediction_digests_equal") is None


def _artifact_valid(
    path: Path,
    *,
    cell: Cell,
    implementation: str,
    dependencies: dict[str, str],
    arms: dict[str, BuiltCausalArm],
    p2_gate_passed: bool,
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    identity = tuple(
        payload.get("cell", {}).get(name)
        for name in (
            "scale",
            "budget",
            "context",
            "generation",
            "profile",
            "batch",
            "active_requests",
        )
    )
    repetitions = payload.get("repetitions", [])
    warmup_attempted = payload.get("warmup_repetitions_attempted")
    warmup_paired = payload.get("warmup_paired_repetitions_completed")
    warmup_runs = payload.get("warmup_policy_runs_completed")
    warmup_failures = payload.get("warmup_failures")

    def dependency_valid(name: str, digest: str) -> bool:
        metadata = payload.get(name, {})
        dependency_path = Path(metadata.get("path", ""))
        return (
            dependency_path.is_file()
            and metadata.get("sha256") == digest
            and reference.sha256(dependency_path) == digest
        )

    if not (
        payload.get("schema_version") == 1
        and payload.get("experiment_id") == "p4-adaptive-systems-cell-v1"
        and identity == cell
        and payload.get("status") in reference.TERMINAL_STATUSES
        and payload.get("source", {}).get("dirty") is False
        and payload.get("source", {}).get("implementation_digest") == implementation
        and payload.get("input_seed_base") == INPUT_SEED_BASE
        and payload.get("p2_causal_gate_passed") is p2_gate_passed
        and payload.get("warmups") == WARMUPS
        and payload.get("warmup_accounting_available") is True
        and type(warmup_attempted) is int
        and 0 <= warmup_attempted <= WARMUPS
        and type(warmup_paired) is int
        and 0 <= warmup_paired <= warmup_attempted
        and isinstance(warmup_runs, dict)
        and set(warmup_runs) == set(POLICIES)
        and all(
            type(warmup_runs[policy]) is int and 0 <= warmup_runs[policy] <= warmup_attempted
            for policy in POLICIES
        )
        and warmup_paired == min(warmup_runs.values())
        and isinstance(warmup_failures, list)
        and all(reference._valid_failure(failure, phases={"warmup"}) for failure in warmup_failures)
        and payload.get("measured_repetitions") == MEASURED_REPETITIONS
        and type(payload.get("cell_timeout_seconds")) in (int, float)
        and 0.0 < payload["cell_timeout_seconds"] <= CELL_TIMEOUT_SECONDS
        and isinstance(repetitions, list)
        and len(repetitions) <= MEASURED_REPETITIONS
        and all(dependency_valid(name, digest) for name, digest in dependencies.items())
        and all(
            _valid_repetition(row, cell=cell, repetition=index, arms=arms)
            for index, row in enumerate(repetitions)
        )
    ):
        return False
    policy_status = payload.get("policy_status", {})
    if not isinstance(policy_status, dict) or set(policy_status) != set(POLICIES):
        return False
    counts = {
        policy: sum(policy in row.get("policies", {}) for row in repetitions) for policy in POLICIES
    }
    complete = {policy for policy, count in counts.items() if count == MEASURED_REPETITIONS}
    expected_status = "complete" if len(complete) == 2 else "partial" if complete else "failed"
    return payload["status"] == expected_status and all(
        policy_status[policy].get("measured_repetitions") == counts[policy]
        and policy_status[policy].get("status")
        == ("complete" if counts[policy] == MEASURED_REPETITIONS else "failed")
        and (
            policy_status[policy].get("failure") is None
            if counts[policy] == MEASURED_REPETITIONS
            else reference._valid_failure(
                policy_status[policy].get("failure"), phases={"warmup", "measured"}
            )
        )
        for policy in POLICIES
    )


def _dependency(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": reference.sha256(path)}


def _run_row(cell: Cell, artifact: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **dict(
            zip(
                (
                    "scale",
                    "budget",
                    "context",
                    "generation",
                    "profile",
                    "batch",
                    "active_requests",
                ),
                cell,
                strict=True,
            )
        ),
        "status": payload["status"],
        "artifact": {"path": str(artifact), "sha256": reference.sha256(artifact)},
    }


def _failure(error: Exception, *, phase: str, repetition: int) -> dict[str, Any]:
    return {
        "failure_type": (
            "oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            else "timeout"
            if isinstance(error, TimeoutError)
            else "error"
        ),
        "error_type": type(error).__name__,
        "error": str(error),
        "phase": phase,
        "repetition": repetition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen adaptive P4 systems matrix.")
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--budget", action="append", choices=BUDGETS)
    parser.add_argument("--context", type=int, action="append", choices=CONTEXTS)
    parser.add_argument("--generation", type=int, action="append", choices=GENERATIONS)
    parser.add_argument(
        "--profile", action="append", choices=tuple(row[0] for row in LOAD_PROFILES)
    )
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--cell-timeout-seconds", type=float, default=CELL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix"),
    )
    parser.add_argument(
        "--memory-match-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_hot_memory"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p4-adaptive-systems-matrix-v1.json"),
    )
    parser.add_argument(
        "--p2-audit",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"),
    )
    parser.add_argument(
        "--p3-audit",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-suite.summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-systems"),
    )
    parser.add_argument(
        "--matrix-progress",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-systems-matrix.json"),
    )
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        raise ValueError("max-new-cells must be positive.")
    reference.validate_cell_timeout(args.cell_timeout_seconds)
    if reference._dirty():
        raise RuntimeError("Adaptive P4 execution requires a clean source tree.")
    p2 = require_nine_seed_causal_audit(args.p2_audit)
    reference.require_p3_audit(args.p3_audit)
    manifest = json.loads(args.manifest.read_text())
    _require(
        manifest.get("experiment_id") == "p4-adaptive-systems-matrix-v1"
        and manifest.get("primary_paired_cells") == EXPECTED_CELLS
        and manifest.get("input_seed_base") == INPUT_SEED_BASE
        and manifest.get("paired_policies") == list(POLICIES),
        "The frozen adaptive P4 manifest is required.",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Adaptive P4 execution requires CUDA.")
    lock = acquire_gpu_lock("p4-adaptive-systems-matrix")
    implementation = implementation_digest()
    base_dependencies = {
        "manifest": reference.sha256(args.manifest),
        "p2_audit": reference.sha256(args.p2_audit),
        "p3_audit": reference.sha256(args.p3_audit),
    }
    selected = [
        cell
        for cell in frozen_cells()
        if (not args.scale or cell[0] in args.scale)
        and (not args.budget or cell[1] in args.budget)
        and (not args.context or cell[2] in args.context)
        and (not args.generation or cell[3] in args.generation)
        and (not args.profile or cell[4] in args.profile)
    ]
    models: dict[str, DeepSeekV4ForCausalLM] = {}
    bundles: dict[tuple[str, str], tuple[dict[str, BuiltCausalArm], dict[str, Any]]] = {}
    dependency_metadata: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    device = torch.device("cuda")
    for scale in SCALES:
        checkpoint = args.training_root / scale / "seed-6071401" / f"{scale}-step-1000.pt"
        calibration = args.calibration_root / scale / "seed-6071401" / "p1-layer-quotas.json"
        memory_match = (
            args.memory_match_root
            / scale
            / "seed-6071401"
            / "p2-causal-hot-memory-match.summary.json"
        )
        for budget in BUDGETS:
            bundles[(scale, budget)] = load_policy_arms(
                scale=scale,
                budget=budget,
                checkpoint=checkpoint,
                calibration_path=calibration,
                memory_match_path=memory_match,
            )
            dependency_metadata[(scale, budget)] = {
                "checkpoint": _dependency(checkpoint),
                "calibration": _dependency(calibration),
                "memory_match": _dependency(memory_match),
            }
    runs: dict[Cell, dict[str, Any]] = {}
    p2_gate_passed = bool(p2["primary_causal_gate"]["passed"])
    for existing_cell in frozen_cells():
        existing_scale, existing_budget = existing_cell[:2]
        existing_arms, _metadata = bundles[(existing_scale, existing_budget)]
        existing_dependencies = {
            **base_dependencies,
            **{
                name: value["sha256"]
                for name, value in dependency_metadata[(existing_scale, existing_budget)].items()
            },
        }
        existing_artifact = cell_path(args.output_root, existing_cell) / "cell.json"
        if _artifact_valid(
            existing_artifact,
            cell=existing_cell,
            implementation=implementation,
            dependencies=existing_dependencies,
            arms=existing_arms,
            p2_gate_passed=p2_gate_passed,
        ):
            runs[existing_cell] = _run_row(
                existing_cell,
                existing_artifact,
                json.loads(existing_artifact.read_text()),
            )
    new_cells = 0
    for cell in selected:
        scale, budget, context, generation, profile, batch, active_requests = cell
        arms, arm_metadata = bundles[(scale, budget)]
        per_cell_dependencies = {
            **base_dependencies,
            **{
                name: value["sha256"]
                for name, value in dependency_metadata[(scale, budget)].items()
            },
        }
        artifact = cell_path(args.output_root, cell) / "cell.json"
        if _artifact_valid(
            artifact,
            cell=cell,
            implementation=implementation,
            dependencies=per_cell_dependencies,
            arms=arms,
            p2_gate_passed=p2_gate_passed,
        ):
            payload = json.loads(artifact.read_text())
        else:
            if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                break
            if scale not in models:
                checkpoint = Path(dependency_metadata[(scale, budget)]["checkpoint"]["path"])
                models[scale] = reference._load_model(checkpoint, device)
            model = models[scale]
            started = time.monotonic()
            repetitions: list[dict[str, Any]] = []
            policy_failures: dict[str, dict[str, Any]] = {}
            warmup_attempted = 0
            warmup_paired = 0
            warmup_runs = {policy: 0 for policy in POLICIES}
            warmup_failures: list[dict[str, Any]] = []
            previous_handler = reference.arm_cell_timeout(args.cell_timeout_seconds)
            timeout_cancelled = False
            try:
                for repetition in range(WARMUPS + MEASURED_REPETITIONS):
                    prompts, decode, input_digest = reference.generate_inputs(
                        model,
                        context=context,
                        generation=generation,
                        batch=batch,
                        active_requests=active_requests,
                        seed=INPUT_SEED_BASE + repetition,
                    )
                    if repetition < WARMUPS:
                        warmup_attempted += 1
                    order_index = reference.phase_repetition_index(repetition)
                    order = POLICIES if order_index % 2 == 0 else tuple(reversed(POLICIES))
                    policy_runs: dict[str, dict[str, Any]] = {}
                    policy_configs: dict[str, dict[str, str]] = {}
                    failures_this_repetition: dict[str, dict[str, Any]] = {}
                    for policy in order:
                        if policy in policy_failures:
                            continue
                        config, variant = policy_config(
                            arms, policy=policy, cell=cell, repetition=repetition
                        )
                        try:
                            run = reference.run_policy(
                                model,
                                policy=policy,
                                prompts=prompts,
                                decode_tokens=decode,
                                input_digest=input_digest,
                                device=device,
                                controller_config=config,
                                protected_end_positions=PROTECTED_END_POSITIONS,
                            )
                            run["adaptive_controller"]["config_sha256"] = _config_sha256(config)
                            policy_runs[policy] = run
                            policy_configs[policy] = {
                                "sha256": _config_sha256(config),
                                "variant": variant,
                            }
                            if repetition < WARMUPS:
                                warmup_runs[policy] += 1
                        except Exception as error:
                            if isinstance(error, TimeoutError):
                                raise
                            failure = _failure(
                                error,
                                phase="warmup" if repetition < WARMUPS else "measured",
                                repetition=repetition,
                            )
                            policy_failures[policy] = failure
                            failures_this_repetition[policy] = failure
                            if repetition < WARMUPS:
                                warmup_failures.append(failure)
                            reference._cleanup()
                    if repetition < WARMUPS and set(policy_runs) == set(POLICIES):
                        warmup_paired += 1
                    if repetition >= WARMUPS:
                        paired = set(policy_runs) == set(POLICIES)
                        repetitions.append(
                            {
                                "repetition": repetition - WARMUPS,
                                "input_seed": INPUT_SEED_BASE + repetition,
                                "execution_order": list(order),
                                "input_digest": input_digest,
                                "prediction_digests_equal": (
                                    policy_runs[POLICIES[0]]["prediction_digest"]
                                    == policy_runs[POLICIES[1]]["prediction_digest"]
                                    if paired
                                    else None
                                ),
                                "policy_configs": policy_configs,
                                "policies": policy_runs,
                                "policy_failures": failures_this_repetition,
                            }
                        )
                    del prompts, decode, policy_runs
                    if len(policy_failures) == len(POLICIES):
                        break
                counts = {
                    policy: sum(policy in row["policies"] for row in repetitions)
                    for policy in POLICIES
                }
                policy_status = {
                    policy: {
                        "status": (
                            "complete" if counts[policy] == MEASURED_REPETITIONS else "failed"
                        ),
                        "measured_repetitions": counts[policy],
                        "failure": policy_failures.get(policy),
                    }
                    for policy in POLICIES
                }
                complete = sum(row["status"] == "complete" for row in policy_status.values())
                status = "complete" if complete == 2 else "partial" if complete else "failed"
                payload = {
                    "schema_version": 1,
                    "experiment_id": "p4-adaptive-systems-cell-v1",
                    "status": status,
                    "cell": dict(
                        zip(
                            (
                                "scale",
                                "budget",
                                "context",
                                "generation",
                                "profile",
                                "batch",
                                "active_requests",
                            ),
                            cell,
                            strict=True,
                        )
                    ),
                    "warmups": WARMUPS,
                    "input_seed_base": INPUT_SEED_BASE,
                    "cell_timeout_seconds": args.cell_timeout_seconds,
                    "warmup_accounting_available": True,
                    "warmup_repetitions_attempted": warmup_attempted,
                    "warmup_paired_repetitions_completed": warmup_paired,
                    "warmup_policy_runs_completed": warmup_runs,
                    "warmup_failures": warmup_failures,
                    "measured_repetitions": MEASURED_REPETITIONS,
                    "repetitions": repetitions,
                    "policy_status": policy_status,
                    "arm_metadata": arm_metadata,
                    "p2_causal_gate_passed": p2_gate_passed,
                    "elapsed_seconds": time.monotonic() - started,
                }
            except Exception as error:
                reference.cancel_cell_timeout(previous_handler)
                timeout_cancelled = True
                phase = "warmup" if any(warmup_runs[p] < WARMUPS for p in POLICIES) else "measured"
                terminal = {
                    policy: policy_failures.get(policy)
                    or _failure(error, phase=phase, repetition=WARMUPS + len(repetitions))
                    for policy in POLICIES
                }
                payload = {
                    "schema_version": 1,
                    "experiment_id": "p4-adaptive-systems-cell-v1",
                    "status": "failed",
                    "failure_type": next(iter(terminal.values()))["failure_type"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "cell": dict(
                        zip(
                            (
                                "scale",
                                "budget",
                                "context",
                                "generation",
                                "profile",
                                "batch",
                                "active_requests",
                            ),
                            cell,
                            strict=True,
                        )
                    ),
                    "warmups": WARMUPS,
                    "input_seed_base": INPUT_SEED_BASE,
                    "cell_timeout_seconds": args.cell_timeout_seconds,
                    "warmup_accounting_available": True,
                    "warmup_repetitions_attempted": warmup_attempted,
                    "warmup_paired_repetitions_completed": warmup_paired,
                    "warmup_policy_runs_completed": warmup_runs,
                    "warmup_failures": [
                        failure for failure in terminal.values() if failure["phase"] == "warmup"
                    ],
                    "measured_repetitions": MEASURED_REPETITIONS,
                    "repetitions": repetitions,
                    "policy_status": {
                        policy: {
                            "status": "failed",
                            "measured_repetitions": sum(
                                policy in row.get("policies", {}) for row in repetitions
                            ),
                            "failure": terminal[policy],
                        }
                        for policy in POLICIES
                    },
                    "arm_metadata": arm_metadata,
                    "p2_causal_gate_passed": p2_gate_passed,
                    "elapsed_seconds": time.monotonic() - started,
                }
                reference._cleanup()
            if not timeout_cancelled:
                reference.cancel_cell_timeout(previous_handler)
            payload.update(
                {
                    "source": {
                        "commit": reference._head(),
                        "dirty": False,
                        "implementation_digest": implementation,
                    },
                    "manifest": _dependency(args.manifest),
                    "p2_audit": _dependency(args.p2_audit),
                    "p3_audit": _dependency(args.p3_audit),
                    **dependency_metadata[(scale, budget)],
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(device),
                    },
                    "command": [sys.executable, *sys.argv],
                }
            )
            reference._write_json(artifact, payload)
            new_cells += 1
        runs[cell] = _run_row(cell, artifact, payload)
        reference._write_json(
            args.matrix_progress,
            {
                "schema_version": 1,
                "experiment_id": "p4-adaptive-systems-matrix-progress-v1",
                "source_commit": reference._head(),
                "implementation_digest": implementation,
                "manifest": _dependency(args.manifest),
                "p2_audit": _dependency(args.p2_audit),
                "p3_audit": _dependency(args.p3_audit),
                "expected_cells": EXPECTED_CELLS,
                "terminal_cells": len(runs),
                "complete_cells": sum(row["status"] == "complete" for row in runs.values()),
                "partial_cells": sum(row["status"] == "partial" for row in runs.values()),
                "failed_cells": sum(row["status"] == "failed" for row in runs.values()),
                "runs": sorted(
                    runs.values(),
                    key=lambda row: tuple(
                        row[name]
                        for name in (
                            "scale",
                            "budget",
                            "context",
                            "generation",
                            "profile",
                        )
                    ),
                ),
            },
        )
        print(json.dumps({"cell": payload["cell"], "status": payload["status"]}), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
