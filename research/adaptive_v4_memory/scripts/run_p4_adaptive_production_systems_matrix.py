from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import p4_adaptive_continuous_batch_adapter as adapter_contract
import run_p4_adaptive_systems_matrix as adaptive
import run_p4_production_systems_matrix as production
from adaptive_v4_gpu_lock import acquire_gpu_lock

SCALES = adaptive.SCALES
BUDGETS = adaptive.BUDGETS
CONTEXTS = production.CONTEXTS
GENERATIONS = production.GENERATIONS
LOAD_PROFILES = production.LOAD_PROFILES
POLICIES = adapter_contract.POLICIES
WARMUPS = adapter_contract.WARMUPS
MEASURED_REPETITIONS = adapter_contract.MEASURED_REPETITIONS
INPUT_SEED_BASE = adapter_contract.INPUT_SEED_BASE
CELL_TIMEOUT_SECONDS = production.CELL_TIMEOUT_SECONDS
EXPECTED_CELLS = 432
Cell = tuple[str, str, int, int, str, int, int]

IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4",
    "research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json",
    "research/adaptive_v4_memory/manifests/p4-adaptive-production-systems-matrix-v1.json",
    "research/adaptive_v4_memory/scripts/p4_continuous_batch_adapter.py",
    "research/adaptive_v4_memory/scripts/p4_adaptive_continuous_batch_adapter.py",
    "research/adaptive_v4_memory/scripts/run_p4_adaptive_systems_matrix.py",
    "research/adaptive_v4_memory/scripts/run_p4_production_systems_matrix.py",
    "research/adaptive_v4_memory/scripts/run_p4_adaptive_production_systems_matrix.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def frozen_cells() -> tuple[Cell, ...]:
    return tuple(
        (scale, budget, context, generation, name, batch, concurrency)
        for scale, budget, context, generation, (name, batch, concurrency) in product(
            SCALES, BUDGETS, CONTEXTS, GENERATIONS, LOAD_PROFILES
        )
    )


def cell_dict(cell: Cell) -> dict[str, Any]:
    return dict(
        zip(
            ("scale", "budget", "context", "generation", "profile", "batch", "concurrency"),
            cell,
            strict=True,
        )
    )


def cell_path(root: Path, cell: Cell) -> Path:
    scale, budget, context, generation, profile, _batch, _concurrency = cell
    return (
        root
        / scale
        / f"budget-{budget}"
        / f"context-{context}"
        / f"generation-{generation}"
        / profile
    )


def schedule_index(cell: Cell, repetition: int) -> int:
    if not 0 <= repetition < WARMUPS + MEASURED_REPETITIONS:
        raise ValueError("Adaptive production repetition is outside the frozen schedule.")
    return frozen_cells().index(cell) * (WARMUPS + MEASURED_REPETITIONS) + repetition


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
        raise RuntimeError(f"Untracked adaptive production implementation paths: {missing}")
    return hashlib.sha256(tree.encode()).hexdigest()


def _dependency(path: Path) -> dict[str, str]:
    _require(path.is_file(), f"Missing adaptive production dependency: {path}")
    return {"path": str(path), "sha256": production.sha256(path)}


def require_p3_adaptive_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-ruler-audit-v1"
        and payload.get("status") == "terminal"
        and audit.get("total_predictions") == 65_000
        and audit.get("paired_examples") == 32_500
        and audit.get("all_raw_records_verified") is True
        and audit.get("all_dependency_digests_verified") is True
        and audit.get("quota_physical_audits_verified") is True
        and audit.get("same_global_token_budget_verified") is True
        and audit.get("outcome_dependent_execution") is False,
        "Adaptive production requires the terminal natural adaptive-quota audit.",
    )
    return payload


def _schedule(arms: dict[str, Any], *, cell: Cell) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repetition in range(WARMUPS + MEASURED_REPETITIONS):
        policies: dict[str, Any] = {}
        totals: set[int] = set()
        for policy in POLICIES:
            arm = arms[policy]
            config = arm.config_for_batch(schedule_index(cell, repetition))
            variant = "single"
            if len(arm.configs) == 2:
                variant = "low" if config is arm.configs[0] else "high"
            policies[policy] = {
                "config": asdict(config),
                "sha256": adapter_contract.config_digest(config),
                "variant": variant,
            }
            totals.add(sum(blocks for _, blocks in config.layer_budgets))
        _require(len(totals) == 1, "Paired controller schedules lost memory matching.")
        rows.append(
            {
                "phase_repetition": repetition,
                "global_schedule_index": schedule_index(cell, repetition),
                "policies": policies,
            }
        )
    return rows


def build_spec(
    *,
    cell: Cell,
    arms: dict[str, Any],
    checkpoint: Path,
    manifest: Path,
    p2_audit: Path,
    p3_adaptive_audit: Path,
    calibration: Path,
    memory_match: Path,
    cell_timeout_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "p4-adaptive-production-adapter-spec-v1",
        "cell": cell_dict(cell),
        "policies": list(POLICIES),
        "protected_end_positions": list(adapter_contract.PROTECTED_END_POSITIONS),
        "warmups": WARMUPS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "input_seed_base": INPUT_SEED_BASE,
        "cell_timeout_seconds": cell_timeout_seconds,
        "repetition_seeds": [
            INPUT_SEED_BASE + index for index in range(WARMUPS + MEASURED_REPETITIONS)
        ],
        "checkpoint": str(checkpoint),
        "manifest": _dependency(manifest),
        "p2_audit": _dependency(p2_audit),
        "p3_adaptive_audit": _dependency(p3_adaptive_audit),
        "calibration": _dependency(calibration),
        "memory_match": _dependency(memory_match),
        "controller_schedule": _schedule(arms, cell=cell),
    }


def _validate_controller(run: dict[str, Any], *, config: Any, batch: int, concurrency: int) -> None:
    controller = run.get("adaptive_controller")
    if not isinstance(controller, dict):
        raise ValueError("Adaptive controller audit is missing.")
    configured = dict(config.layer_budgets)
    expected_physical = {
        layer: blocks * batch * concurrency for layer, blocks in configured.items()
    }
    recorded = {
        int(layer): value
        for layer, value in controller.get("configured_blocks_per_sequence_by_layer", {}).items()
    }
    physical = {
        int(layer): value
        for layer, value in controller.get("configured_physical_hot_blocks_by_layer", {}).items()
    }
    observed = {
        int(layer): value
        for layer, value in controller.get("observed_hot_blocks_by_layer", {}).items()
    }
    _require(
        controller.get("enabled") is True
        and controller.get("protected_end_positions")
        == list(adapter_contract.PROTECTED_END_POSITIONS)
        and controller.get("config_sha256") == adapter_contract.config_digest(config)
        and recorded == configured
        and physical == expected_physical
        and set(observed) == set(configured)
        and all(
            type(value) is int and 0 <= value <= expected_physical[layer]
            for layer, value in observed.items()
        )
        and controller.get("controller_time_ns") == run.get("timing", {}).get("controller_time_ns"),
        "Adaptive controller physical audit drifted.",
    )


def validate_adapter_payload(payload: dict[str, Any], *, cell: Cell, spec: dict[str, Any]) -> None:
    _require(
        payload.get("experiment_id") == "p4-adaptive-production-adapter-cell-v1"
        and payload.get("cell") == cell_dict(cell)
        and payload.get("input_seed_base") == INPUT_SEED_BASE
        and payload.get("warmups") == WARMUPS
        and payload.get("measured_repetitions") == MEASURED_REPETITIONS,
        "Adaptive production adapter identity drifted.",
    )
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) > MEASURED_REPETITIONS:
        raise ValueError("Invalid adaptive production repetition coverage.")
    failures: dict[str, Any] = {}
    warmup_failures = payload.get("warmup_failures")
    if not isinstance(warmup_failures, list):
        raise ValueError("Adaptive production warmup failures are missing.")
    for failure in warmup_failures:
        if not isinstance(failure, dict):
            raise ValueError("Warmup failure provenance drifted.")
        policy = failure.get("policy")
        if not isinstance(policy, str) or policy not in POLICIES or policy in failures:
            raise ValueError("Warmup failure provenance drifted.")
        failures[policy] = failure
    reference_cell = (cell[0], cell[2], cell[3], cell[4], cell[5], cell[6])
    for index, row in enumerate(repetitions):
        _require(
            row.get("repetition") == index
            and row.get("input_seed") == INPUT_SEED_BASE + WARMUPS + index
            and tuple(row.get("execution_order", ()))
            == (POLICIES if index % 2 == 0 else tuple(reversed(POLICIES))),
            "Adaptive production repetition schedule drifted.",
        )
        runs = row.get("policies", {})
        run_failures = row.get("policy_failures", {})
        if not isinstance(runs, dict) or not isinstance(run_failures, dict):
            raise ValueError("Adaptive production policy coverage drifted.")
        _require(
            set(runs).issubset(POLICIES)
            and set(run_failures).issubset(POLICIES)
            and not (set(runs) & set(run_failures)),
            "Adaptive production policy coverage drifted.",
        )
        for policy, failure in run_failures.items():
            _require(policy not in failures, "Adaptive production policy failed twice.")
            failures[policy] = failure
        schedule_index = WARMUPS + index
        for policy, run in runs.items():
            production.validate_policy_run(run, cell=reference_cell)
            config = adapter_contract.config_from_payload(
                spec["controller_schedule"][schedule_index]["policies"][policy]["config"]
            )
            _validate_controller(run, config=config, batch=cell[5], concurrency=cell[6])
            expected_metadata = spec["controller_schedule"][schedule_index]["policies"][policy]
            _require(
                row.get("policy_configs", {}).get(policy)
                == {"sha256": expected_metadata["sha256"], "variant": expected_metadata["variant"]},
                "Adaptive production config provenance drifted.",
            )
        if set(runs) == set(POLICIES):
            equal = runs[POLICIES[0]]["prediction_digest"] == runs[POLICIES[1]]["prediction_digest"]
            _require(
                row.get("prediction_digests_equal") is equal, "Prediction equality record drifted."
            )
    status = payload.get("policy_status", {})
    if not isinstance(status, dict) or set(status) != set(POLICIES):
        raise ValueError("Policy status set drifted.")
    counts = {
        policy: sum(policy in row.get("policies", {}) for row in repetitions) for policy in POLICIES
    }
    for policy in POLICIES:
        _require(
            status[policy].get("measured_repetitions") == counts[policy]
            and status[policy].get("status")
            == ("complete" if counts[policy] == MEASURED_REPETITIONS else "failed"),
            "Adaptive production terminal accounting drifted.",
        )
    complete = sum(count == MEASURED_REPETITIONS for count in counts.values())
    expected = "complete" if complete == 2 else "partial" if complete == 1 else "failed"
    _require(payload.get("status") == expected, "Adaptive production cell status drifted.")


def _run_adapter(adapter: Path, spec: Path, output: Path, timeout: float) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    process = subprocess.Popen(
        [str(adapter), "--spec", str(spec), "--output", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    try:
        _stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise TimeoutError("Adaptive production adapter exceeded its cell timeout.") from error
    if process.returncode != 0:
        raise RuntimeError(
            f"Adaptive production adapter exited {process.returncode}: {stderr[-2000:]}"
        )
    _require(output.is_file(), "Adaptive production adapter wrote no raw artifact.")
    return json.loads(output.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_row(cell: Cell, artifact: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **cell_dict(cell),
        "status": payload["adapter_payload"]["status"],
        "artifact": {"path": str(artifact), "sha256": production.sha256(artifact)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen adaptive production P4 matrix.")
    parser.add_argument(
        "--adapter-executable",
        type=Path,
        default=Path("research/adaptive_v4_memory/scripts/p4_adaptive_continuous_batch_adapter.py"),
    )
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
        default=Path(
            "research/adaptive_v4_memory/manifests/p4-adaptive-production-systems-matrix-v1.json"
        ),
    )
    parser.add_argument(
        "--p2-audit",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"),
    )
    parser.add_argument(
        "--p3-adaptive-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/ruler-qwen3-4b.summary.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-production-systems"),
    )
    parser.add_argument(
        "--matrix-progress",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-production-systems-matrix.json"
        ),
    )
    args = parser.parse_args()
    production.validate_cell_timeout(args.cell_timeout_seconds)
    _require(
        args.max_new_cells is None or args.max_new_cells > 0, "max-new-cells must be positive."
    )
    adapter = args.adapter_executable.resolve()
    _require(
        adapter.is_file() and os.access(adapter, os.X_OK),
        "Adaptive production adapter is not executable.",
    )
    _require(not production._dirty(), "Adaptive production execution requires a clean tree.")
    adaptive.require_nine_seed_causal_audit(args.p2_audit)
    require_p3_adaptive_audit(args.p3_adaptive_audit)
    manifest = json.loads(args.manifest.read_text())
    _require(
        manifest.get("experiment_id") == "p4-adaptive-production-systems-matrix-v1"
        and manifest.get("primary_paired_cells") == EXPECTED_CELLS
        and manifest.get("input_seed_base") == INPUT_SEED_BASE,
        "Frozen adaptive production manifest is required.",
    )
    lock = acquire_gpu_lock("p4-adaptive-production-systems-matrix")
    implementation = implementation_digest()
    selected = [
        cell
        for cell in frozen_cells()
        if (not args.scale or cell[0] in args.scale)
        and (not args.budget or cell[1] in args.budget)
        and (not args.context or cell[2] in args.context)
        and (not args.generation or cell[3] in args.generation)
        and (not args.profile or cell[4] in args.profile)
    ]
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    paths: dict[str, tuple[Path, Path, Path]] = {}
    for scale in SCALES:
        checkpoint = args.training_root / scale / "seed-6071401" / f"{scale}-step-1000.pt"
        calibration = args.calibration_root / scale / "seed-6071401" / "p1-layer-quotas.json"
        memory_match = (
            args.memory_match_root
            / scale
            / "seed-6071401"
            / "p2-causal-hot-memory-match.summary.json"
        )
        paths[scale] = (checkpoint, calibration, memory_match)
        for budget in BUDGETS:
            bundles[(scale, budget)] = adaptive.load_policy_arms(
                scale=scale,
                budget=budget,
                checkpoint=checkpoint,
                calibration_path=calibration,
                memory_match_path=memory_match,
            )[0]
    runs: dict[Cell, dict[str, Any]] = {}
    new_cells = 0
    for cell in selected:
        scale, budget = cell[:2]
        root = cell_path(args.output_root, cell)
        artifact = root / "cell.json"
        if artifact.is_file():
            try:
                existing = json.loads(artifact.read_text())
                spec_path = Path(existing["adapter_spec"]["path"])
                spec = adapter_contract.validate_spec(spec_path)
                validate_adapter_payload(existing["adapter_payload"], cell=cell, spec=spec)
                runs[cell] = _run_row(cell, artifact, existing)
                continue
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if args.max_new_cells is not None and new_cells >= args.max_new_cells:
            break
        checkpoint, calibration, memory_match = paths[scale]
        spec = build_spec(
            cell=cell,
            arms=bundles[(scale, budget)],
            checkpoint=checkpoint,
            manifest=args.manifest,
            p2_audit=args.p2_audit,
            p3_adaptive_audit=args.p3_adaptive_audit,
            calibration=calibration,
            memory_match=memory_match,
            cell_timeout_seconds=args.cell_timeout_seconds,
        )
        spec_path = root / "adapter-spec.json"
        raw_path = root / "adapter-raw.json"
        _write_json(spec_path, spec)
        adapter_contract.validate_spec(spec_path)
        started = time.monotonic()
        adapter_payload = _run_adapter(adapter, spec_path, raw_path, args.cell_timeout_seconds)
        validate_adapter_payload(adapter_payload, cell=cell, spec=spec)
        payload = {
            "schema_version": 1,
            "experiment_id": "p4-adaptive-production-systems-cell-v1",
            "cell": cell_dict(cell),
            "source": {
                "commit": production._head(),
                "dirty": False,
                "implementation_digest": implementation,
            },
            "manifest": _dependency(args.manifest),
            "p2_audit": _dependency(args.p2_audit),
            "p3_adaptive_audit": _dependency(args.p3_adaptive_audit),
            "adapter": {"path": str(adapter), "sha256": production.sha256(adapter)},
            "adapter_spec": {"path": str(spec_path), "sha256": production.sha256(spec_path)},
            "adapter_raw": {"path": str(raw_path), "sha256": production.sha256(raw_path)},
            "cell_timeout_seconds": args.cell_timeout_seconds,
            "wall_time_seconds": time.monotonic() - started,
            "adapter_payload": adapter_payload,
        }
        _write_json(artifact, payload)
        runs[cell] = _run_row(cell, artifact, payload)
        new_cells += 1
        _write_json(
            args.matrix_progress,
            {
                "schema_version": 1,
                "experiment_id": "p4-adaptive-production-systems-matrix-progress-v1",
                "implementation_digest": implementation,
                "expected_cells": EXPECTED_CELLS,
                "terminal_cells": len(runs),
                "runs": sorted(
                    runs.values(),
                    key=lambda row: tuple(row[name] for name in cell_dict(frozen_cells()[0])),
                ),
            },
        )
    del lock


if __name__ == "__main__":
    main()
