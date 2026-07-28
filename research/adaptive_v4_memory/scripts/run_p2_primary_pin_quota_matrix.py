from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import calibrate_p2_direct_soft_lag as calibrator
import evaluate_p2_direct_controller_shard_v1_3 as evaluator
import p2_direct_attestation as attestation
import p2_direct_controller_contract_v1_3 as contract
import p2_direct_controller_reuse_admission_v1_3 as reuse_admission
import torch
from adaptive_v4_gpu_lock import acquire_device_guard, acquire_gpu_lock

from nano_deepseek_v4 import generate_adaptive_memory_workload

EXPERIMENT_ID = "p2-primary-pin-quota-research-matrix-v1"
ARMS = ("fixed", "fixed+pins", "calibrated-no-pins", "calibrated+pins")
CALIBRATION_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration-v1-2")
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_primary_pin_quota_research_v1")
EXPECTED_CELLS = 9_000
def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
def _implementation_digest() -> str:
    paths = (Path(__file__), Path(calibrator.__file__), Path(evaluator.__file__), Path(contract.__file__))
    return _digest([[str(path), _file_digest(path)] for path in paths])
def coordinates() -> list[dict[str, Any]]:
    return [
        {
            "scale": scale,
            "training_seed": seed,
            "budget": budget,
            "family": family,
            "context": context,
            "replicate": replicate,
        }
        for scale, seed, budget, family, context, replicate in product(
            contract.SCALES,
            contract.TRAINING_SEEDS,
            contract.BUDGETS,
            contract.FAMILIES,
            contract.CONTEXTS,
            contract.REPLICATES,
        )
    ]
def cell_path(root: Path, coordinate: dict[str, Any]) -> Path:
    return (
        root
        / cast(str, coordinate["scale"])
        / f"seed-{coordinate['training_seed']}"
        / cast(str, coordinate["budget"])
        / cast(str, coordinate["family"])
        / f"context-{coordinate['context']}"
        / f"replicate-{coordinate['replicate']}.json"
    )
@dataclass
class TokenAggregate:
    rows: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    def __post_init__(self) -> None:
        self._hot_blocks = hashlib.sha256()
        self._hot_bytes = hashlib.sha256()
    def add(self, row: dict[str, Any]) -> None:
        evaluator._validate_sealed_row(row, evaluator.TOKEN_SCHEMA_ID)
        materialization = cast(list[dict[str, Any]], row["post_rebalance_materialization"])
        hot_blocks = sum(cast(int, item["hot_blocks"]) for item in materialization)
        hot_bytes = sum(cast(int, item["hot_bytes"]) for item in materialization)
        self._hot_blocks.update(f"{hot_blocks}\n".encode())
        self._hot_bytes.update(f"{hot_bytes}\n".encode())
        deltas = cast(list[dict[str, Any]], row["decode_incremental_transfer_deltas"])
        self.h2d_bytes += sum(cast(int, item["h2d_delta_bytes"]) for item in deltas)
        self.d2h_bytes += sum(cast(int, item["d2h_delta_bytes"]) for item in deltas)
        self.peak_allocated_bytes = max(
            self.peak_allocated_bytes, cast(int, row["cuda_peak_allocated_bytes"])
        )
        self.peak_reserved_bytes = max(
            self.peak_reserved_bytes, cast(int, row["cuda_peak_reserved_bytes"])
        )
        self.rows += 1
    def result(self) -> dict[str, Any]:
        return {
            "token_rows": self.rows,
            "hot_blocks_sequence_sha256": self._hot_blocks.hexdigest(),
            "hot_bytes_sequence_sha256": self._hot_bytes.hexdigest(),
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "cuda_hbm_evidence": True,
        }
def _cohort_paths(scale: str, seed: int) -> dict[str, Path]:
    calibration = CALIBRATION_ROOT / scale / f"seed-{seed}" / f"{scale}-calibration.json"
    payload = json.loads(calibration.read_text())
    training = cast(dict[str, Any], payload["training_summary"])
    return {
        "calibration": calibration,
        "checkpoint": Path(cast(str, cast(dict[str, Any], payload["checkpoint"])["path"])),
        "training_summary": Path(cast(str, training["path"])),
        "training_matrix": Path(
            cast(str, cast(dict[str, Any], training["terminal_matrix_ledger"])["path"])
        ),
    }
def _bound_file(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"Frozen research input is a symlink: {path}")
    digest = _file_digest(path)
    if (path.resolve(strict=True) != Path(binding["path"]).resolve(strict=True)
            or path.stat().st_size != binding["bytes"] or digest != binding["sha256"]):
        raise ValueError(f"Frozen research input binding drifted: {path}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
def _cohort_binding(paths: dict[str, Path], calibration: dict[str, Any]) -> dict[str, Any]:
    training = cast(dict[str, Any], calibration["training_summary"])
    expected = {"checkpoint": cast(dict[str, Any], calibration["checkpoint"]),
                "training_summary": training,
                "training_matrix": cast(dict[str, Any], training["terminal_matrix_ledger"])}
    dependencies = {"calibration": {"path": str(paths["calibration"]),
                    "sha256": _file_digest(paths["calibration"]),
                    "bytes": paths["calibration"].stat().st_size}}
    dependencies.update({name: _bound_file(paths[name], item) for name, item in expected.items()})
    body = {"calibration_payload_sha256": calibration["payload_sha256"], "dependencies": dependencies}
    return {**body, "input_binding_digest": _digest(body)}
def _research_arms(calibration: dict[str, Any], budget: str, scale: str, seed: int) -> dict[str, Any]:
    cell, validated_budgets = contract.v1_2._validated_calibration_coordinate(
        calibration, budget, expected_scale=scale, expected_training_seed=seed,
        expected_global_block_budget=contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
        expected_csa_layers=contract.DIRECT_CSA_LAYERS_BY_SCALE[scale])
    legacy = dict(calibration)
    legacy["experiment_id"] = contract.v1_2.LEGACY_CALIBRATION_SCAFFOLD_ID
    scaffolds, metadata = contract.build_arm_configs(legacy, budget)
    calibrated_budgets = tuple(metadata["calibrated_layer_budgets"])
    if calibrated_budgets != validated_budgets:
        raise ValueError("Research arm calibration budgets drifted.")
    exact = contract.v1_2._balanced_fixed_budgets(
        calibrated_budgets, cast(str, cell["quota"]["calibration_digest"]))
    signal = scaffolds["hierarchical+pins"].configs[0].signal
    built: dict[str, Any] = {}
    for name in contract.ALL_ARM_NAMES:
        semantics = contract.EXPECTED_ARM_SEMANTICS[name]
        if semantics.quota_runtime in {"balanced-feasible", "soft-lag", "soft-lag-permuted"}:
            built[name] = contract.v1_2._exact_fixed_scaffold(
                scaffolds[semantics.scaffold_name], semantics,
                layer_budgets=exact, signal_config=signal)
        else:
            built[name] = contract.v1_2._rename_scaffold(scaffolds[semantics.scaffold_name], semantics)
    contract.validate_arm_semantics(built)
    return {name: built[name] for name in ARMS}
def _load_cohort(
    scale: str, seed: int, *, trust_root: attestation.TrustRoot, device: torch.device
) -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    paths = _cohort_paths(scale, seed)
    if paths["calibration"].is_symlink():
        raise ValueError("Frozen calibration input is a symlink.")
    calibration = calibrator.validate_calibration_artifact(
        json.loads(paths["calibration"].read_text()), verify_bindings=False, trust_root=trust_root)
    cohort_binding = _cohort_binding(paths, calibration)
    arms_by_budget = {budget: _research_arms(calibration, budget, scale, seed)
                      for budget in contract.BUDGETS}
    raw_checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    model = evaluator._load_checkpoint_model(
        raw_checkpoint,
        scale=scale,
        training_seed=seed,
        device=device,
        dtype=torch.bfloat16,
    )
    del raw_checkpoint
    return model, calibration, arms_by_budget, cohort_binding
def _run_cell(
    model: Any,
    calibration: dict[str, Any],
    arms: dict[str, Any],
    coordinate: dict[str, Any],
    *,
    source: dict[str, Any],
    implementation_digest: str,
    cohort_binding: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    scale = cast(str, coordinate["scale"])
    seed = cast(int, coordinate["training_seed"])
    budget = cast(str, coordinate["budget"])
    family = cast(str, coordinate["family"])
    context = cast(int, coordinate["context"])
    replicate = cast(int, coordinate["replicate"])
    calibration_seed, evaluation_seed = evaluator._validate_coordinate(
        scale=scale,
        training_seed=seed,
        budget=budget,
        family=family,
        context=context,
        replicate=replicate,
    )
    generation_seed = contract.generation_seed(evaluation_seed, family, context, replicate)
    generator = torch.Generator().manual_seed(generation_seed)
    task = evaluator._task(model)
    examples: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    features = contract.expected_arm_features()
    for example_index in range(contract.EXAMPLES_PER_SHARD):
        offset = replicate * contract.EXAMPLES_PER_SHARD + example_index
        workload = generate_adaptive_memory_workload(
            task,
            family=family,
            batch_size=contract.BATCH_SIZE,
            sequence_length=context,
            generator=generator,
            conversation_offset=offset,
            device=next(model.parameters()).device,
        )
        schedule = evaluator.shard_schedule_index(
            family=family, context=context, replicate=replicate, example_index=example_index
        )
        order = tuple(name for name in evaluator.arm_execution_order(schedule) if name in ARMS)
        examples.append(
            {
                "example_index": example_index,
                "workload_digest": evaluator._workload_digest(workload),
                "targets": workload.targets.detach().cpu().tolist()[0],
                "query_positions": workload.query_positions.detach().cpu().tolist()[0],
                "protected_end_positions": list(workload.protected_end_positions),
                "execution_order": list(order),
            }
        )
        for execution_index, arm_name in enumerate(order):
            aggregate = TokenAggregate()
            outcome = evaluator.run_arm_example(
                model,
                workload,
                arm_name=arm_name,
                built_arm=arms[arm_name],
                calibration=calibration,
                budget=budget,
                schedule_index=schedule,
                execution_index=execution_index,
                example_index=example_index,
                token_sink=aggregate.add,
            )
            evaluator._validate_sealed_row(outcome, evaluator.OUTCOME_SUCCESS_SCHEMA_ID)
            expected_config = evaluator.runtime_config_for_arm(
                arm_name, arms[arm_name], calibration, budget, schedule
            )
            if (
                outcome["semantics"] != features[arm_name]
                or evaluator._runtime_config_from_payload(outcome["runtime_config"])
                != expected_config
                or outcome["config_sha256"] != contract.json_digest(outcome["runtime_config"])
            ):
                raise ValueError("Primary arm semantics or runtime config drifted.")
            outcomes.append(
                {
                    "example_index": example_index,
                    "arm": arm_name,
                    "execution_index": execution_index,
                    "config_sha256": outcome["config_sha256"],
                    "predictions": outcome["predictions"],
                    "token_rows_digest": outcome["token_rows_digest"],
                    "token_summary": aggregate.result(),
                }
            )
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "terminal",
        "claim_boundary": (
            "Research-only fresh-cohort 2x2 pin/quota contrast; not a v1.3.4 artifact or "
            "official DeepSeek-V4 result."
        ),
        "source": source,
        "implementation_digest": implementation_digest,
        "coordinate": {
            **coordinate,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "generation_seed": generation_seed,
            "global_block_budget": contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
        },
        "arms": list(ARMS),
        "cohort_binding": cohort_binding,
        "environment": environment,
        "examples": examples,
        "outcomes": outcomes,
    }
    return {**body, "payload_sha256": _digest(body)}
def validate_cell(payload: dict[str, Any], coordinate: dict[str, Any]) -> None:
    source = dict(payload)
    observed_digest = source.pop("payload_sha256", None)
    scale = cast(str, coordinate["scale"])
    seed = cast(int, coordinate["training_seed"])
    budget = cast(str, coordinate["budget"])
    family = cast(str, coordinate["family"])
    context = cast(int, coordinate["context"])
    replicate = cast(int, coordinate["replicate"])
    calibration_seed, evaluation_seed = evaluator._validate_coordinate(
        scale=scale,
        training_seed=seed,
        budget=budget,
        family=family,
        context=context,
        replicate=replicate,
    )
    expected_coordinate = {
        **coordinate,
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "generation_seed": contract.generation_seed(
            evaluation_seed, family, context, replicate
        ),
        "global_block_budget": contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != "terminal"
        or payload.get("coordinate") != expected_coordinate
        or tuple(payload.get("arms", ())) != ARMS
        or len(payload.get("examples", ())) != contract.EXAMPLES_PER_SHARD
        or len(payload.get("outcomes", ())) != contract.EXAMPLES_PER_SHARD * len(ARMS)
        or observed_digest != _digest(source)
    ):
        raise ValueError(f"Invalid primary pin/quota cell: {coordinate}")
    expected_orders = {
        example_index: tuple(
            name
            for name in evaluator.arm_execution_order(
                evaluator.shard_schedule_index(
                    family=family,
                    context=context,
                    replicate=replicate,
                    example_index=example_index,
                )
            )
            if name in ARMS
        )
        for example_index in range(contract.EXAMPLES_PER_SHARD)
    }
    examples = cast(list[dict[str, Any]], payload["examples"])
    if (
        {cast(int, example["example_index"]) for example in examples}
        != set(expected_orders)
        or any(
            tuple(example.get("execution_order", ()))
            != expected_orders[cast(int, example["example_index"])]
            for example in examples
        )
    ):
        raise ValueError("Primary pin/quota example inventory or execution order drifted.")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in cast(list[dict[str, Any]], payload["outcomes"]):
        grouped.setdefault(cast(int, row["example_index"]), []).append(row)
    if set(grouped) != set(expected_orders):
        raise ValueError("Primary pin/quota outcome example inventory drifted.")
    for example_index, rows in grouped.items():
        observed_order = {
            cast(int, row["execution_index"]): cast(str, row["arm"]) for row in rows
        }
        if (
            len(rows) != len(ARMS)
            or len(observed_order) != len(ARMS)
            or tuple(observed_order.get(index) for index in range(len(ARMS)))
            != expected_orders[example_index]
        ):
            raise ValueError("Primary pin/quota arm inventory drifted.")
        for field in ("hot_blocks_sequence_sha256", "hot_bytes_sequence_sha256"):
            if len({cast(dict[str, Any], row["token_summary"])[field] for row in rows}) != 1:
                raise ValueError("Paired arms did not use identical actual hot memory.")
def _summary(completed: list[tuple[dict[str, Any], Path]]) -> dict[str, Any]:
    cells = [{"coordinate": coordinate, "path": str(path), "sha256": _file_digest(path)} for coordinate, path in completed]
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if len(cells) == EXPECTED_CELLS else "in_progress",
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": len(cells),
        "arms": list(ARMS),
        "quality_analysis_performed": False,
        "cells": cells,
    }
    return {**body, "payload_sha256": _digest(body)}
def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only P2 primary pin/quota matrix.")
    parser.add_argument("--attestation-key-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--prerequisites-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        raise ValueError("max-new-cells must be positive.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip():
        raise RuntimeError("Primary pin/quota matrix requires a clean source tree.")
    source = {"commit": commit, "dirty": False}
    manifest = contract.load_v1_3_4_manifest()
    trust_root = attestation.load_trust_root(
        args.attestation_key_path,
        repository_root=Path.cwd(),
        artifact_roots=(Path("artifacts"),),
        expected_key_id=cast(str, manifest["attestation"]["key_id"]),
    )
    activation = json.loads(contract.V1_3_4_QUALITY_START_ACTIVATION_PATH.read_text())
    reuse_admission._verify_attested_payload(
        activation, trust_root=trust_root,
        purpose=reuse_admission.V1_3_4_QUALITY_START_ACTIVATION_PURPOSE,
        label="Research input v1.3.4 activation")
    captured = execution_environment.capture_execution_environment()
    expected_environment = activation["base_prerequisites_binding"][
        "execution_environment_projection"
    ]
    if execution_environment.controller_compatible_environment_projection(captured) != expected_environment:
        raise RuntimeError("Current execution environment differs from the frozen direct cohort.")
    routing_identity = execution_environment.selected_device_routing_identity(captured)
    device, _ = execution_environment.activate_explicit_cuda_device(
        execution_environment.explicit_cuda_device_spec(captured),
        expected_routing_identity=routing_identity,
    )
    for scale, seed in product(contract.SCALES, contract.TRAINING_SEEDS):
        paths = _cohort_paths(scale, seed)
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Missing direct cohort prerequisite: {scale}/{seed}")
    if args.prerequisites_only:
        print(json.dumps({"status": "prerequisites_validated", "expected_cells": EXPECTED_CELLS}))
        return
    implementation_digest = _implementation_digest()
    selected = coordinates()
    completed: list[tuple[dict[str, Any], Path]] = []
    dependency_bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for coordinate in selected:
        path = cell_path(args.output_root, coordinate)
        if path.is_file():
            payload = json.loads(path.read_text())
            validate_cell(payload, coordinate)
            cohort = (cast(str, coordinate["scale"]), cast(int, coordinate["training_seed"]))
            if cohort not in dependency_bindings:
                paths = _cohort_paths(*cohort)
                dependency_bindings[cohort] = _cohort_binding(
                    paths, json.loads(paths["calibration"].read_text()))
            if (
                payload.get("source") != source
                or payload.get("implementation_digest") != implementation_digest
                or payload.get("cohort_binding") != dependency_bindings[cohort]
            ):
                raise RuntimeError("Primary pin/quota resume binding drifted.")
            completed.append((coordinate, path))
    unexpected = {
        path
        for path in args.output_root.rglob("*")
        if path.is_file() and path != args.output_root / "matrix.summary.json"
    } - {path for _, path in completed}
    if unexpected:
        raise RuntimeError("Primary pin/quota output root contains unregistered files.")
    new_cells = 0
    generic_lease = acquire_gpu_lock("p2-primary-pin-quota-research")
    device_lease = acquire_device_guard("p2-primary-pin-quota-research", routing_identity)
    try:
        current_cohort: tuple[str, int] | None = None
        model: Any = None
        calibration: dict[str, Any] = {}
        arms_by_budget: dict[str, dict[str, Any]] = {}
        cohort_binding: dict[str, Any] = {}
        for coordinate in selected:
            path = cell_path(args.output_root, coordinate)
            if path.is_file():
                continue
            if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                break
            cohort = (cast(str, coordinate["scale"]), cast(int, coordinate["training_seed"]))
            if cohort != current_cohort:
                if model is not None:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                model, calibration, arms_by_budget, cohort_binding = _load_cohort(
                    *cohort, trust_root=trust_root, device=device
                )
                current_cohort = cohort
            payload = _run_cell(
                model,
                calibration,
                arms_by_budget[cast(str, coordinate["budget"])],
                coordinate,
                source=source,
                implementation_digest=implementation_digest,
                cohort_binding=cohort_binding,
                environment={
                    "compatible_projection": expected_environment,
                    "routing_identity": routing_identity,
                    "device": str(device),
                },
            )
            validate_cell(payload, coordinate)
            _atomic_json(path, payload)
            completed.append((coordinate, path))
            new_cells += 1
            print(json.dumps({"status": "cell_committed", "coordinate": coordinate}), flush=True)
        _atomic_json(args.output_root / "matrix.summary.json", _summary(completed))
    finally:
        device_lease.close()
        generic_lease.close()
    status = "complete" if len(completed) == EXPECTED_CELLS else "in_progress"
    print(json.dumps({"status": status, "completed_cells": len(completed),
                      "expected_cells": EXPECTED_CELLS, "new_cells": new_cells,
                      "quality_analysis_performed": False}, sort_keys=True))
if __name__ == "__main__":
    main()
