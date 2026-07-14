#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import p4_continuous_batch_adapter as base
import torch

from nano_deepseek_v4 import SameTokenControllerConfig, TrainingFreeControllerConfig

POLICIES = ("fixed+pins", "calibrated+pins")
WARMUPS = 5
MEASURED_REPETITIONS = 30
INPUT_SEED_BASE = 9_271_400
PROTECTED_END_POSITIONS = (3,)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def config_digest(config: SameTokenControllerConfig) -> str:
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def config_from_payload(payload: Any) -> SameTokenControllerConfig:
    _require(isinstance(payload, dict), "Controller config payload is missing.")
    signal = payload.get("signal")
    _require(isinstance(signal, dict), "Controller signal config is missing.")
    values = dict(payload)
    values["signal"] = TrainingFreeControllerConfig(**signal)
    for name in ("layer_budgets", "dense_layer_budgets"):
        rows = values.get(name)
        _require(isinstance(rows, list), f"Controller {name} is missing.")
        assert isinstance(rows, list)
        values[name] = tuple(tuple(row) for row in rows)
    try:
        return SameTokenControllerConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("Controller config is invalid.") from error


def validate_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    _require(
        spec.get("experiment_id") == "p4-adaptive-production-adapter-spec-v1",
        "The frozen adaptive production adapter spec is required.",
    )
    cell = spec.get("cell", {})
    _require(
        isinstance(cell, dict)
        and cell.get("scale") in {"s55", "s151"}
        and cell.get("budget") in {"2x", "4x"}
        and cell.get("context") in {8_192, 32_768, 131_072}
        and cell.get("generation") in {128, 512, 2_048}
        and isinstance(cell.get("profile"), str)
        and cell.get("batch") in {1, 4, 8, 16}
        and cell.get("concurrency") in {1, 8, 32},
        "The adaptive production cell is invalid.",
    )
    _require(tuple(spec.get("policies", ())) == POLICIES, "Policy order drifted.")
    _require(
        spec.get("protected_end_positions") == list(PROTECTED_END_POSITIONS),
        "Protected-prefix contract drifted.",
    )
    _require(
        spec.get("warmups") == WARMUPS
        and spec.get("measured_repetitions") == MEASURED_REPETITIONS
        and spec.get("input_seed_base") == INPUT_SEED_BASE
        and spec.get("repetition_seeds")
        == [INPUT_SEED_BASE + index for index in range(WARMUPS + MEASURED_REPETITIONS)],
        "Warmup, repetition, or seed schedule drifted.",
    )
    timeout = spec.get("cell_timeout_seconds")
    base._require(
        type(timeout) in (int, float) and 0.0 < timeout <= base.MAX_CELL_TIMEOUT_SECONDS,
        "Adaptive production timeout contract drifted.",
    )
    checkpoint = Path(spec.get("checkpoint", ""))
    _require(checkpoint.is_file(), "The frozen checkpoint is missing.")
    for dependency in (
        "manifest",
        "p2_audit",
        "p3_adaptive_audit",
        "calibration",
        "memory_match",
    ):
        metadata = spec.get(dependency, {})
        dependency_path = Path(metadata.get("path", ""))
        _require(dependency_path.is_file(), f"Missing {dependency} dependency.")
        _require(
            metadata.get("sha256") == base.sha256(dependency_path),
            f"{dependency} drifted.",
        )
    schedule = spec.get("controller_schedule")
    _require(
        isinstance(schedule, list) and len(schedule) == WARMUPS + MEASURED_REPETITIONS,
        "Controller schedule coverage drifted.",
    )
    previous_global_index = -1
    for index, row in enumerate(schedule):
        _require(
            isinstance(row, dict)
            and row.get("phase_repetition") == index
            and type(row.get("global_schedule_index")) is int
            and row["global_schedule_index"] > previous_global_index,
            "Controller schedule ordering drifted.",
        )
        previous_global_index = row["global_schedule_index"]
        policy_configs = row.get("policies", {})
        _require(
            isinstance(policy_configs, dict) and set(policy_configs) == set(POLICIES),
            "Controller schedule policy set drifted.",
        )
        totals: dict[str, int] = {}
        for policy in POLICIES:
            metadata = policy_configs[policy]
            _require(
                isinstance(metadata, dict) and metadata.get("variant") in {"single", "low", "high"},
                "Controller schedule metadata drifted.",
            )
            config = config_from_payload(metadata.get("config"))
            _require(
                metadata.get("sha256") == config_digest(config),
                "Controller config digest drifted.",
            )
            totals[policy] = sum(blocks for _, blocks in config.layer_budgets)
            _require(
                config.enable_protected_pins is True,
                "Adaptive production policy disabled protected pins.",
            )
        _require(
            len(set(totals.values())) == 1,
            "Paired policies do not have the same global hot-block budget.",
        )
    return spec


def _policy_config(
    spec: dict[str, Any], *, repetition: int, policy: str
) -> tuple[SameTokenControllerConfig, dict[str, str]]:
    metadata = spec["controller_schedule"][repetition]["policies"][policy]
    return config_from_payload(metadata["config"]), {
        "sha256": metadata["sha256"],
        "variant": metadata["variant"],
    }


def _run_policy(
    model: Any,
    *,
    spec: dict[str, Any],
    repetition: int,
    policy: str,
    prompt: torch.Tensor,
    decode: torch.Tensor,
    input_digest: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, str]]:
    config, metadata = _policy_config(spec, repetition=repetition, policy=policy)
    cell = spec["cell"]
    run = base.run_policy(
        model,
        policy=policy,
        prompt_cpu=prompt,
        decode_cpu=decode,
        input_digest=input_digest,
        batch=cell["batch"],
        concurrency=cell["concurrency"],
        device=device,
        controller_config=config,
        protected_end_positions=PROTECTED_END_POSITIONS,
    )
    run["adaptive_controller"]["config_sha256"] = metadata["sha256"]
    return run, metadata


def execute(spec: dict[str, Any], *, executable: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for adaptive production.")
    _require(not base._git("status", "--porcelain"), "Adaptive production requires a clean tree.")
    cell = spec["cell"]
    device = torch.device("cuda:0")
    model = base._load_model(Path(spec["checkpoint"]), device)
    failures: dict[str, dict[str, Any] | None] = {policy: None for policy in POLICIES}
    warmup_failures: list[dict[str, Any]] = []
    warmup_attempted = 0
    warmup_paired = 0
    warmup_runs = {policy: 0 for policy in POLICIES}
    repetitions: list[dict[str, Any]] = []
    for schedule_index in range(WARMUPS + MEASURED_REPETITIONS):
        if all(failures[policy] is not None for policy in POLICIES):
            break
        prompt, decode, input_digest = base.generate_inputs(
            model,
            context=cell["context"],
            generation=cell["generation"],
            batch=cell["batch"],
            concurrency=cell["concurrency"],
            seed=spec["repetition_seeds"][schedule_index],
        )
        warmup = schedule_index < WARMUPS
        phase_index = schedule_index if warmup else schedule_index - WARMUPS
        if warmup:
            warmup_attempted += 1
        order = POLICIES if phase_index % 2 == 0 else tuple(reversed(POLICIES))
        policy_runs: dict[str, dict[str, Any]] = {}
        policy_configs: dict[str, dict[str, str]] = {}
        failures_this_repetition: dict[str, dict[str, Any]] = {}
        for policy in order:
            if failures[policy] is not None:
                continue
            try:
                run, metadata = _run_policy(
                    model,
                    spec=spec,
                    repetition=schedule_index,
                    policy=policy,
                    prompt=prompt,
                    decode=decode,
                    input_digest=input_digest,
                    device=device,
                )
                policy_runs[policy] = run
                policy_configs[policy] = metadata
                if warmup:
                    warmup_runs[policy] += 1
            except Exception as error:
                failure = base.failure_record(
                    error,
                    phase="warmup" if warmup else "measured",
                    policy=policy,
                    repetition=phase_index,
                )
                failures[policy] = failure
                failures_this_repetition[policy] = failure
                if warmup:
                    warmup_failures.append(failure)
        paired = set(policy_runs) == set(POLICIES)
        if warmup and paired:
            warmup_paired += 1
        if not warmup:
            repetitions.append(
                {
                    "repetition": phase_index,
                    "input_seed": spec["repetition_seeds"][schedule_index],
                    "input_digest": input_digest,
                    "execution_order": list(order),
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
        del prompt, decode, policy_runs
    counts = {policy: sum(policy in row["policies"] for row in repetitions) for policy in POLICIES}
    complete = sum(count == MEASURED_REPETITIONS for count in counts.values())
    status = "complete" if complete == 2 else "partial" if complete == 1 else "failed"
    del model
    base._cleanup()
    return {
        "schema_version": 1,
        "experiment_id": "p4-adaptive-production-adapter-cell-v1",
        "cell": cell,
        "status": status,
        "input_seed_base": INPUT_SEED_BASE,
        "warmups": WARMUPS,
        "warmup_accounting_available": True,
        "warmup_repetitions_attempted": warmup_attempted,
        "warmup_paired_repetitions_completed": warmup_paired,
        "warmup_policy_runs_completed": warmup_runs,
        "warmup_failures": warmup_failures,
        "measured_repetitions": MEASURED_REPETITIONS,
        "backend": base.backend_provenance(executable, device),
        "repetitions": repetitions,
        "policy_status": {
            policy: {
                "status": "complete" if counts[policy] == MEASURED_REPETITIONS else "failed",
                "measured_repetitions": counts[policy],
                "failure": failures[policy],
            }
            for policy in POLICIES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen adaptive P4 cell in a simultaneous static GPU batch."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = validate_spec(args.spec)
    base._atomic_json(args.output, execute(spec, executable=Path(__file__).resolve()))


if __name__ == "__main__":
    main()
