from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from nano_deepseek_v4 import SameTokenControllerConfig, TrainingFreeControllerConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p4_adaptive_continuous_batch_adapter as adapter  # noqa: E402


def _config(*, blocks: int = 2) -> SameTokenControllerConfig:
    signal = TrainingFreeControllerConfig(
        global_block_budget=blocks,
        dense_fallback_block_budget=blocks,
    )
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=((1, blocks),),
        dense_layer_budgets=((1, blocks),),
    )


def _spec(tmp_path: Path) -> dict[str, object]:
    dependencies: dict[str, dict[str, str]] = {}
    for name in (
        "manifest",
        "p2_audit",
        "p3_adaptive_audit",
        "calibration",
        "memory_match",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        dependencies[name] = {"path": str(path), "sha256": adapter.base.sha256(path)}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint")
    config = _config()
    config_metadata = {
        "config": asdict(config),
        "sha256": adapter.config_digest(config),
        "variant": "single",
    }
    return {
        "schema_version": 1,
        "experiment_id": "p4-adaptive-production-adapter-spec-v1",
        "cell": {
            "scale": "s55",
            "budget": "2x",
            "context": 8192,
            "generation": 128,
            "profile": "serving-b1-c8",
            "batch": 1,
            "concurrency": 8,
        },
        "policies": list(adapter.POLICIES),
        "protected_end_positions": list(adapter.PROTECTED_END_POSITIONS),
        "warmups": adapter.WARMUPS,
        "measured_repetitions": adapter.MEASURED_REPETITIONS,
        "input_seed_base": adapter.INPUT_SEED_BASE,
        "cell_timeout_seconds": adapter.base.MAX_CELL_TIMEOUT_SECONDS,
        "repetition_seeds": [
            adapter.INPUT_SEED_BASE + index
            for index in range(adapter.WARMUPS + adapter.MEASURED_REPETITIONS)
        ],
        "checkpoint": str(checkpoint),
        **dependencies,
        "controller_schedule": [
            {
                "phase_repetition": index,
                "global_schedule_index": 100 + index,
                "policies": {
                    policy: deepcopy(config_metadata) for policy in adapter.POLICIES
                },
            }
            for index in range(adapter.WARMUPS + adapter.MEASURED_REPETITIONS)
        ],
    }


def test_controller_config_json_round_trip_is_digest_stable() -> None:
    config = _config()
    payload = json.loads(json.dumps(asdict(config)))

    restored = adapter.config_from_payload(payload)

    assert restored == config
    assert adapter.config_digest(restored) == adapter.config_digest(config)


def test_adaptive_production_adapter_spec_binds_full_controller_schedule(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))

    validated = adapter.validate_spec(path)
    assert validated["cell"] == spec["cell"]
    assert len(validated["controller_schedule"]) == 35

    spec["controller_schedule"][7]["global_schedule_index"] = 1
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="schedule ordering drifted"):
        adapter.validate_spec(path)


def test_adaptive_production_adapter_rejects_unmatched_physical_budget(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    different = _config(blocks=3)
    metadata = spec["controller_schedule"][0]["policies"]["calibrated+pins"]
    metadata["config"] = asdict(different)
    metadata["sha256"] = adapter.config_digest(different)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))

    with pytest.raises(ValueError, match="same global hot-block budget"):
        adapter.validate_spec(path)
