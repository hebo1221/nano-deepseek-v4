from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import calibrate_p2_causal_hot_memory as hot_match  # noqa: E402
import evaluate_p2_causal_factorial_shard as shard  # noqa: E402
import run_p2_causal_factorial_matrix as matrix  # noqa: E402
import run_p2_causal_prerequisites as prerequisites  # noqa: E402
import summarize_p2_causal_factorial as summary  # noqa: E402
from freeze_p2_causal_factorial_arms import build_arm_configs  # noqa: E402

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES  # noqa: E402


def _calibration(budgets: list[list[int]], *, global_budget: int) -> dict[str, Any]:
    return {
        "experiment_id": "p1-layer-quota-calibration-pilot-v1",
        "calibrations": {
            "2x": {
                "signal_config": {
                    "global_block_budget": global_budget,
                    "dense_fallback_block_budget": global_budget * 2,
                },
                "quota": {
                    "layer_budgets": budgets,
                    "calibration_digest": "2" * 64,
                },
            }
        },
    }


def test_causal_matrix_size_matches_frozen_cartesian_product() -> None:
    root = Path(__file__).resolve().parents[1]
    design = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json").read_text()
    )

    expected = (
        2
        * len(shard.TRAINING_SEEDS)
        * len(shard.BUDGET_LABELS)
        * len(PAPER_GRADE_WORKLOAD_FAMILIES)
        * len(shard.CONTEXTS)
        * len(shard.REPLICATES)
    )
    assert expected == matrix.EXPECTED_CAUSAL_SHARDS == 9_000
    assert design["execution"]["total_expected_shards"] == expected
    assert design["execution"]["total_quality_arm_conversations"] == (
        expected * shard.EXAMPLES_PER_SHARD * len(shard.ALL_ARM_NAMES)
    )
    assert len(shard.ALL_ARM_NAMES) == 16
    assert design["execution"]["total_physical_arm_conversations"] == (
        expected * shard.EXAMPLES_PER_SHARD * len(shard.PHYSICAL_ARM_NAMES)
    )
    equivalence = design["execution"]["equivalence_validation"]
    assert shard.EXPECTED_EQUIVALENCE_RECORDS == 5_760
    assert equivalence["required_exact_records_per_seed_scale"] == 5_760
    assert equivalence["required_exact_records_total"] == 57_600
    assert shard.MEMORY_MATCH_MIXTURE_DENOMINATOR == 2_250


def test_prerequisite_paths_match_the_causal_matrix_contract(tmp_path: Path) -> None:
    paths = prerequisites._cell_paths(
        scale="s151",
        training_seed=6_071_405,
        training_root=tmp_path / "training",
        calibration_root=tmp_path / "calibration",
        memory_match_root=tmp_path / "memory",
        equivalence_root=tmp_path / "equivalence",
    )

    assert paths["checkpoint"].as_posix().endswith("training/s151/seed-6071405/s151-step-1000.pt")
    assert (
        paths["memory_summary"]
        .as_posix()
        .endswith("memory/s151/seed-6071405/p2-causal-hot-memory-match.summary.json")
    )
    assert (
        paths["equivalence_summary"]
        .as_posix()
        .endswith("equivalence/s151/seed-6071405.summary.json")
    )


def test_schedule_indices_cover_each_seed_budget_cartesian_matrix_once() -> None:
    indices = [
        shard.schedule_batch_index(
            family=family,
            context=context,
            replicate=replicate,
            local_batch_index=batch,
        )
        for family, context, replicate, batch in product(
            PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
            shard.REPLICATES,
            range(shard.BATCHES_PER_SHARD),
        )
    ]

    assert sorted(indices) == list(range(len(indices)))
    assert len(indices) == 2_250


def test_fixed_bresenham_schedule_exactly_matches_calibrated_mean() -> None:
    arms, metadata = build_arm_configs(
        _calibration([[2, 1], [4, 2], [6, 2]], global_budget=6), "2x"
    )
    fixed = arms["fixed+pins"]
    totals = [
        sum(value for _, value in fixed.config_for_batch(index).layer_budgets)
        for index in range(2_250)
    ]

    assert totals.count(6) == 1_500
    assert totals.count(3) == 750
    assert sum(totals) / len(totals) == 5.0
    assert metadata["calibrated_total_blocks"] == 5


def test_exact_config_reuse_requires_digest_and_dataclass_identity() -> None:
    arms, _metadata = build_arm_configs(
        _calibration([[2, 2], [4, 2], [6, 2]], global_budget=6), "2x"
    )
    fixed = arms["fixed"].configs[0]
    calibrated = arms["calibrated-no-pins"].configs[0]
    pinned = arms["fixed+pins"].configs[0]
    cache = {shard.config_digest(fixed): ("fixed", fixed, {"predictions": [[1]]})}

    assert fixed == calibrated
    assert shard.exact_config_reuse(cache, calibrated) == (
        "fixed",
        {"predictions": [[1]]},
    )
    assert fixed != pinned
    assert shard.exact_config_reuse(cache, pinned) is None


def test_causal_manifest_discloses_exact_config_reuse_before_execution() -> None:
    root = Path(__file__).resolve().parents[1]
    design = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json").read_text()
    )

    assert design["status"] == "amended_and_frozen_before_execution"
    assert design["protocol_amendments"][0]["timing"].startswith("before any")
    assert design["execution"]["exact_config_reuse"]["scope"] == ("within one paired batch only")


def test_exact_config_reuse_audit_counts_forwards_without_double_counting() -> None:
    rows = [
        {
            "schedule_batch_index": 3,
            "arm": "fixed",
            "execution_index": 0,
            "execution_mode": "executed",
            "reused_from_arm": None,
            "config_sha256": "a" * 64,
            "wall_ms": 1.5,
        },
        {
            "schedule_batch_index": 3,
            "arm": "calibrated",
            "execution_index": 1,
            "execution_mode": "reused-exact-config",
            "reused_from_arm": "fixed",
            "config_sha256": "a" * 64,
            "wall_ms": 0.0,
        },
    ]

    assert summary.validate_exact_config_reuse(rows, expected_arms=("fixed", "calibrated")) == {
        "executed": 1,
        "reused_exact_config": 1,
    }

    rows[1]["config_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="reuse source drifted"):
        summary.validate_exact_config_reuse(rows, expected_arms=("fixed", "calibrated"))


def test_causal_shard_executes_each_exact_config_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration([[2, 2], [4, 2], [6, 2]], global_budget=6)
    memory_match = {
        "matches": {
            "2x": {
                "calibration_digest": "2" * 64,
                "uniform_low_blocks_per_layer": 2,
                "uniform_high_blocks_per_layer": 2,
                "mixture_high_numerator": 0,
                "mixture_denominator": 2_250,
                "passed": True,
            }
        }
    }
    generated = 0
    quality_calls: list[str] = []
    physical_calls: list[str] = []

    def fake_workload(*_args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal generated
        batch_size = int(kwargs["batch_size"])
        offset = int(kwargs["conversation_offset"])
        generated += batch_size
        return SimpleNamespace(
            family="single-remote-retrieval",
            targets=torch.ones((batch_size, 1), dtype=torch.long),
            query_positions=torch.zeros((batch_size, 1), dtype=torch.long),
            evidence_positions=torch.zeros((batch_size, 1), dtype=torch.long),
            conversation_ids=[f"conversation-{offset + row}" for row in range(batch_size)],
            protected_end_positions=(),
        )

    def fake_quality(
        _model: object,
        workload: SimpleNamespace,
        *,
        arm_name: str,
        config: object,
        chunk_size: int,
    ) -> dict[str, Any]:
        del config, chunk_size
        quality_calls.append(arm_name)
        batch_size = len(workload.conversation_ids)
        return {
            "predictions": [[1] for _ in range(batch_size)],
            "correct": [[True] for _ in range(batch_size)],
            "wall_ms": 1.0,
            "controller": {},
            "controller_rows": [{"budget_violations": 0} for _ in range(batch_size)],
        }

    def fake_physical(
        _model: object,
        workload: SimpleNamespace,
        *,
        arm_name: str,
        config: object,
    ) -> dict[str, Any]:
        del config
        physical_calls.append(arm_name)
        batch_size = len(workload.conversation_ids)
        return {
            "predictions": [[1] for _ in range(batch_size)],
            "correct": [[True] for _ in range(batch_size)],
            "wall_ms": 2.0,
            "accounting": {"hot_resident_bytes": 64},
            "tier": {},
            "controller_rows": [{} for _ in range(batch_size)],
            "physical_hot_budget_blocks_by_layer": {},
        }

    monkeypatch.setattr(shard, "generate_adaptive_memory_workload", fake_workload)
    monkeypatch.setattr(shard, "run_chunked_quality", fake_quality)
    monkeypatch.setattr(shard, "run_sequential_physical", fake_physical)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=2_048, sliding_window=64))

    records, metrics, physical, _seed, _metadata = shard.evaluate_shard(
        model,
        calibration=calibration,
        memory_match=memory_match,
        scale="s55",
        budget_label="2x",
        family="single-remote-retrieval",
        context=80,
        replicate=0,
        training_seed=6_071_401,
        batch_size=4,
    )

    assert generated == 20
    assert len(records) == 20 * len(shard.ALL_ARM_NAMES)
    assert len(metrics) == 5 * len(shard.ALL_ARM_NAMES)
    assert len(quality_calls) == 5 * 12
    assert sum(row["execution_mode"] == "reused-exact-config" for row in metrics) == 20
    assert len(physical) == 5 * len(shard.PHYSICAL_ARM_NAMES)
    assert len(physical_calls) == 15
    assert sum(row["execution_mode"] == "reused-exact-config" for row in physical) == 5


def test_registered_arm_oracle_is_target_aware_and_not_the_fixed_score() -> None:
    rows = {
        (arm, "conversation"): {
            "total": 2,
            "correct_count": 1 if arm == shard.PRIMARY_ARM_NAMES[1] else 0,
        }
        for arm in shard.ALL_ARM_NAMES
    }
    rows[("fixed-top-p-0.8", "conversation")]["correct_count"] = 2

    oracle, fixed = summary.registered_arm_oracle_scores(rows, "conversation")

    assert oracle == 1.0
    assert fixed == 0.5


def test_physical_memory_match_overrides_fixed_schedule_without_changing_calibrated() -> None:
    calibration = _calibration([[2, 1], [4, 2], [6, 2]], global_budget=6)
    match = {
        "matches": {
            "2x": {
                "calibration_digest": "2" * 64,
                "uniform_low_blocks_per_layer": 2,
                "uniform_high_blocks_per_layer": 3,
                "mixture_high_numerator": 7,
                "mixture_denominator": 10,
                "passed": True,
            }
        }
    }

    arms, metadata = build_arm_configs(calibration, "2x", fixed_match=match)

    fixed = arms["fixed+pins"]
    calibrated = arms["calibrated+pins"]
    assert fixed.mixture_high_numerator == 7
    assert fixed.mixture_denominator == 10
    assert {sum(value for _, value in config.layer_budgets) for config in fixed.configs} == {
        6,
        9,
    }
    assert sum(value for _, value in calibrated.configs[0].layer_budgets) == 5
    assert metadata["fixed_match_source"] == "calibration-physical-hot-bytes"


def test_supplemental_fixed_top_p_arms_freeze_threshold_and_uniform_budget() -> None:
    arms, _metadata = build_arm_configs(
        _calibration([[2, 1], [4, 2], [6, 2]], global_budget=6), "2x"
    )

    for threshold in (0.5, 0.8):
        arm = arms[f"fixed-top-p-{threshold:.1f}"]
        assert all(config.signal.top_p == threshold for config in arm.configs)
        assert all(config.signal.max_extra_blocks_per_layer == 0 for config in arm.configs)
        assert all(config.enable_score_concentration for config in arm.configs)
        assert all(not config.enable_temporal_reuse for config in arm.configs)
        assert all(not config.enable_cross_layer_signal for config in arm.configs)
        assert all(not config.enable_protected_pins for config in arm.configs)


def test_physical_memory_match_rejects_calibration_drift() -> None:
    match = {
        "matches": {
            "2x": {
                "calibration_digest": "3" * 64,
                "uniform_low_blocks_per_layer": 1,
                "uniform_high_blocks_per_layer": 2,
                "mixture_high_numerator": 1,
                "mixture_denominator": 2,
                "passed": True,
            }
        }
    }

    with pytest.raises(ValueError, match="calibration digest drifted"):
        build_arm_configs(
            _calibration([[2, 1], [4, 2], [6, 2]], global_budget=6),
            "2x",
            fixed_match=match,
        )


def test_hot_memory_calibration_selects_a_frozen_interpolated_schedule() -> None:
    match = hot_match._choose_match(
        calibration_digest="2" * 64,
        target_values=[150] * 225,
        candidate_values={2: [100] * 225, 3: [200] * 225},
    )

    assert match["uniform_low_blocks_per_layer"] == 2
    assert match["uniform_high_blocks_per_layer"] == 3
    assert match["mixture_high_numerator"] == 1_125
    assert match["mixture_denominator"] == 2_250
    assert match["relative_difference"] == 0.0
    assert match["passed"] is True


def test_hot_memory_calibration_withholds_an_unbracketed_match() -> None:
    match = hot_match._choose_match(
        calibration_digest="2" * 64,
        target_values=[300] * 225,
        candidate_values={2: [100] * 225, 3: [200] * 225},
    )

    assert match["target_bracketed"] is False
    assert match["passed"] is False


def test_causal_runner_refuses_an_incomplete_p2_matrix(tmp_path: Path) -> None:
    p2_matrix = tmp_path / "p2.json"
    p2_matrix.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-progress-v1",
                "completed_shards": 122,
                "frozen_design": {"total_expected_shards": 4500},
                "runs": [],
            }
        )
    )

    with pytest.raises(RuntimeError, match="122/4500 recorded"):
        matrix.require_p2_audit(p2_matrix, tmp_path / "missing-audit.json")


def test_causal_equivalence_gate_rejects_unverified_payload(tmp_path: Path) -> None:
    fake = tmp_path / "equivalence.json"
    fake.write_text(
        json.dumps(
            {
                "experiment_id": "p2-causal-factorial-equivalence-audit-v1",
                "scale": "s55",
                "validation": {
                    "budgets": list(shard.BUDGET_LABELS),
                    "arms": list(shard.ALL_ARM_NAMES),
                    "chunk_size": shard.CHUNK_SIZE_BY_SCALE["s55"],
                    "all_predictions_identical": False,
                    "all_budget_checks_passed": True,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="equivalence is missing"):
        shard._equivalence(fake, "s55")
