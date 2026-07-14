from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import calibrate_p2_causal_hot_memory as hot_match  # noqa: E402
import evaluate_p2_causal_factorial_shard as shard  # noqa: E402
import run_p2_causal_factorial_matrix as matrix  # noqa: E402
import run_p2_causal_prerequisites as prerequisites  # noqa: E402
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
    assert design["execution"]["total_physical_arm_conversations"] == (
        expected * shard.EXAMPLES_PER_SHARD * len(shard.PHYSICAL_ARM_NAMES)
    )
    equivalence = design["execution"]["equivalence_validation"]
    assert shard.EXPECTED_EQUIVALENCE_RECORDS == 5_040
    assert equivalence["required_exact_records_per_seed_scale"] == 5_040
    assert equivalence["required_exact_records_total"] == 50_400
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

    assert paths["checkpoint"].as_posix().endswith(
        "training/s151/seed-6071405/s151-step-1000.pt"
    )
    assert paths["memory_summary"].as_posix().endswith(
        "memory/s151/seed-6071405/p2-causal-hot-memory-match.summary.json"
    )
    assert paths["equivalence_summary"].as_posix().endswith(
        "equivalence/s151/seed-6071405.summary.json"
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
