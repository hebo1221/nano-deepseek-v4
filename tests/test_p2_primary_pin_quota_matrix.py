from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
matrix = importlib.import_module("run_p2_primary_pin_quota_matrix")


def test_primary_runner_frozen_grid() -> None:
    assert matrix.ARMS == ("fixed", "fixed+pins", "calibrated-no-pins", "calibrated+pins")
    assert len(matrix.coordinates()) == matrix.EXPECTED_CELLS == 9000
    gb10 = matrix.coordinates("gb10")
    rtx4090 = matrix.coordinates("rtx4090")
    assert len(gb10) == 3600
    assert len(rtx4090) == 5400
    assert {matrix._coordinate_key(coordinate) for coordinate in gb10}.isdisjoint(
        {matrix._coordinate_key(coordinate) for coordinate in rtx4090}
    )
    assert (
        len({matrix._coordinate_key(coordinate) for coordinate in (*gb10, *rtx4090)})
        == matrix.EXPECTED_CELLS
    )
    prefix = {
        matrix._coordinate_key(coordinate)
        for coordinate in matrix.coordinates()[: matrix.LEGACY_PREFIX_CELLS]
    }
    assert sum(matrix._coordinate_key(coordinate) in prefix for coordinate in gb10) == 796
    assert sum(matrix._coordinate_key(coordinate) in prefix for coordinate in rtx4090) == 1190


def test_implementation_inventory_is_repository_relative() -> None:
    inventory = matrix._implementation_inventory()
    assert inventory
    assert all(not Path(path).is_absolute() for path, _digest in inventory)
    assert all(len(digest) == 64 for _path, digest in inventory)
    assert "nano_deepseek_v4/hierarchical_memory_controller.py" in {
        path for path, _digest in inventory
    }


def test_frozen_cohort_paths_rebase_only_within_repository() -> None:
    relative = Path("artifacts/example.json")
    frozen = matrix.LEGACY_REPOSITORY_ROOT / relative
    assert matrix._materialized_frozen_path(frozen) == Path.cwd().resolve() / relative
    assert matrix._materialized_frozen_path(relative) == Path.cwd().resolve() / relative
    with pytest.raises(ValueError):
        matrix._materialized_frozen_path(Path("/tmp/outside-repository.json"))


def _sealed_cell() -> tuple[dict[str, object], dict[str, object]]:
    coordinate = matrix.coordinates()[0]
    scale = coordinate["scale"]
    budget = coordinate["budget"]
    family = coordinate["family"]
    context = coordinate["context"]
    replicate = coordinate["replicate"]
    calibration_seed, evaluation_seed = matrix.evaluator._validate_coordinate(
        scale=scale,
        training_seed=coordinate["training_seed"],
        budget=budget,
        family=family,
        context=context,
        replicate=replicate,
    )
    examples = []
    outcomes = []
    token_summary = {
        "hot_blocks_sequence_sha256": "blocks",
        "hot_bytes_sequence_sha256": "bytes",
    }
    for example_index in range(matrix.contract.EXAMPLES_PER_SHARD):
        schedule = matrix.evaluator.shard_schedule_index(
            family=family,
            context=context,
            replicate=replicate,
            example_index=example_index,
        )
        order = tuple(
            name for name in matrix.evaluator.arm_execution_order(schedule) if name in matrix.ARMS
        )
        examples.append({"example_index": example_index, "execution_order": list(order)})
        outcomes.extend(
            {
                "example_index": example_index,
                "arm": arm,
                "execution_index": execution_index,
                "token_summary": token_summary,
            }
            for execution_index, arm in enumerate(order)
        )
    body = {
        "schema_version": 1,
        "experiment_id": matrix.EXPERIMENT_ID,
        "status": "terminal",
        "protocol_manifest": {
            "path": str(matrix.PROTOCOL_MANIFEST_PATH),
            "sha256": "a" * 64,
            "bytes": 1,
            "experiment_id": matrix.EXPERIMENT_ID,
        },
        "coordinate": {
            **coordinate,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "generation_seed": matrix.contract.generation_seed(
                evaluation_seed, family, context, replicate
            ),
            "global_block_budget": matrix.contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
        },
        "arms": list(matrix.ARMS),
        "examples": examples,
        "outcomes": outcomes,
    }
    return {**body, "payload_sha256": matrix._digest(body)}, coordinate


@pytest.mark.parametrize("corruption", ["outcome_example", "execution_order", "derived_seed"])
def test_primary_runner_rejects_resealed_inventory_corruption(corruption: str) -> None:
    payload, coordinate = _sealed_cell()
    corrupted = deepcopy(payload)
    if corruption == "outcome_example":
        corrupted["outcomes"][0]["example_index"] = 100
    elif corruption == "execution_order":
        corrupted["examples"][0]["execution_order"].reverse()
    else:
        corrupted["coordinate"]["generation_seed"] += 1
    body = dict(corrupted)
    body.pop("payload_sha256")
    corrupted["payload_sha256"] = matrix._digest(body)

    with pytest.raises(ValueError):
        matrix.validate_cell(corrupted, coordinate)


def test_runtime_binding_match_is_exact() -> None:
    payload = {
        "source": {"commit": "a" * 40, "dirty": False},
        "implementation_digest": "b" * 64,
        "protocol_manifest": {"sha256": "c" * 64},
        "cohort_binding": {"input_binding_digest": "d" * 64},
    }
    arguments = {
        "source": payload["source"],
        "implementation_digest": payload["implementation_digest"],
        "protocol_binding": payload["protocol_manifest"],
        "cohort_binding": payload["cohort_binding"],
    }

    assert matrix._runtime_binding_matches(payload, **arguments)
    assert not matrix._runtime_binding_matches(
        payload, **{**arguments, "implementation_digest": "e" * 64}
    )
