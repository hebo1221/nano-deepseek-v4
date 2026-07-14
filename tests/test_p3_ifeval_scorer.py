from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from score_p3_ifeval import (  # noqa: E402
    _records as load_cell_records,
)
from score_p3_ifeval import (  # noqa: E402
    aggregate,
    paired_effect,
    score_arm,
)


class FakeOfficial:
    InputExample = SimpleNamespace

    @staticmethod
    def test_instruction_following_strict(inp, prompt_to_response):
        followed = prompt_to_response[inp.prompt] == "PASS"
        return SimpleNamespace(
            follow_instruction_list=[followed] * len(inp.instruction_id_list),
            follow_all_instructions=followed,
        )

    test_instruction_following_loose = test_instruction_following_strict


def _inputs() -> list[dict[str, object]]:
    return [
        {
            "key": 1,
            "prompt": "one",
            "instruction_id_list": ["a", "b"],
            "kwargs": [{}, {}],
        },
        {
            "key": 2,
            "prompt": "two",
            "instruction_id_list": ["c"],
            "kwargs": [{}],
        },
    ]


def _records(first: str, second: str) -> list[dict[str, object]]:
    return [
        {
            "source_id": 1,
            "example_id": "ifeval:1",
            "status": "generated",
            "failure_type": None,
            "raw_response": first,
        },
        {
            "source_id": 2,
            "example_id": "ifeval:2",
            "status": "generated",
            "failure_type": None,
            "raw_response": second,
        },
    ]


def test_official_ifeval_scoring_and_aggregation_are_failure_conservative() -> None:
    outputs = score_arm(inputs=_inputs(), records=_records("PASS", "FAIL"), official=FakeOfficial)

    assert aggregate(outputs) == {
        "expected_prompts": 2,
        "scored_prompts": 2,
        "scorer_failures": 0,
        "generation_failures": 0,
        "prompt_level_strict_accuracy": 0.5,
        "instruction_level_strict_accuracy": 2 / 3,
        "prompt_level_loose_accuracy": 0.5,
        "instruction_level_loose_accuracy": 2 / 3,
        "instruction_total": 3,
    }
    failed = _records("PASS", "PASS")
    failed[1] = {
        "source_id": 2,
        "example_id": "ifeval:2",
        "status": "failure",
        "failure_type": "oom",
    }
    conservative = aggregate(score_arm(inputs=_inputs(), records=failed, official=FakeOfficial))
    assert conservative["generation_failures"] == 1
    assert conservative["prompt_level_strict_accuracy"] == 0.5


def test_ifeval_paired_effect_preserves_prompt_pairing() -> None:
    native = score_arm(inputs=_inputs(), records=_records("FAIL", "FAIL"), official=FakeOfficial)
    fixed = score_arm(inputs=_inputs(), records=_records("PASS", "FAIL"), official=FakeOfficial)

    result = paired_effect(fixed, native, seed=3, replicates=100)

    assert result["paired_prompts"] == 2
    assert result["mean_difference"] == 0.5
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 0)


def test_ifeval_main_contract_exposes_p5_audit_fields() -> None:
    source = (SCRIPTS / "score_p3_ifeval.py").read_text()
    for field in (
        "required_arms_terminal",
        "input_pairing_verified",
        "official_scoring_accounted",
        "expected_prompts_per_arm",
    ):
        assert f'"{field}"' in source


def test_ifeval_cell_rejects_record_arm_drift(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"source_id":1,"benchmark":"IFEval","arm":"wrong",'
        f'"raw_prompt_sha256":"{"a" * 64}",'
        f'"input_token_ids_sha256":"{"b" * 64}"}}\n'
    )
    digest = hashlib.sha256(records.read_bytes()).hexdigest()
    cell = tmp_path / "cell.json"
    cell.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-generation-arm-cell-v1",
                "benchmark": "IFEval",
                "arm": "native-dense",
                "status": "terminal",
                "source": {"dirty": False},
                "expected_generations": 1,
                "manifest": {"sha256": "c" * 64},
                "asset_inventory": {"sha256": "d" * 64},
                "raw_records": {"path": str(records), "sha256": digest},
            }
        )
    )

    with pytest.raises(ValueError, match="record provenance drifted"):
        load_cell_records(
            cell,
            "native-dense",
            1,
            manifest_digest="c" * 64,
            inventory_digest="d" * 64,
        )
