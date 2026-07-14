from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from score_p3_ifeval import aggregate, paired_effect, score_arm  # noqa: E402


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
