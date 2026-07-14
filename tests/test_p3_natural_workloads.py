from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_natural_workloads import (  # noqa: E402
    build_longbench_v2_prompt,
    build_longmem_full_history_prompt,
    build_scbench_workload,
    parse_mrcr_messages,
    render_chat,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return "|".join(f"{row['role']}:{row['content']}" for row in conversation) + "|assistant:"


def test_longbench_prompt_matches_pinned_direct_template_substitution() -> None:
    template = "$DOC$\n$Q$\n(A) $C_A$ (B) $C_B$ (C) $C_C$ (D) $C_D$"
    row = {
        "context": " context ",
        "question": " question ",
        "choice_A": " alpha ",
        "choice_B": " beta ",
        "choice_C": " gamma ",
        "choice_D": " delta ",
    }

    assert build_longbench_v2_prompt(row, template) == (
        "context\nquestion\n(A) alpha (B) beta (C) gamma (D) delta"
    )


def test_longmem_prompt_uses_all_sessions_in_date_order_without_mutation() -> None:
    row = {
        "haystack_dates": ["2025/02/02", "2025/01/01"],
        "haystack_sessions": [
            [{"role": "user", "content": " later ", "has_answer": True}],
            [
                {"role": "user", "content": " earlier "},
                {"role": "assistant", "content": " response "},
            ],
        ],
        "question_date": "2025/03/03",
        "question": "What happened?",
    }

    prompt = build_longmem_full_history_prompt(row)

    assert prompt.index("2025/01/01") < prompt.index("2025/02/02")
    assert "\n\nuser: earlier\n\nassistant: response" in prompt
    assert prompt.endswith("Current Date: 2025/03/03\nQuestion: What happened?\nAnswer:")
    assert row["haystack_sessions"][0][0]["has_answer"] is True


def test_mrcr_messages_and_prefix_are_fail_closed() -> None:
    row = {
        "prompt": json.dumps(
            [{"role": "user", "content": "ask"}, {"role": "assistant", "content": "reply"}]
        ),
        "answer": "aB12answer",
        "random_string_to_prepend": "aB12",
    }
    messages = parse_mrcr_messages(row)
    assert render_chat(FakeTokenizer(), messages).endswith("assistant:")

    row["answer"] = "wrong"
    with pytest.raises(ValueError, match="must start"):
        parse_mrcr_messages(row)


def test_scbench_uses_official_builders_and_preserves_shared_context_semantics() -> None:
    calls: list[str] = []

    def multiturn(row, task, tokenizer, use_chat_template, disable_golden_context):
        calls.append(f"multi:{task}:{use_chat_template}:{disable_golden_context}")
        return {"prompts": ["full-0", "follow-1"], "ground_truth": ["a", "b"]}

    def scdq(row, task, tokenizer, use_chat_template):
        calls.append(f"scdq:{task}:{use_chat_template}")
        return {"prompts": ["shared", "query-0", "query-1"], "ground_truth": ["a", "b"]}

    module = SimpleNamespace(
        create_multiturn_prompt=multiturn,
        create_scdq_prompt=scdq,
        DATA_NAME_TO_MAX_NEW_TOKENS={"scbench_kv": 150},
    )
    row = {"multi_turns": [{"answer": "a"}, {"answer": "b"}]}

    multi = build_scbench_workload(
        row=row,
        task="scbench_kv",
        mode="multi-turn",
        tokenizer=FakeTokenizer(),
        official_module=module,
    )
    scdq_result = build_scbench_workload(
        row=row,
        task="scbench_kv",
        mode="multi-request",
        tokenizer=FakeTokenizer(),
        official_module=module,
    )

    assert multi["shared_context"] is None
    assert [turn["prompt_segment"] for turn in multi["turns"]] == ["full-0", "follow-1"]
    assert scdq_result["shared_context"] == "shared"
    assert [turn["prompt_segment"] for turn in scdq_result["turns"]] == [
        "query-0",
        "query-1",
    ]
    assert all(turn["generation_reserve_tokens"] == 150 for turn in multi["turns"])
    assert calls == ["multi:scbench_kv:True:False", "scdq:scbench_kv:True"]
