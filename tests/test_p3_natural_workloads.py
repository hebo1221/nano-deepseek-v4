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
    build_longbench_v2_segments,
    build_longmem_full_history_prompt,
    build_scbench_workload,
    mrcr_bin_index,
    mrcr_generation_reserve,
    mrcr_official_token_count,
    parse_mrcr_messages,
    render_chat,
    render_chat_split_last_user,
    render_chat_split_user_content,
    select_mrcr_primary_rows,
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

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


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
    context, query = build_longbench_v2_segments(row, template)
    assert context == "context"
    assert query == "\nquestion\n(A) alpha (B) beta (C) gamma (D) delta"
    assert context + query == build_longbench_v2_prompt(row, template)


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
    context, query = render_chat_split_last_user(
        FakeTokenizer(),
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "final query"},
        ],
    )
    assert context.endswith("user:")
    assert query == "final query|assistant:"
    context, query = render_chat_split_user_content(FakeTokenizer(), "long context", "final query")
    assert context.endswith("long context")
    assert query == "final query|assistant:"

    row["answer"] = "wrong"
    with pytest.raises(ValueError, match="must start"):
        parse_mrcr_messages(row)


def test_mrcr_official_bins_include_answer_and_preserve_every_primary_cell() -> None:
    encoder = FakeTokenizer()
    boundaries = [[1, 3], [4, 6], [7, 9]]
    rows = []
    for ordinal, words in enumerate((1, 1, 2, 3)):
        prefix = f"p{ordinal}"
        rows.append(
            {
                "prompt": json.dumps([{"role": "user", "content": "x " * words}]),
                "answer": f"{prefix} answer",
                "random_string_to_prepend": prefix,
            }
        )

    assert mrcr_bin_index(3, boundaries) == 0
    assert mrcr_bin_index(4, boundaries) == 1
    messages = parse_mrcr_messages(rows[0])
    assert mrcr_official_token_count(rows[0], messages, encoder) == 3
    selected = select_mrcr_primary_rows(
        rows=rows,
        encoder=encoder,
        boundaries=boundaries,
        primary_bins=2,
        samples_per_bin=2,
    )
    assert [row["official_bin_index"] for row in selected] == [0, 0, 1, 1]
    assert mrcr_generation_reserve(rows[0], encoder) == 34


def test_mrcr_primary_selection_rejects_missing_samples() -> None:
    row = {
        "prompt": json.dumps([{"role": "user", "content": "one"}]),
        "answer": "prefix answer",
        "random_string_to_prepend": "prefix",
    }
    with pytest.raises(ValueError, match="exactly 2 samples"):
        select_mrcr_primary_rows(
            rows=[row],
            encoder=FakeTokenizer(),
            boundaries=[[1, 10]],
            primary_bins=1,
            samples_per_bin=2,
        )


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
