from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_longmemeval import judge_blocked_record, prompt_parts  # noqa: E402


def test_longmemeval_prompt_split_preserves_full_history_and_query() -> None:
    row = {
        "haystack_dates": ["2025/01/01", "2025/02/01"],
        "haystack_sessions": [
            [{"role": "user", "content": "first"}],
            [{"role": "assistant", "content": "second"}],
        ],
        "question_date": "2025/03/01",
        "question": "What did I say?",
    }

    context, question = prompt_parts(row)

    assert "Session Date: 2025/01/01" in context
    assert "Session Date: 2025/02/01" in context
    assert question == (
        "\n\nCurrent Date: 2025/03/01\nQuestion: What did I say?\nAnswer:"
    )


def test_longmemeval_blocked_judge_retains_generation_and_physical_metrics() -> None:
    record = judge_blocked_record(
        {"example_id": "question-1"},
        response="model answer",
        latency_ms=12.5,
        peak_hbm_bytes=100,
        hot_resident_bytes=50,
        reason="no-key",
    )

    assert record["status"] == "failure"
    assert record["failure_type"] == "judge-blocked"
    assert record["raw_response"] == "model answer"
    assert record["score"] is None
    assert record["hot_resident_bytes"] == 50
    assert record["judge"] == {
        "model": "gpt-4o-2024-08-06",
        "status": "blocked",
        "reason": "no-key",
    }
