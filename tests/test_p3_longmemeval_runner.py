from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_longmemeval import (  # noqa: E402
    _existing_records,
    judge_blocked_record,
    judge_response,
    prompt_parts,
)


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
    assert question == ("\n\nCurrent Date: 2025/03/01\nQuestion: What did I say?\nAnswer:")


def test_longmemeval_blocked_judge_retains_generation_and_physical_metrics() -> None:
    record = judge_blocked_record(
        {"example_id": "question-1"},
        response="model answer",
        latency_ms=12.5,
        peak_hbm_bytes=100,
        hot_resident_bytes=50,
        reason="no-key",
        generated_tokens=12,
        generation_stop_reason="eos-or-special-token",
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
        "latency_ms": 0.0,
    }
    assert record["generated_tokens_observed"] == 12
    assert record["stop_reason"] == "eos-or-special-token"


def test_longmemeval_official_judge_retains_request_and_response_provenance() -> None:
    class FakeModule:
        @staticmethod
        def get_anscheck_prompt(*_args: object, **_kwargs: object) -> str:
            return "frozen judge prompt"

        @staticmethod
        def chat_completions_with_backoff(_client: object, **kwargs: object) -> object:
            assert kwargs["model"] == "gpt-4o-2024-08-06"
            return SimpleNamespace(
                id="response-1",
                model="gpt-4o-2024-08-06",
                created=123,
                choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"))],
            )

    score, judge = judge_response(
        module=FakeModule,
        client=object(),
        row={
            "question_type": "multi-session",
            "question": "question",
            "answer": "answer",
            "question_id": "id",
        },
        response="response",
    )

    assert score == 1.0
    assert judge["prompt"] == "frozen judge prompt"
    assert judge["raw_response"] == "Yes"
    assert judge["response_id"] == "response-1"
    assert judge["returned_model"] == "gpt-4o-2024-08-06"
    assert judge["created"] == 123
    assert judge["latency_ms"] >= 0


def test_longmemeval_progress_recovers_empty_crash_window(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    partial = tmp_path / "records.partial.jsonl"
    identity = {"digest": "a" * 64}

    assert _existing_records(progress, partial, identity) == []
    assert progress.is_file() and partial.is_file()
    partial.unlink()
    assert _existing_records(progress, partial, identity) == []
    progress.unlink()
    assert _existing_records(progress, partial, identity) == []


def test_longmemeval_is_sequence_gated_before_model_dataset_or_judge_io(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    p2 = tmp_path / "p2.json"
    p2.write_text(
        json.dumps({"completed_shards": 21, "frozen_design": {"total_expected_shards": 4500}})
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "research/adaptive_v4_memory/scripts/run_p3_longmemeval.py"),
            "--kvpress-root",
            str(tmp_path / "missing-kvpress"),
            "--model-snapshot",
            str(tmp_path / "missing-model"),
            "--p2-matrix",
            str(p2),
            "--causal-gate",
            str(tmp_path / "missing-causal.json"),
            "--judge-mode",
            "openai",
            "--output-root",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "21/4500 shards" in completed.stderr
    assert "OPENAI_API_KEY" not in completed.stderr
    assert not output.exists()
