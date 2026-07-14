from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p3_ruler_matrix import paired_cell_statistics  # noqa: E402


def _frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(len(scores)),
            "task": ["task-a" if index % 2 == 0 else "task-b" for index in range(len(scores))],
            "string_match": scores,
        }
    )


def test_ruler_screen_covers_token_layer_and_head_adaptive_baselines() -> None:
    import run_p3_ruler_matrix as runner
    import summarize_p3_ruler_matrix as summary

    assert runner.ARMS["pyramidkv"][0] == "pyramidkv"
    assert runner.ARMS["adakv_snapkv"][0] == "adakv_snapkv"
    assert len(runner.cells(runner.LENGTHS, tuple(runner.ARMS), None)) == 57
    assert summary.EXPECTED_CELLS == 57


def test_ruler_loader_imports_press_code_from_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "kvpress-checkout"
    package = checkout / "kvpress"
    evaluation = checkout / "evaluation"
    package.mkdir(parents=True)
    evaluation.mkdir()
    (package / "__init__.py").write_text("PINNED_TEST_PACKAGE = True\n")
    (evaluation / "evaluate.py").write_text(
        "class EvaluationConfig:\n    pass\n\nclass EvaluationRunner:\n    pass\n"
    )
    required_presses = {
        "no_press",
        "streaming_llm",
        "snapkv",
        "pyramidkv",
        "adakv_snapkv",
        "expected_attention",
        "critical_expected_attention",
    }
    (evaluation / "evaluate_registry.py").write_text(
        f"PRESS_REGISTRY = {dict.fromkeys(sorted(required_presses))!r}\n"
        "SCORER_REGISTRY = {'ruler': object()}\n"
    )
    code = f"""
import json
import sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import run_p3_ruler_matrix as runner
runner.load_evaluator(Path({str(checkout)!r}))
print(json.dumps(runner.kvpress_runtime_binding(), sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    binding = json.loads(result.stdout.splitlines()[-1])

    assert Path(binding["module_path"]).is_relative_to(checkout)
    assert Path(binding["registry_path"]).is_relative_to(checkout)
    assert len(binding["module_sha256"]) == 64
    assert len(binding["registry_sha256"]) == 64
    import p3_source_provenance as provenance
    import summarize_p3_ruler_matrix as summary

    monkeypatch.setattr(provenance, "_verify_kvpress_checkout", lambda _root: checkout)
    assert summary.verify_runtime_kvpress_binding(binding)["module_sha256"] == binding[
        "module_sha256"
    ]
    with pytest.raises(ValueError, match="import binding drifted"):
        summary.verify_runtime_kvpress_binding(
            {**binding, "module_sha256": "0" * 64}
        )


def test_paired_ruler_statistics_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import summarize_p3_ruler_matrix as summary

    monkeypatch.setattr(summary, "EXPECTED_ROWS_PER_CELL", 20)
    native = _frame([0.0] * 20)
    candidate = _frame([1.0] * 20)

    first = paired_cell_statistics(candidate, native, label="paired", resamples=1_000)
    second = paired_cell_statistics(candidate, native, label="paired", resamples=1_000)

    assert first == second
    assert first["paired_rows"] == 20
    assert first["overall"]["mean_difference"] == pytest.approx(1.0)
    assert first["overall"]["paired_cluster_bootstrap_95_ci"] == pytest.approx([1.0, 1.0])
    assert len(first["by_task_with_holm_bonferroni"]) == 2


def test_paired_ruler_statistics_reject_task_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    import summarize_p3_ruler_matrix as summary

    monkeypatch.setattr(summary, "EXPECTED_ROWS_PER_CELL", 4)
    native = _frame([0.0] * 4)
    candidate = _frame([0.0] * 4)
    candidate.loc[0, "task"] = "drifted"

    with pytest.raises(ValueError, match="tasks drifted"):
        paired_cell_statistics(candidate, native, label="drift", resamples=100)
