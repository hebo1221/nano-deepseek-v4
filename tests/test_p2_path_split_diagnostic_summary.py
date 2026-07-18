from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import diagnose_p2_causal_equivalence as diagnostic  # noqa: E402
import summarize_p2_path_split_diagnostic as summary  # noqa: E402


def _historical_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, cell in enumerate(diagnostic.KNOWN_FAILURE_CELLS):
        conversation_id = f"{diagnostic.TARGET_FAMILY}:{cell.context}:0"
        chunked = [index, index + 1, index + 2, index + 3]
        physical = [index, index + 1, index + 2, index + 4]
        cells.append(
            {
                "budget": cell.budget,
                "context": cell.context,
                "arm": cell.arm,
                "workload": {"conversation_ids": [conversation_id]},
                "runs": {
                    "resident-chunk2": [
                        {"status": "success", "predictions": [list(chunked)]}
                        for _repeat in range(diagnostic.FROZEN_REPEATS)
                    ],
                    "tiered-tokenwise": [
                        {"status": "success", "predictions": [list(physical)]}
                        for _repeat in range(diagnostic.FROZEN_REPEATS)
                    ],
                },
            }
        )
        records.append(
            {
                "budget": cell.budget,
                "context": cell.context,
                "arm": cell.arm,
                "conversation_id": conversation_id,
                "chunked_predictions": list(chunked),
                "physical_predictions": list(physical),
                "predictions_identical": False,
            }
        )
    return cells, {"records": records}


def test_historical_reproduction_requires_all_seven_exact_vectors() -> None:
    cells, blocker = _historical_fixture()

    assert summary._validate_historical_reproduction(cells, blocker) == 7

    cells[0]["runs"]["resident-chunk2"][2]["predictions"][0][-1] += 1
    with pytest.raises(ValueError, match="prediction vectors"):
        summary._validate_historical_reproduction(cells, blocker)


def test_comparison_rows_preserves_cell_and_repeat_coverage() -> None:
    cells = [
        {
            "comparisons": {
                "paired_path_observations": {
                    "A": [{"cell": cell, "repeat": repeat} for repeat in range(3)]
                }
            }
        }
        for cell in range(2)
    ]

    assert summary._comparison_rows(cells, "A") == [
        {"cell": 0, "repeat": 0},
        {"cell": 0, "repeat": 1},
        {"cell": 0, "repeat": 2},
        {"cell": 1, "repeat": 0},
        {"cell": 1, "repeat": 1},
        {"cell": 1, "repeat": 2},
    ]


def test_summary_publish_is_atomic_and_no_replace(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    summary._write_json_exclusive(output, {"status": "first"})
    assert json.loads(output.read_text()) == {"status": "first"}

    with pytest.raises(FileExistsError, match="already exists"):
        summary._write_json_exclusive(output, {"status": "replacement"})
    assert json.loads(output.read_text()) == {"status": "first"}


def test_claim_boundary_keeps_localization_narrow() -> None:
    source = Path(summary.__file__).read_text()

    assert "Known failure cells are outcome-selected and localization-only" in source
    assert "does not prove rounding is the sole mechanism" in source
    assert "prospective_sequential_tiered_integrity" in source
    assert "Do not relax" in source
