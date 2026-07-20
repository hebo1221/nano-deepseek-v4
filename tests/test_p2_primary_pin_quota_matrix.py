from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
matrix = importlib.import_module("run_p2_primary_pin_quota_matrix")
def test_primary_runner_frozen_grid() -> None:
    assert matrix.ARMS == ("fixed", "fixed+pins", "calibrated-no-pins", "calibrated+pins")
    assert len(matrix.coordinates()) == matrix.EXPECTED_CELLS == 9000
