from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
matrix = importlib.import_module("run_p2_primary_pin_quota_matrix")
evaluator = importlib.import_module("evaluate_p2_direct_controller_shard_v1_3")
def _token_row(*, hot_blocks: int = 3, hot_bytes: int = 384) -> dict[str, object]:
    return evaluator._seal_row(
        evaluator.TOKEN_SCHEMA_ID,
        {
            "example_index": 0,
            "arm": "fixed",
            "execution_index": 0,
            "token_index": 0,
            "position": 10,
            "actions": [],
            "runtime_soft_lag_snapshot": None,
            "applied_materialization": [],
            "post_rebalance_materialization": [{"hot_blocks": hot_blocks, "hot_bytes": hot_bytes}],
            "signal_diagnostics": [],
            "cuda_peak_allocated_bytes": 1000,
            "cuda_peak_reserved_bytes": 2000,
            "is_cuda_hbm_evidence": True,
            "physical_snapshot_kind": "runtime-soft-lag-exact-fill",
            "decode_incremental_transfer_deltas": [{"layer_index": 2, "h2d_delta_bytes": 128, "d2h_delta_bytes": 64}],
            "action_snapshot_link_digest": "0" * 64,
        },
    )
def test_primary_runner_grid_and_physical_aggregation() -> None:
    left = matrix.TokenAggregate()
    left.add(_token_row())
    left.add(_token_row(hot_blocks=4, hot_bytes=512))
    assert left.result()["h2d_bytes"] == 256
    assert len(matrix.coordinates()) == matrix.EXPECTED_CELLS == 9000
