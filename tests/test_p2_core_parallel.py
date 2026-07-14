from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p2_core_parallel import FAMILY_WEIGHTS, partition_seed_families  # noqa: E402


def test_parallel_partition_is_complete_disjoint_and_weight_balanced() -> None:
    seeds = (1, 2, 3, 4, 5)
    families = tuple(FAMILY_WEIGHTS)

    partitions = partition_seed_families(seeds, families, workers=3)
    flattened = [task for partition in partitions for task in partition]
    loads = [sum(FAMILY_WEIGHTS[family] for _seed, family in partition) for partition in partitions]

    assert len(flattened) == len(set(flattened)) == len(seeds) * len(families)
    assert set(flattened) == {(seed, family) for seed in seeds for family in families}
    assert max(loads) - min(loads) <= max(FAMILY_WEIGHTS.values())


def test_parallel_partition_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        partition_seed_families((1,), tuple(FAMILY_WEIGHTS), workers=0)
