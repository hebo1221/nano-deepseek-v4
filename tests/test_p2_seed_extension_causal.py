from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_seed_extension as extension  # noqa: E402
import p2_seed_extension_causal as causal_extension  # noqa: E402


def test_causal_extension_grid_is_exactly_7200_unique_shards() -> None:
    coordinates = causal_extension.expected_coordinates()

    assert len(coordinates) == causal_extension.EXPECTED_SHARDS == 7_200
    assert all(seed in extension.EXTENSION_TRAINING_SEEDS for _, seed, *_ in coordinates)


def test_causal_extension_registry_is_scoped_and_index_aligned() -> None:
    original = (
        causal_extension.causal.TRAINING_SEEDS,
        causal_extension.causal.EVALUATION_SEEDS,
        extension.heldout.CALIBRATION_SEEDS,
    )

    with causal_extension.bind_registry():
        assert causal_extension.causal.TRAINING_SEEDS == (
            extension.EXTENSION_TRAINING_SEEDS
        )
        assert causal_extension.causal.EVALUATION_SEEDS == (
            extension.EXTENSION_EVALUATION_SEEDS
        )
        assert extension.heldout.CALIBRATION_SEEDS == (
            extension.EXTENSION_CALIBRATION_SEEDS
        )

    assert (
        causal_extension.causal.TRAINING_SEEDS,
        causal_extension.causal.EVALUATION_SEEDS,
        extension.heldout.CALIBRATION_SEEDS,
    ) == original


def test_causal_prerequisite_digest_contract_contains_only_present_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    extension_paths = {
        "research/adaptive_v4_memory/scripts/p2_seed_extension_causal.py",
        "research/adaptive_v4_memory/scripts/"
        "run_p2_seed_extension_causal_prerequisites.py",
    }

    assert extension_paths.issubset(causal_extension.IMPLEMENTATION_PATHS)
    assert all((root / path).exists() for path in causal_extension.IMPLEMENTATION_PATHS)
    assert causal_extension.PRIMARY_EXPECTED_SHARDS == 9_000
