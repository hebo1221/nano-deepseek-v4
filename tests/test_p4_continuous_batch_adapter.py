from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p4_continuous_batch_adapter as adapter  # noqa: E402


def test_generate_inputs_combines_concurrent_requests_deterministically() -> None:
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=128))

    prompt, decode, digest = adapter.generate_inputs(
        model,
        context=32,
        generation=8,
        batch=2,
        concurrency=4,
        seed=7,
    )
    repeated = adapter.generate_inputs(
        model,
        context=32,
        generation=8,
        batch=2,
        concurrency=4,
        seed=7,
    )

    assert prompt.shape == (8, 32)
    assert decode.shape == (8, 8)
    assert digest == repeated[2]
    assert (
        digest
        == hashlib.sha256(
            b"".join(tensor.contiguous().numpy().tobytes() for tensor in (prompt, decode))
        ).hexdigest()
    )


def test_validate_spec_binds_checkpoint_and_dependencies(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "manifest.json"
    p3_audit = tmp_path / "p3.json"
    for path in (checkpoint, manifest, p3_audit):
        path.write_text(path.name)
    spec = {
        "experiment_id": "p4-production-adapter-spec-v1",
        "cell": {
            "scale": "s55",
            "context": 8192,
            "generation": 128,
            "profile": "serving-b1-c8",
            "batch": 1,
            "concurrency": 8,
        },
        "policies": list(adapter.POLICIES),
        "warmups": 5,
        "measured_repetitions": 30,
        "cell_timeout_seconds": adapter.MAX_CELL_TIMEOUT_SECONDS,
        "repetition_seeds": list(range(35)),
        "checkpoint": str(checkpoint),
        "manifest": {"path": str(manifest), "sha256": adapter.sha256(manifest)},
        "p3_audit": {"path": str(p3_audit), "sha256": adapter.sha256(p3_audit)},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))

    assert adapter.validate_spec(path) == spec

    invalid_timeout = {**spec, "cell_timeout_seconds": float("inf")}
    path.write_text(json.dumps(invalid_timeout))
    with pytest.raises(ValueError, match="timeout contract drifted"):
        adapter.validate_spec(path)

    spec["p3_audit"]["sha256"] = "0" * 64
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="p3_audit drifted"):
        adapter.validate_spec(path)
