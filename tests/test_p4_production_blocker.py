from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_unavailable_production_adapter_fails_closed_with_bound_cell(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = root / (
        "research/adaptive_v4_memory/scripts/p4_unavailable_production_adapter.py"
    )
    spec = tmp_path / "spec.json"
    output = tmp_path / "output.json"
    cell = {
        "scale": "s55",
        "context": 8192,
        "generation": 128,
        "profile": "serving-b1-c1",
        "batch": 1,
        "concurrency": 1,
    }
    spec.write_text(
        json.dumps(
            {
                "experiment_id": "p4-production-adapter-spec-v1",
                "cell": cell,
            }
        )
    )

    result = subprocess.run(
        [str(adapter), "--spec", str(spec), "--output", str(output)],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 78
    assert not output.exists()
    failure = json.loads(result.stderr)
    assert (
        failure["failure_type"]
        == "external-fused-dynamic-production-runtime-unavailable"
    )
    assert failure["cell"] == cell
    blocker = Path(failure["blocker"]["path"])
    assert blocker.is_file()
    assert len(failure["blocker"]["sha256"]) == 64
    blocker_payload = json.loads(blocker.read_text())
    missing = " ".join(blocker_payload["missing_requirements"])
    assert "non-contiguous residency layout" in missing
    assert "position-aware cache-miss recomputation" in missing
