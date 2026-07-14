from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCALES = ("s55", "s151")
WORKLOADS = ("single-retrieval", "query-shift", "dense-memory")
ARMS = ("native", "fixed", "online", "dense")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    _require(
        summary.get("experiment_id") == "m5-final-validation-v1"
        and summary.get("decision") == "pilot_negative_result"
        and summary.get("gates", {}).get("quality_vs_fixed") is False,
        "M5 pilot conclusion drifted.",
    )
    _require(
        summary.get("claim_scope")
        == "The tested one-token cross-layer global M2 interface on two Tier-S "
        "checkpoints and three synthetic pilot workloads.",
        "M5 pilot claim boundary drifted.",
    )
    raw_artifacts = summary.get("raw_artifacts", {})
    rows: dict[str, Any] = {}
    for scale in SCALES:
        metadata = raw_artifacts.get(scale, {})
        path = Path(metadata.get("path", ""))
        _require(
            path.is_file() and metadata.get("sha256") == sha256(path),
            f"M5 {scale} raw artifact drifted.",
        )
        raw = json.loads(path.read_text())
        _require(
            raw.get("experiment_id") == "m5-online-controller-v1"
            and raw.get("scale") == scale
            and set(raw.get("workloads", {})) == set(WORKLOADS)
            and any("one-token-lookahead" in item for item in raw.get("limitations", [])),
            f"M5 {scale} raw contract drifted.",
        )
        for workload in WORKLOADS:
            arms = raw["workloads"][workload]
            _require(set(arms) == set(ARMS), f"M5 {scale}/{workload} arm set drifted.")
            _require(
                all(
                    arm.get("budget_violations") == 0
                    and arm.get("total", 0) > 0
                    and arm.get("accuracy") == arm.get("correct") / arm.get("total")
                    for arm in arms.values()
                ),
                f"M5 {scale}/{workload} accounting drifted.",
            )
            for arm_name in ARMS:
                _require(
                    summary["accuracy"][scale][workload][arm_name]
                    == arms[arm_name]["accuracy"],
                    f"M5 {scale}/{workload}/{arm_name} summary drifted.",
                )
        rows[scale] = {
            "raw_artifact": {"path": str(path), "sha256": sha256(path)},
            "workloads": raw["workloads"],
        }
    official = raw_artifacts.get("official_audit", {})
    official_path = Path(official.get("path", ""))
    _require(
        official_path.is_file() and official.get("sha256") == sha256(official_path),
        "M5 official feasibility artifact drifted.",
    )
    return {
        "schema_version": 1,
        "experiment_id": "m5-one-token-pilot-baseline-audit-v1",
        "legacy_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "audit": {
            "raw_artifacts_verified": True,
            "scales_verified": len(SCALES),
            "workloads_per_scale": len(WORKLOADS),
            "required_arms_verified": len(ARMS),
            "one_token_semantics_verified": True,
            "pilot_negative_result_verified": True,
        },
        "scales": rows,
        "official_feasibility": {
            "path": str(official_path),
            "sha256": sha256(official_path),
            "official_weight_run": summary["gates"]["official_weight_run"],
        },
        "decision": "pilot_negative_result",
        "claim_boundary": summary["claim_scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit legacy one-token M5 baseline evidence.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("research/adaptive_v4_memory/results/m5-final-validation.summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1/m5-pilot-baseline.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "M5 pilot audit requires a clean source tree.")
    payload = audit(args.summary)
    payload["source"] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": False,
        "implementation_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
