from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCALES = ("s55", "s151")
SPLITS = {"train": 768, "calibration": 384, "test": 768}
VARIANTS = {
    "context-layer-only",
    "full-asymmetric-h32",
    "small-h8",
    "symmetric-loss",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(summary_path: Path, raw_root: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    _require(
        summary.get("experiment_id") == "m3-tier-s-learned-risk-controller-v1"
        and summary.get("m3_gate") == "failed_on_both_scales"
        and summary.get("claim_eligible") is False,
        "M3 learned-lookahead conclusion drifted.",
    )
    protocol = summary.get("protocol", {})
    _require(
        protocol.get("train_examples_per_scale") == SPLITS["train"]
        and protocol.get("calibration_examples_per_scale") == SPLITS["calibration"]
        and protocol.get("test_examples_per_scale") == SPLITS["test"]
        and len(set(protocol.get("split_seeds", {}).values())) == len(SPLITS),
        "M3 learned-lookahead split protocol drifted.",
    )
    scales: dict[str, Any] = {}
    for scale in SCALES:
        path = raw_root / scale / "m3-learned.summary.json"
        expected_digest = summary["scales"][scale]["raw_summary_sha256"]
        _require(path.is_file() and sha256(path) == expected_digest, f"M3 {scale} raw drifted.")
        raw = json.loads(path.read_text())
        _require(
            raw.get("experiment_id") == "m3-tier-s-learned-risk-controller-v1"
            and raw.get("scale") == scale
            and raw.get("split_examples") == SPLITS
            and set(raw.get("variants", {})) == VARIANTS
            and raw.get("claim_eligible") is False
            and raw.get("passes_m3_scale_gate") is False
            and raw.get("actual_quality", {}).get("pareto_improved") is False,
            f"M3 {scale} raw contract drifted.",
        )
        checked = summary["scales"][scale]
        _require(
            checked["learned_actual"] == {
                "accuracy": raw["actual_quality"]["learned"]["accuracy"],
                "mean_selected_blocks": raw["actual_quality"]["learned"][
                    "mean_selected_blocks"
                ],
            }
            and checked["m2_actual"] == {
                "accuracy": raw["actual_quality"]["m2_training_free"]["accuracy"],
                "mean_selected_blocks": raw["actual_quality"]["m2_training_free"][
                    "mean_selected_blocks"
                ],
            },
            f"M3 {scale} checked metrics drifted.",
        )
        scales[scale] = {
            "raw_summary": {"path": str(path), "sha256": expected_digest},
            "split_examples": raw["split_examples"],
            "actual_quality": raw["actual_quality"],
            "decision": raw["decision"],
        }
    return {
        "schema_version": 1,
        "experiment_id": "m3-learned-lookahead-baseline-audit-v1",
        "legacy_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "audit": {
            "raw_summaries_verified": True,
            "scales_verified": len(SCALES),
            "independent_splits_verified": True,
            "train_examples_per_scale": SPLITS["train"],
            "calibration_examples_per_scale": SPLITS["calibration"],
            "test_examples_per_scale": SPLITS["test"],
            "ablation_variants_verified": len(VARIANTS),
            "pareto_failure_verified": True,
        },
        "scales": scales,
        "decision": "negative-result",
        "claim_boundary": (
            "Two-scale synthetic learned-lookahead pilot only; its modest quality gain "
            "required 6-11x more selected blocks and was not a Pareto improvement."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the M3 learned-lookahead baseline.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/results/m3-tier-s-learned-risk-controller.summary.json"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/tier_s"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1/m3-learned-baseline.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "M3 learned-lookahead audit requires a clean source tree.")
    payload = audit(args.summary, args.raw_root)
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
