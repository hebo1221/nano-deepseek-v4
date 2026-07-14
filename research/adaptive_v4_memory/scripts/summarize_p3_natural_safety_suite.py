from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from p3_source_provenance import verify_git_implementation
from prepare_p3_natural_safety_assets import sha256
from validate_p3_natural_safety_manifest import validate_manifest

ARMS = ("native-dense", "strongest-memory-matched-fixed")
LONGSAFETY_SUMMARIZER_PATH = "research/adaptive_v4_memory/scripts/summarize_p3_longsafety.py"
IFEVAL_SCORER_PATH = "research/adaptive_v4_memory/scripts/score_p3_ifeval.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _unit_interval_number(value: Any) -> bool:
    return type(value) in (int, float) and (
        type(value) is int or math.isfinite(value)
    ) and 0.0 <= value <= 1.0


def _signed_unit_interval_number(value: Any) -> bool:
    return type(value) in (int, float) and (
        type(value) is int or math.isfinite(value)
    ) and -1.0 <= value <= 1.0


def _bound_artifact(metadata: Any, label: str) -> None:
    _require(isinstance(metadata, dict), f"Missing {label} metadata.")
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == sha256(path),
        f"{label} artifact drifted.",
    )


def summarize(*, manifest_path: Path, longsafety_path: Path, ifeval_path: Path) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    validation = validate_manifest(manifest)
    longsafety = json.loads(longsafety_path.read_text())
    ifeval = json.loads(ifeval_path.read_text())
    _require(
        longsafety.get("experiment_id") == "p3-natural-safety-longsafety-generation-audit-v1"
        and longsafety.get("source", {}).get("dirty") is False
        and longsafety.get("manifest", {}).get("sha256") == manifest_digest,
        "LongSafety generation audit is missing, dirty, or stale.",
    )
    verify_git_implementation(
        longsafety.get("source"),
        expected_path=LONGSAFETY_SUMMARIZER_PATH,
        label="LongSafety audit",
    )
    long_contract = manifest["benchmarks"]["LongSafety"]
    expected_long = long_contract["prompt_protocol"]["expected_predictions_per_arm"]
    long_audit = longsafety.get("audit", {})
    _require(
        long_audit.get("generation_arms_terminal") is True
        and long_audit.get("input_pairing_verified") is True
        and long_audit.get("generation_failure_accounting_complete") is True
        and long_audit.get("source_implementations_verified") is True
        and long_audit.get("dependency_digests_verified") is True
        and long_audit.get("record_revisions_verified") is True
        and long_audit.get("terminal_measurement_schema_verified") is True
        and long_audit.get("generation_seed_verified") is True
        and long_audit.get("official_judge_status") == "blocked"
        and long_audit.get("expected_generations_per_arm") == expected_long
        and long_audit.get("expected_generations_total") == expected_long * len(ARMS)
        and set(longsafety.get("arms", {})) == set(ARMS),
        "LongSafety generation coverage is incomplete.",
    )
    for arm in ARMS:
        row = longsafety["arms"][arm]
        failures = row.get("failures_by_type")
        _require(
            row.get("terminal_generation") is True
            and row.get("expected_generations") == expected_long
            and _nonnegative_integer(row.get("generated"))
            and isinstance(failures, dict)
            and set(failures).issubset(manifest["failure_accounting"])
            and all(_nonnegative_integer(count) for count in failures.values())
            and row["generated"] + sum(failures.values()) == expected_long,
            f"LongSafety arm accounting is incomplete: {arm}.",
        )
        _bound_artifact(row.get("raw_cell"), f"LongSafety/{arm} raw cell")
    judge = longsafety.get("official_judge", {})
    _require(
        judge.get("status") == "blocked"
        and judge.get("safety_scores_reported") is False
        and judge.get("raw_generations_preserved") is True,
        "LongSafety judge boundary drifted.",
    )

    _require(
        ifeval.get("experiment_id") == "p3-natural-safety-ifeval-official-audit-v1"
        and ifeval.get("source", {}).get("dirty") is False
        and ifeval.get("manifest", {}).get("sha256") == manifest_digest
        and ifeval.get("input_pairing_verified") is True
        and set(ifeval.get("arms", {})) == set(ARMS),
        "IFEval official audit is missing, dirty, stale, or unpaired.",
    )
    verify_git_implementation(
        ifeval.get("source"),
        expected_path=IFEVAL_SCORER_PATH,
        label="IFEval audit",
    )
    expected_ifeval = manifest["benchmarks"]["IFEval"]["protocol"]["expected_prompts_per_arm"]
    _require(
        set(ifeval.get("generation_cells", {})) == set(ARMS),
        "IFEval generation cell set drifted.",
    )
    ifeval_audit = ifeval.get("audit", {})
    _require(
        ifeval_audit.get("required_arms_terminal") is True
        and ifeval_audit.get("input_pairing_verified") is True
        and ifeval_audit.get("official_scoring_accounted") is True
        and ifeval_audit.get("source_implementations_verified") is True
        and ifeval_audit.get("expected_prompts_per_arm") == expected_ifeval,
        "IFEval official audit contract drifted.",
    )
    for arm in ARMS:
        metrics = ifeval["arms"][arm].get("metrics", {})
        scored = metrics.get("scored_prompts")
        scorer_failures = metrics.get("scorer_failures")
        generation_failures = metrics.get("generation_failures")
        instruction_total = metrics.get("instruction_total")
        _require(
            type(metrics.get("expected_prompts")) is int
            and metrics["expected_prompts"] == expected_ifeval
            and _nonnegative_integer(scored)
            and _nonnegative_integer(scorer_failures)
            and scored + scorer_failures == expected_ifeval
            and _nonnegative_integer(generation_failures)
            and generation_failures <= expected_ifeval
            and _positive_integer(instruction_total)
            and instruction_total >= expected_ifeval
            and all(
                _unit_interval_number(metrics.get(name))
                for name in manifest["benchmarks"]["IFEval"]["protocol"]["official_metrics"]
            ),
            f"IFEval official accounting is incomplete: {arm}.",
        )
        _bound_artifact(
            ifeval["generation_cells"].get(arm), f"IFEval/{arm} generation cell"
        )
        _bound_artifact(
            ifeval["arms"][arm].get("raw_official_results"),
            f"IFEval/{arm} official results",
        )
    paired = ifeval.get("paired_fixed_minus_native", {})
    interval = paired.get("paired_bootstrap_95_ci")
    _require(
        type(paired.get("paired_prompts")) is int
        and paired["paired_prompts"] == expected_ifeval
        and type(paired.get("bootstrap_replicates")) is int
        and paired["bootstrap_replicates"]
        == manifest["statistics"]["paired_bootstrap_replicates"],
        "IFEval paired statistics drifted.",
    )
    _require(
        paired.get("bootstrap_seed") == manifest["statistics"]["paired_bootstrap_seed"]
        and _signed_unit_interval_number(paired.get("mean_difference"))
        and isinstance(interval, list)
        and len(interval) == 2
        and all(
            type(value) in (int, float)
            and (type(value) is int or math.isfinite(value))
            and -1.0 <= value <= 1.0
            for value in interval
        )
        and interval[0] <= interval[1]
        and all(
            _nonnegative_integer(paired.get(name)) for name in ("wins", "ties", "losses")
        )
        and paired["wins"] + paired["ties"] + paired["losses"] == expected_ifeval,
        "IFEval paired statistical schema drifted.",
    )
    return {
        "schema_version": 1,
        "experiment_id": "p3-natural-safety-suite-audit-v1",
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
            "validation": validation,
        },
        "audit": {
            "required_arms": len(ARMS),
            "longsafety_generation_terminal": True,
            "longsafety_input_pairing_verified": True,
            "longsafety_expected_generations_per_arm": expected_long,
            "longsafety_official_judge_status": "blocked",
            "longsafety_safety_scores_reported": False,
            "longsafety_raw_evidence_verified": True,
            "ifeval_official_terminal": True,
            "ifeval_input_pairing_verified": True,
            "ifeval_expected_prompts_per_arm": expected_ifeval,
            "failure_accounting_complete": True,
            "source_implementations_verified": True,
            "raw_artifact_digests_verified": True,
            "statistical_schema_verified": True,
            "comparative_long_context_safety_claim_available": False,
        },
        "longsafety": {
            "summary": {"path": str(longsafety_path), "sha256": sha256(longsafety_path)},
            "arms": longsafety["arms"],
            "official_judge": judge,
        },
        "ifeval": {
            "summary": {"path": str(ifeval_path), "sha256": sha256(ifeval_path)},
            "arms": ifeval["arms"],
            "paired_fixed_minus_native": paired,
            "evidence_role": "official short-prompt instruction-following control",
        },
        "classification": "bounded-generation-and-control-result-with-paid-judge-blocker",
        "claim_boundary": manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the P3 natural safety suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    )
    parser.add_argument(
        "--longsafety",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/longsafety.summary.json"
        ),
    )
    parser.add_argument(
        "--ifeval",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/ifeval.summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety.summary.json"),
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Natural safety suite summarization requires a clean source tree.")
    payload = summarize(
        manifest_path=args.manifest,
        longsafety_path=args.longsafety,
        ifeval_path=args.ifeval,
    )
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
