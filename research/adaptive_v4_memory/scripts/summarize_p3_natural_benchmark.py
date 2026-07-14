from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from summarize_p3_natural_suite import BENCHMARK_IDS, sha256

STATUS_VALUES = {"scored", "failure"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Invalid {label} digest.")
    int(value, 16)


def _dependency(metadata: Any, label: str) -> dict[str, str]:
    _require(isinstance(metadata, dict), f"Missing {label} dependency.")
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing {label}: {path}")
    _require(metadata.get("sha256") == sha256(path), f"{label} digest drifted.")
    return {"path": str(path), "sha256": metadata["sha256"]}


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), f"Raw record {line_number} is not an object.")
            records.append(row)
    return records


def audit_arm(
    *,
    benchmark: str,
    arm: str,
    artifact_path: Path,
    expected_examples: int,
    manifest_digest: str,
    allowed_failures: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(artifact_path.read_text())
    _require(
        artifact.get("experiment_id") == "p3-natural-benchmark-arm-cell-v1",
        f"Wrong {benchmark}/{arm} cell id.",
    )
    _require(
        artifact.get("benchmark") == benchmark and artifact.get("arm") == arm,
        f"Wrong {benchmark}/{arm} cell coordinates.",
    )
    _require(artifact.get("status") == "terminal", f"{benchmark}/{arm} is not terminal.")
    _require(
        artifact.get("source", {}).get("dirty") is False,
        f"Dirty {benchmark}/{arm} source.",
    )
    _require(
        artifact.get("experiment_manifest", {}).get("sha256") == manifest_digest,
        f"{benchmark}/{arm} manifest drifted.",
    )
    raw_metadata = artifact.get("raw_records", {})
    raw_path = Path(raw_metadata.get("path", ""))
    _require(raw_path.is_file(), f"Missing {benchmark}/{arm} raw records.")
    _require(raw_metadata.get("sha256") == sha256(raw_path), f"{benchmark}/{arm} raw drift.")
    records = _records(raw_path)
    _require(len(records) == expected_examples, f"{benchmark}/{arm} record count drifted.")
    identifiers: set[str] = set()
    failures: dict[str, int] = {}
    scored = 0
    record_digests: list[str] = []
    for row in records:
        identifier = row.get("example_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Natural example id is invalid.")
        _require(identifier not in identifiers, f"Duplicate natural example id: {identifier}")
        identifiers.add(identifier)
        _require(row.get("benchmark") == benchmark and row.get("arm") == arm, "Record drift.")
        _require(row.get("status") in STATUS_VALUES, f"Invalid record status: {identifier}")
        _require(
            isinstance(row.get("exact_input_tokens"), int)
            and row["exact_input_tokens"] >= 0
            and isinstance(row.get("generation_reserve_tokens"), int)
            and row["generation_reserve_tokens"] > 0,
            f"Invalid token accounting: {identifier}",
        )
        _sha256_value(row.get("raw_prompt_sha256"), f"{identifier} prompt")
        _require(
            isinstance(row.get("latency_ms"), (int, float)) and row["latency_ms"] >= 0,
            f"Invalid latency: {identifier}",
        )
        _require(
            isinstance(row.get("peak_hbm_bytes"), int) and row["peak_hbm_bytes"] >= 0,
            f"Invalid HBM accounting: {identifier}",
        )
        if row["status"] == "scored":
            score = row.get("score")
            _require(
                isinstance(score, (int, float)) and 0.0 <= score <= 1.0,
                f"Invalid score: {identifier}",
            )
            _require(row.get("failure_type") is None, f"Scored record has failure: {identifier}")
            scored += 1
        else:
            failure = row.get("failure_type")
            if not isinstance(failure, str) or failure not in allowed_failures:
                raise ValueError(f"Unregistered failure: {failure}")
            _require(row.get("score") is None, f"Failed record has a score: {identifier}")
            failures[failure] = failures.get(failure, 0) + 1
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        record_digests.append(hashlib.sha256(canonical.encode()).hexdigest())
    _require(scored + sum(failures.values()) == expected_examples, "Arm accounting does not close.")
    dependencies = {
        "causal_gate": _dependency(artifact.get("causal_gate"), "causal gate"),
        "dataset_inventory": _dependency(
            artifact.get("dataset_inventory"), "dataset inventory"
        ),
        "model_snapshot_digest_set_sha256": artifact.get(
            "model_snapshot_digest_set_sha256"
        ),
    }
    _sha256_value(
        dependencies["model_snapshot_digest_set_sha256"], "model snapshot set"
    )
    return (
        {
            "terminal": True,
            "expected_examples": expected_examples,
            "accounted_examples": expected_examples,
            "scored_examples": scored,
            "failures_by_type": failures,
            "mean_score_over_scored": (
                sum(float(row["score"]) for row in records if row["status"] == "scored")
                / scored
                if scored
                else None
            ),
            "raw_cell": {"path": str(artifact_path), "sha256": sha256(artifact_path)},
            "raw_record_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(record_digests)).encode()
            ).hexdigest(),
        },
        dependencies,
    )


def summarize_benchmark(
    *,
    benchmark: str,
    manifest_path: Path,
    arm_artifacts: dict[str, Path],
    conditional_arms: dict[str, str],
) -> dict[str, Any]:
    _require(benchmark in BENCHMARK_IDS, f"Unknown natural benchmark: {benchmark}")
    manifest = json.loads(manifest_path.read_text())
    manifest_digest = sha256(manifest_path)
    required = tuple(manifest["common_protocol"]["p4_gate_baseline_arms"])
    _require(set(arm_artifacts) == set(required), "Required natural arm set drifted.")
    allowed_failures = set(manifest["common_protocol"]["failure_accounting"])
    expected = manifest["suite_audit"]["per_arm_minimum_accounted_examples"][benchmark]
    arms: dict[str, Any] = {}
    dependencies: list[dict[str, Any]] = []
    for arm in required:
        arms[arm], dependency = audit_arm(
            benchmark=benchmark,
            arm=arm,
            artifact_path=arm_artifacts[arm],
            expected_examples=expected,
            manifest_digest=manifest_digest,
            allowed_failures=allowed_failures,
        )
        dependencies.append(dependency)
    causal = {row["causal_gate"]["sha256"] for row in dependencies}
    inventories = {row["dataset_inventory"]["sha256"] for row in dependencies}
    models = {row["model_snapshot_digest_set_sha256"] for row in dependencies}
    _require(len(causal) == len(inventories) == len(models) == 1, "Arm dependencies drifted.")
    allowed_conditional = {"complete", "incompatible", "withheld-by-causal-gate"}
    _require(
        set(conditional_arms) == set(manifest["common_protocol"]["conditional_arms"])
        and set(conditional_arms.values()).issubset(allowed_conditional),
        "Conditional natural arm disposition drifted.",
    )
    return {
        "schema_version": 1,
        "experiment_id": BENCHMARK_IDS[benchmark],
        "benchmark": benchmark,
        "experiment_manifest": {"path": str(manifest_path), "sha256": manifest_digest},
        "audit": {
            "all_raw_artifacts_verified": True,
            "all_failure_accounting_complete": True,
            "raw_record_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(row["raw_record_digest_set_sha256"] for row in arms.values())).encode()
            ).hexdigest(),
        },
        "arms": arms,
        "conditional_arms": {
            name: {"status": status} for name, status in conditional_arms.items()
        },
        "causal_gate": dependencies[0]["causal_gate"],
        "dataset_inventory": dependencies[0]["dataset_inventory"],
        "model_snapshot_digest_set_sha256": next(iter(models)),
    }


def _mapping(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        _require(bool(separator and name and item), f"Invalid {label}: {value}")
        _require(name not in result, f"Duplicate {label}: {name}")
        result[name] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one natural benchmark across arms.")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_IDS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument("--arm-artifact", action="append", required=True)
    parser.add_argument("--conditional-arm", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize_benchmark(
        benchmark=args.benchmark,
        manifest_path=args.manifest,
        arm_artifacts={
            name: Path(path) for name, path in _mapping(args.arm_artifact, "arm artifact").items()
        },
        conditional_arms=_mapping(args.conditional_arm, "conditional arm"),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Natural benchmark summarization requires a clean source tree.")
    payload["source"] = {"commit": commit, "dirty": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
