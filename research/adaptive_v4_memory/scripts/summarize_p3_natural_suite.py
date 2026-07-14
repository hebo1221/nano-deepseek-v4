from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

BENCHMARK_IDS = {
    "RULER": "p3-natural-ruler-audit-v1",
    "SCBench": "p3-natural-scbench-audit-v1",
    "LongBench-v2": "p3-natural-longbench-v2-audit-v1",
    "LongMemEval": "p3-natural-longmemeval-audit-v1",
    "MRCR": "p3-natural-mrcr-audit-v1",
}
DEPENDENCY_IDS = {
    "causal_gate": "p2-causal-ablation-audit-v1",
    "dataset_inventory": "p3-natural-dataset-inventory-v1",
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


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Invalid {label} digest.")
    int(value, 16)


def audit_benchmark(
    *,
    name: str,
    path: Path,
    expected_examples: int,
    required_arms: tuple[str, ...],
    allowed_failures: set[str],
    manifest_digest: str,
    model_snapshot_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file(), f"Missing natural benchmark summary: {path}")
    payload = json.loads(path.read_text())
    _require(payload.get("experiment_id") == BENCHMARK_IDS[name], f"Wrong {name} audit id.")
    _require(payload.get("benchmark") == name, f"Wrong benchmark label for {name}.")
    _require(payload.get("source", {}).get("dirty") is False, f"Dirty {name} evidence.")
    _require(
        payload.get("experiment_manifest", {}).get("sha256") == manifest_digest,
        f"{name} manifest dependency drifted.",
    )
    audit = payload.get("audit", {})
    _require(audit.get("all_raw_artifacts_verified") is True, f"{name} raw audit failed.")
    _require(
        audit.get("all_failure_accounting_complete") is True,
        f"{name} failure accounting is incomplete.",
    )
    _sha256_value(audit.get("raw_record_digest_set_sha256"), f"{name} raw-record set")
    arms = payload.get("arms", {})
    _require(set(required_arms).issubset(arms), f"{name} is missing a required baseline arm.")
    arm_rows: dict[str, Any] = {}
    for arm in required_arms:
        row = arms[arm]
        failures = row.get("failures_by_type", {})
        _require(
            set(failures).issubset(allowed_failures),
            f"{name}/{arm} contains an unregistered failure type.",
        )
        _require(
            all(isinstance(count, int) and count >= 0 for count in failures.values()),
            f"{name}/{arm} has an invalid failure count.",
        )
        scored = row.get("scored_examples")
        accounted = row.get("accounted_examples")
        _require(isinstance(scored, int) and scored >= 0, f"{name}/{arm} score count invalid.")
        _require(
            accounted == scored + sum(failures.values()),
            f"{name}/{arm} score and failure counts do not close.",
        )
        _require(
            row.get("expected_examples") == expected_examples
            and accounted == expected_examples
            and row.get("terminal") is True,
            f"{name}/{arm} does not account for every frozen example.",
        )
        arm_rows[arm] = {
            "terminal": True,
            "expected_examples": expected_examples,
            "scored_examples": scored,
            "failed_examples": sum(failures.values()),
            "failures_by_type": failures,
        }
    conditional = payload.get("conditional_arms", {})
    _require(
        all(
            row.get("status") in {"complete", "incompatible", "withheld-by-causal-gate"}
            for row in conditional.values()
        ),
        f"{name} conditional-arm disposition is incomplete.",
    )
    result = {
        "terminal": True,
        "native_and_fixed_terminal": all(arm_rows[arm]["terminal"] for arm in required_arms),
        "expected_examples_per_required_arm": expected_examples,
        "required_arms": arm_rows,
        "conditional_arms": conditional,
        "summary": {"path": str(path), "sha256": sha256(path)},
    }
    dependencies = {
        "causal_gate": payload.get("causal_gate"),
        "dataset_inventory": payload.get("dataset_inventory"),
        "model_snapshot_digest_set_sha256": payload.get("model_snapshot_digest_set_sha256"),
    }
    _sha256_value(dependencies["model_snapshot_digest_set_sha256"], f"{name} model set")
    _require(
        dependencies["model_snapshot_digest_set_sha256"] == model_snapshot_digest,
        f"{name} model snapshot does not match the frozen manifest.",
    )
    for dependency_name in ("causal_gate", "dataset_inventory"):
        metadata = dependencies[dependency_name]
        _require(isinstance(metadata, dict), f"Missing {name} {dependency_name} dependency.")
        dependency_path = Path(metadata.get("path", ""))
        _require(dependency_path.is_file(), f"Missing {name} dependency: {dependency_path}")
        _require(
            metadata.get("sha256") == sha256(dependency_path),
            f"{name} {dependency_name} digest drifted.",
        )
        dependency = json.loads(dependency_path.read_text())
        _require(
            dependency.get("experiment_id") == DEPENDENCY_IDS[dependency_name],
            f"{name} {dependency_name} has the wrong experiment id.",
        )
        _require(
            dependency.get("source", {}).get("dirty") is False,
            f"{name} {dependency_name} was produced from a dirty source tree.",
        )
    return result, dependencies


def summarize(manifest_path: Path, summary_paths: dict[str, Path]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    _require(
        manifest.get("experiment_id") == "p3-natural-language-suite-v1",
        "Wrong natural-suite manifest.",
    )
    manifest_digest = sha256(manifest_path)
    required_arms = tuple(manifest["common_protocol"]["p4_gate_baseline_arms"])
    expected = manifest["suite_audit"]["per_arm_minimum_accounted_examples"]
    allowed_failures = set(manifest["common_protocol"]["failure_accounting"])
    model_snapshot_digest = manifest["model"]["snapshot_digest_set_sha256"]
    _sha256_value(model_snapshot_digest, "frozen model snapshot set")
    _require(set(summary_paths) == set(BENCHMARK_IDS), "Natural benchmark summary set drifted.")
    benchmarks: dict[str, Any] = {}
    dependency_sets: list[dict[str, Any]] = []
    for name in manifest["execution_order"]:
        result, dependencies = audit_benchmark(
            name=name,
            path=summary_paths[name],
            expected_examples=expected[name],
            required_arms=required_arms,
            allowed_failures=allowed_failures,
            manifest_digest=manifest_digest,
            model_snapshot_digest=model_snapshot_digest,
        )
        benchmarks[name] = result
        dependency_sets.append(dependencies)
    causal_digests = {row["causal_gate"]["sha256"] for row in dependency_sets}
    inventory_digests = {row["dataset_inventory"]["sha256"] for row in dependency_sets}
    model_digests = {row["model_snapshot_digest_set_sha256"] for row in dependency_sets}
    _require(len(causal_digests) == 1, "Natural benchmarks used different causal gates.")
    _require(len(inventory_digests) == 1, "Natural benchmarks used different datasets.")
    _require(len(model_digests) == 1, "Natural benchmarks used different model snapshots.")
    totals = {
        arm: sum(
            benchmarks[name]["required_arms"][arm]["expected_examples"]
            for name in manifest["execution_order"]
        )
        for arm in required_arms
    }
    minimum = manifest["execution_totals"]["minimum_predictions_per_arm"]
    _require(all(total == minimum for total in totals.values()), "Natural arm total drifted.")
    return {
        "schema_version": 1,
        "experiment_id": "p3-natural-language-suite-audit-v1",
        "experiment_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
        },
        "audit": {
            "all_required_artifacts_verified": True,
            "all_required_baseline_cells_terminal": True,
            "all_failure_accounting_complete": True,
            "benchmarks_terminal": len(benchmarks),
            "minimum_protocol_examples_accounted_per_arm": minimum,
            "accounted_examples_by_required_arm": totals,
            "causal_gate_sha256": next(iter(causal_digests)),
            "dataset_inventory_sha256": next(iter(inventory_digests)),
            "model_snapshot_digest_set_sha256": next(iter(model_digests)),
        },
        "benchmarks": benchmarks,
        "claim_boundary": (
            "Five-benchmark evidence on one pinned compatible Qwen3 model. Explicit "
            "failures count toward coverage but not quality; this is not official DeepSeek-V4 evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the five-benchmark P3 natural suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-suite.summary.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary_paths = {
        name: Path(path) for name, path in manifest["suite_audit"]["benchmark_summaries"].items()
    }
    payload = summarize(args.manifest, summary_paths)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Natural-suite summarization requires a clean source tree.")
    payload["source"] = {"commit": commit, "dirty": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
