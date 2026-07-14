from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from p3_source_provenance import verify_git_implementation
from summarize_p3_natural_suite import BENCHMARK_IDS, sha256

STATUS_VALUES = {"scored", "failure"}
RUNNER_PATHS = {
    "RULER": "research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py",
    "SCBench": "research/adaptive_v4_memory/scripts/run_p3_scbench.py",
    "LongBench-v2": "research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py",
    "LongMemEval": "research/adaptive_v4_memory/scripts/run_p3_longmemeval.py",
    "MRCR": "research/adaptive_v4_memory/scripts/run_p3_mrcr.py",
}


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


def _verify_source_implementation(artifact: dict[str, Any], benchmark: str) -> dict[str, str]:
    runner_path = RUNNER_PATHS[benchmark]
    verified = verify_git_implementation(
        artifact.get("source"), expected_path=runner_path, label=benchmark
    )
    return {
        "commit": verified["commit"],
        "runner_path": runner_path,
        "implementation_sha256": verified["implementation_sha256"],
    }


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
    _sha256_value(
        artifact.get("source", {}).get("implementation_sha256"),
        f"{benchmark}/{arm} implementation",
    )
    source_implementation = _verify_source_implementation(artifact, benchmark)
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
    score_sum = 0.0
    record_digests: list[str] = []
    paired_input_digests: list[str] = []
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
        paired_input_digests.append(f"{identifier}:{row['raw_prompt_sha256']}")
        _require(
            isinstance(row.get("latency_ms"), (int, float)) and row["latency_ms"] >= 0,
            f"Invalid latency: {identifier}",
        )
        _require(
            isinstance(row.get("peak_hbm_bytes"), int) and row["peak_hbm_bytes"] >= 0,
            f"Invalid HBM accounting: {identifier}",
        )
        _require(
            isinstance(row.get("hot_resident_bytes"), int) and row["hot_resident_bytes"] >= 0,
            f"Invalid hot-memory accounting: {identifier}",
        )
        _require(isinstance(row.get("raw_response"), str), f"Missing response: {identifier}")
        _require(
            row.get("parsed_response") is None or isinstance(row.get("parsed_response"), str),
            f"Invalid parsed response: {identifier}",
        )
        _require(
            isinstance(row.get("stop_reason"), str) and row["stop_reason"],
            f"Missing stop reason: {identifier}",
        )
        revisions = row.get("revisions")
        _require(
            isinstance(revisions, dict)
            and all(
                isinstance(revisions.get(key), str) and revisions[key]
                for key in (
                    "model_revision",
                    "dataset_revision",
                    "code_revision",
                    "scorer_sha256",
                )
            ),
            f"Incomplete revision provenance: {identifier}",
        )
        assert isinstance(revisions, dict)
        _require(isinstance(row.get("arm_config"), dict), f"Missing arm config: {identifier}")
        if benchmark in {"RULER", "SCBench", "LongBench-v2", "LongMemEval", "MRCR"}:
            _require(
                isinstance(row.get("token_boundary_retreat"), int)
                and row["token_boundary_retreat"] >= 0,
                f"Invalid exact-token boundary accounting: {identifier}",
            )
        if benchmark == "RULER":
            _sha256_value(row.get("input_token_ids_sha256"), f"{identifier} token ids")
            _sha256_value(
                revisions.get("official_scorer_sha256"),
                f"{identifier} official scorer",
            )
        if benchmark == "SCBench":
            _sha256_value(row.get("input_token_ids_sha256"), f"{identifier} token ids")
            _require(
                row.get("mode") in {"multi-turn", "multi-request"}
                and isinstance(row.get("task"), str)
                and row["task"].startswith("scbench_")
                and isinstance(row.get("row_index"), int)
                and row["row_index"] >= 0
                and isinstance(row.get("turn_index"), int)
                and row["turn_index"] >= 0,
                f"Invalid SCBench turn coordinates: {identifier}",
            )
            if row["status"] == "scored":
                _require(
                    isinstance(row.get("scorer_detail"), dict)
                    and isinstance(row["scorer_detail"].get("metric"), str)
                    and bool(row["scorer_detail"]["metric"]),
                    f"Missing SCBench official scorer detail: {identifier}",
                )
        if benchmark == "LongMemEval":
            judge = row.get("judge")
            if row["status"] == "scored":
                _require(
                    isinstance(judge, dict)
                    and judge.get("model") == "gpt-4o-2024-08-06"
                    and judge.get("status") == "scored"
                    and isinstance(judge.get("returned_model"), str)
                    and bool(judge["returned_model"])
                    and isinstance(judge.get("response_id"), str)
                    and bool(judge["response_id"])
                    and isinstance(judge.get("created"), int)
                    and isinstance(judge.get("prompt"), str)
                    and bool(judge["prompt"])
                    and isinstance(judge.get("raw_response"), str)
                    and bool(judge["raw_response"])
                    and isinstance(judge.get("latency_ms"), (int, float))
                    and judge["latency_ms"] >= 0,
                    f"Incomplete official judge provenance: {identifier}",
                )
            elif row.get("failure_type") == "judge-blocked":
                _require(
                    isinstance(judge, dict)
                    and judge.get("model") == "gpt-4o-2024-08-06"
                    and judge.get("status") == "blocked"
                    and isinstance(judge.get("reason"), str)
                    and bool(judge["reason"])
                    and isinstance(row.get("generated_tokens_observed"), int)
                    and row["generated_tokens_observed"] >= 0,
                    f"Incomplete blocked-judge provenance: {identifier}",
                )
        if row["status"] == "scored":
            score = row.get("score")
            if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid score: {identifier}")
            _require(row.get("failure_type") is None, f"Scored record has failure: {identifier}")
            scored += 1
            score_sum += float(score)
        else:
            failure = row.get("failure_type")
            if not isinstance(failure, str) or failure not in allowed_failures:
                raise ValueError(f"Unregistered failure: {failure}")
            _require(row.get("score") is None, f"Failed record has a score: {identifier}")
            failures[failure] = failures.get(failure, 0) + 1
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        record_digests.append(hashlib.sha256(canonical.encode()).hexdigest())
    _require(scored + sum(failures.values()) == expected_examples, "Arm accounting does not close.")
    benchmark_dataset_digest = artifact.get("benchmark_dataset_digest_set_sha256")
    if benchmark == "RULER" or benchmark_dataset_digest is not None:
        _sha256_value(benchmark_dataset_digest, f"{benchmark} dataset manifest set")
    dependencies = {
        "causal_gate": _dependency(artifact.get("causal_gate"), "causal gate"),
        "dataset_inventory": _dependency(artifact.get("dataset_inventory"), "dataset inventory"),
        "fixed_baseline_selection": _dependency(
            artifact.get("fixed_baseline_selection"), "fixed baseline selection"
        ),
        "model_snapshot_digest_set_sha256": artifact.get("model_snapshot_digest_set_sha256"),
        "benchmark_dataset_digest_set_sha256": benchmark_dataset_digest,
    }
    _sha256_value(dependencies["model_snapshot_digest_set_sha256"], "model snapshot set")
    return (
        {
            "terminal": True,
            "expected_examples": expected_examples,
            "accounted_examples": expected_examples,
            "scored_examples": scored,
            "failures_by_type": failures,
            "mean_score_over_scored": (score_sum / scored if scored else None),
            "mean_score_over_all_expected_failures_zero": score_sum / expected_examples,
            "paired_example_prompt_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(paired_input_digests)).encode()
            ).hexdigest(),
            "raw_cell": {"path": str(artifact_path), "sha256": sha256(artifact_path)},
            "raw_record_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(record_digests)).encode()
            ).hexdigest(),
            "source_implementation": source_implementation,
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
    fixed_selections = {row["fixed_baseline_selection"]["sha256"] for row in dependencies}
    models = {row["model_snapshot_digest_set_sha256"] for row in dependencies}
    benchmark_datasets = {row["benchmark_dataset_digest_set_sha256"] for row in dependencies}
    _require(
        len(causal)
        == len(inventories)
        == len(fixed_selections)
        == len(models)
        == len(benchmark_datasets)
        == 1,
        "Arm dependencies drifted.",
    )
    paired_inputs = {row["paired_example_prompt_digest_set_sha256"] for row in arms.values()}
    _require(len(paired_inputs) == 1, "Natural arms used different examples or prompts.")
    source_implementations = {
        (
            row["source_implementation"]["commit"],
            row["source_implementation"]["runner_path"],
            row["source_implementation"]["implementation_sha256"],
        )
        for row in arms.values()
    }
    _require(
        len(source_implementations) == 1,
        "Natural arms used different source implementations.",
    )
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
            "all_required_arms_input_paired": True,
            "all_source_implementations_verified": True,
            "raw_record_digest_set_sha256": hashlib.sha256(
                "\n".join(
                    sorted(row["raw_record_digest_set_sha256"] for row in arms.values())
                ).encode()
            ).hexdigest(),
        },
        "arms": arms,
        "conditional_arms": {name: {"status": status} for name, status in conditional_arms.items()},
        "causal_gate": dependencies[0]["causal_gate"],
        "dataset_inventory": dependencies[0]["dataset_inventory"],
        "fixed_baseline_selection": dependencies[0]["fixed_baseline_selection"],
        "model_snapshot_digest_set_sha256": next(iter(models)),
        "benchmark_dataset_digest_set_sha256": next(iter(benchmark_datasets)),
        "source_implementation": arms[required[0]]["source_implementation"],
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
