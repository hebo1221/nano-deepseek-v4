from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
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
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
ScoreVerifier = Callable[[dict[str, Any]], tuple[float, str | None]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Invalid {label} digest.")
    int(value, 16)


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _finite_nonnegative_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and (type(value) is int or math.isfinite(value))
        and value >= 0
    )


def _unit_interval_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and (type(value) is int or math.isfinite(value))
        and 0.0 <= value <= 1.0
    )


def expected_record_revisions(
    benchmark: str, manifest: dict[str, Any]
) -> dict[str, str]:
    contract = manifest["benchmarks"][benchmark]
    model_revision = manifest["model"]["revision"]
    if benchmark == "RULER":
        upstream_revision = contract["upstream_revision"]
        return {
            "model_revision": model_revision,
            "dataset_revision": upstream_revision,
            "code_revision": upstream_revision,
            "scorer_sha256": sha256(Path(__file__).with_name("run_p3_ruler_matrix.py")),
            "official_scorer_sha256": contract["scorer"]["sha256"],
        }
    dataset_revision = contract["dataset"]["revision"]
    if benchmark == "SCBench":
        scorer_bundle = hashlib.sha256(
            "\n".join(
                sorted(
                    [
                        sha256(Path(__file__).with_name("p3_scbench_metrics.py")),
                        sha256(Path(__file__).with_name("p3_scbench_official.py")),
                        contract["rouge_metric"]["script_sha256"],
                        *contract["upstream_code"]["files_sha256"].values(),
                    ]
                )
            ).encode()
        ).hexdigest()
        return {
            "model_revision": model_revision,
            "dataset_revision": dataset_revision,
            "code_revision": contract["upstream_code"]["revision"],
            "scorer_sha256": scorer_bundle,
            "rouge_revision": contract["rouge_metric"]["revision"],
            "rouge_script_sha256": contract["rouge_metric"]["script_sha256"],
        }
    if benchmark == "LongMemEval":
        scorer_digest = contract["upstream_code"]["files_sha256"][
            "src/evaluation/evaluate_qa.py"
        ]
        code_revision = contract["upstream_code"]["revision"]
    else:
        scorer_digest = sha256(Path(__file__).with_name("p3_natural_metrics.py"))
        code_revision = (
            contract["upstream_code"]["revision"]
            if benchmark == "LongBench-v2"
            else f"dataset-readme@{dataset_revision}"
        )
    return {
        "model_revision": model_revision,
        "dataset_revision": dataset_revision,
        "code_revision": code_revision,
        "scorer_sha256": scorer_digest,
    }


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


def _identifier_digest(identifiers: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()


def _dataset_paths(
    *, benchmark: str, manifest: dict[str, Any], inventory_path: Path
) -> list[Path]:
    inventory = json.loads(inventory_path.read_text())
    observed = inventory.get("benchmarks", {}).get(benchmark)
    _require(isinstance(observed, dict), f"Missing {benchmark} dataset inventory entry.")
    contract = manifest["benchmarks"][benchmark]["dataset"]
    _require(
        observed.get("repo_id") == contract["repo_id"]
        and observed.get("revision") == contract["revision"],
        f"{benchmark} dataset inventory revision drifted.",
    )
    expected = {entry["path"]: entry for entry in contract["files"]}
    observed_files = observed.get("files")
    _require(
        isinstance(observed_files, list) and len(observed_files) == len(expected),
        f"{benchmark} dataset inventory file count drifted.",
    )
    paths: list[Path] = []
    seen: set[str] = set()
    for metadata in observed_files:
        _require(isinstance(metadata, dict), f"{benchmark} dataset file metadata drifted.")
        path = Path(metadata.get("path", ""))
        matches = [
            relative
            for relative in expected
            if path.as_posix() == relative or path.as_posix().endswith(f"/{relative}")
        ]
        _require(len(matches) == 1, f"Unexpected {benchmark} dataset path: {path}.")
        relative = matches[0]
        contract_row = expected.get(relative)
        _require(
            contract_row is not None
            and relative not in seen
            and path.is_file()
            and metadata.get("sha256") == contract_row["sha256"]
            and sha256(path) == contract_row["sha256"],
            f"{benchmark} dataset artifact drifted: {relative}.",
        )
        seen.add(relative)
        paths.append(path)
    _require(seen == set(expected), f"{benchmark} dataset file set drifted.")
    return paths


def expected_example_identifiers(
    *, benchmark: str, manifest: dict[str, Any], inventory_path: Path
) -> set[str]:
    if benchmark == "RULER":
        from prepare_p3_natural_ruler_dataset import LENGTHS, SAMPLES_PER_TASK
        from prepare_p3_ruler_dataset import TASKS

        return {
            f"{length}:{task}:{row_number}"
            for length in LENGTHS
            for task in TASKS
            for row_number in range(SAMPLES_PER_TASK)
        }

    paths = _dataset_paths(
        benchmark=benchmark, manifest=manifest, inventory_path=inventory_path
    )
    if benchmark == "SCBench":
        import pyarrow.parquet as pq

        identifiers: set[str] = set()
        modes = manifest["benchmarks"][benchmark]["modes"]
        for path in paths:
            task = path.parent.name
            rows = pq.read_table(path, columns=["multi_turns"]).column(
                "multi_turns"
            ).to_pylist()
            for mode in modes:
                for row_index, turns in enumerate(rows):
                    for turn_index, _turn in enumerate(turns):
                        identifiers.add(f"{mode}:{task}:{row_index}:{turn_index}")
        return identifiers
    if benchmark == "LongBench-v2":
        payload = json.loads(paths[0].read_text())
        _require(isinstance(payload, list), "LongBench v2 dataset payload drifted.")
        return {str(row["_id"]) for row in payload}
    if benchmark == "LongMemEval":
        payload = json.loads(paths[0].read_text())
        _require(isinstance(payload, list), "LongMemEval dataset payload drifted.")
        return {str(row["question_id"]) for row in payload}
    if benchmark == "MRCR":
        import tiktoken
        from run_p3_mrcr import load_rows

        rows = load_rows(
            paths,
            contract=manifest["benchmarks"][benchmark],
            official_encoder=tiktoken.get_encoding("o200k_base"),
        )
        return {str(row["example_id"]) for row in rows}
    raise ValueError(f"Unknown natural benchmark: {benchmark}")


def _source_root(
    *, benchmark: str, cell: dict[str, Any], manifest: dict[str, Any]
) -> Path:
    metadata = cell.get("evaluation_source_inventory")
    dependency = _dependency(metadata, "evaluation source inventory")
    inventory = json.loads(Path(dependency["path"]).read_text())
    source = inventory.get("benchmarks", {}).get(benchmark)
    contract = manifest["benchmarks"][benchmark]["upstream_code"]
    _require(
        isinstance(source, dict)
        and source.get("revision") == contract["revision"]
        and source.get("clean_tracked_tree") is True,
        f"{benchmark} source inventory drifted.",
    )
    root = Path(source.get("path", ""))
    observed = {row["path"]: row["sha256"] for row in source.get("files", [])}
    _require(
        observed == contract["files_sha256"],
        f"{benchmark} source file inventory drifted.",
    )
    for relative, digest in observed.items():
        path = root / relative
        _require(
            path.is_file() and sha256(path) == digest,
            f"{benchmark} source artifact drifted: {relative}.",
        )
    return root


def build_score_verifier(
    *,
    benchmark: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    cell: dict[str, Any],
) -> ScoreVerifier:
    if benchmark == "RULER":
        from prepare_p3_natural_ruler_dataset import LENGTHS
        from run_p3_natural_ruler import load_dataset_contracts
        from run_p3_ruler_matrix import example_score, load_dataset

        generator = Path(__file__).with_name("prepare_p3_natural_ruler_dataset.py")
        dataset_root = Path(manifest["benchmarks"][benchmark]["execution"]["dataset_root"])
        contracts, observed_set = load_dataset_contracts(
            dataset_root,
            natural_manifest_digest=sha256(manifest_path),
            generator_digest=sha256(generator),
        )
        _require(
            observed_set == cell.get("benchmark_dataset_digest_set_sha256"),
            "RULER score-audit dataset set drifted.",
        )
        ruler_references: dict[str, tuple[str, list[str]]] = {}
        for length in LENGTHS:
            frame = load_dataset(contracts[length])
            for row in frame.to_dict(orient="records"):
                ruler_references[f"{length}:{row['row_id']}"] = (
                    row["task"],
                    row["answer"],
                )

        def verify_ruler(row: dict[str, Any]) -> tuple[float, str | None]:
            task, answers = ruler_references[row["example_id"]]
            return example_score(task, row["raw_response"], answers), row["raw_response"]

        return verify_ruler

    inventory = _dependency(cell.get("dataset_inventory"), "dataset inventory")
    paths = _dataset_paths(
        benchmark=benchmark,
        manifest=manifest,
        inventory_path=Path(inventory["path"]),
    )
    if benchmark == "LongBench-v2":
        from p3_natural_metrics import extract_longbench_v2_choice, score_longbench_v2

        longbench_payload = json.loads(paths[0].read_text())
        answers = {str(row["_id"]): str(row["answer"]) for row in longbench_payload}

        def verify_longbench(row: dict[str, Any]) -> tuple[float, str | None]:
            response = row["raw_response"]
            return score_longbench_v2(response, answers[row["example_id"]]), (
                extract_longbench_v2_choice(response)
            )

        return verify_longbench

    if benchmark == "MRCR":
        import tiktoken
        from p3_natural_metrics import score_mrcr
        from run_p3_mrcr import load_rows

        source_rows = load_rows(
            paths,
            contract=manifest["benchmarks"][benchmark],
            official_encoder=tiktoken.get_encoding("o200k_base"),
        )
        mrcr_references = {
            str(row["example_id"]): (
                str(row["answer"]),
                str(row["random_string_to_prepend"]),
            )
            for row in source_rows
        }

        def verify_mrcr(row: dict[str, Any]) -> tuple[float, str | None]:
            answer, prefix = mrcr_references[row["example_id"]]
            response = row["raw_response"]
            return score_mrcr(response, answer, prefix), response

        return verify_mrcr

    if benchmark == "LongMemEval":
        from run_p3_longmemeval import _load_official_judge_module

        longmem_payload = json.loads(paths[0].read_text())
        longmem_rows = {str(row["question_id"]): row for row in longmem_payload}
        source_root = _source_root(benchmark=benchmark, cell=cell, manifest=manifest)
        scorer_digest = manifest["benchmarks"][benchmark]["upstream_code"][
            "files_sha256"
        ]["src/evaluation/evaluate_qa.py"]
        module_cache: dict[str, Any] = {}

        def verify_longmem(row: dict[str, Any]) -> tuple[float, str | None]:
            module = module_cache.get("official")
            if module is None:
                module = _load_official_judge_module(source_root, scorer_digest)
                module_cache["official"] = module
            source = longmem_rows[row["example_id"]]
            judge = row["judge"]
            expected_prompt = module.get_anscheck_prompt(
                source["question_type"],
                source["question"],
                source["answer"],
                row["raw_response"],
                abstention="_abs" in source["question_id"],
            )
            _require(
                judge["prompt"] == expected_prompt,
                f"LongMemEval official judge prompt drifted: {row['example_id']}.",
            )
            return float("yes" in judge["raw_response"].lower()), row["raw_response"]

        return verify_longmem

    if benchmark == "SCBench":
        import pyarrow.parquet as pq
        from p3_scbench_official import (
            OfficialSCBenchScorer,
            load_repoqa_module,
            load_rouge_lsum,
        )

        rows_by_task = {path.parent.name: pq.read_table(path).to_pylist() for path in paths}
        source_root = _source_root(benchmark=benchmark, cell=cell, manifest=manifest)
        source_digests = manifest["benchmarks"][benchmark]["upstream_code"]["files_sha256"]
        scorer = OfficialSCBenchScorer(
            rows_by_task=rows_by_task,
            repo_module=load_repoqa_module(
                source_root, source_digests["scbench/repo_qa_utils.py"]
            ),
            rouge_metric=load_rouge_lsum(),
        )

        def verify_scbench(row: dict[str, Any]) -> tuple[float, str | None]:
            source = rows_by_task[row["task"]][row["row_index"]]
            score, detail = scorer.score(
                task=row["task"],
                row=source,
                turn=source["multi_turns"][row["turn_index"]],
                prediction=row["raw_response"],
                subtask=row["subtask"],
            )
            _require(
                detail == row["scorer_detail"],
                f"SCBench official scorer trace drifted: {row['example_id']}.",
            )
            return score, row["raw_response"]

        return verify_scbench
    raise ValueError(f"Unknown natural benchmark: {benchmark}")


def _distribution(values: list[float | int]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "observations": len(array),
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def _paired_cluster_bootstrap(
    clusters: dict[str, list[float]], *, label: str, cluster_unit: str
) -> dict[str, Any]:
    _require(bool(clusters), "Paired natural contrast is empty.")
    ordered = [clusters[key] for key in sorted(clusters)]
    cluster_sums = np.asarray([sum(values) for values in ordered], dtype=np.float64)
    cluster_counts = np.asarray([len(values) for values in ordered], dtype=np.float64)
    array = np.asarray([value for values in ordered for value in values], dtype=np.float64)
    _require(bool(np.all(cluster_counts > 0)), "Paired natural cluster is empty.")
    rng = np.random.default_rng(_stable_seed(label))
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    batch_size = 128
    probabilities = np.full(len(ordered), 1.0 / len(ordered), dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        sampled = rng.multinomial(len(ordered), probabilities, size=stop - start)
        means[start:stop] = (sampled @ cluster_sums) / (sampled @ cluster_counts)
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    lower_tail = (np.count_nonzero(means <= 0.0) + 1) / (BOOTSTRAP_RESAMPLES + 1)
    upper_tail = (np.count_nonzero(means >= 0.0) + 1) / (BOOTSTRAP_RESAMPLES + 1)
    cluster_means = cluster_sums / cluster_counts
    standard_deviation = (
        float(cluster_means.std(ddof=1)) if len(cluster_means) > 1 else 0.0
    )
    mean = float(array.mean())
    return {
        "paired_examples": len(array),
        "paired_clusters": len(ordered),
        "cluster_unit": cluster_unit,
        "mean_difference": mean,
        "mean_difference_percentage_points": mean * 100.0,
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "paired_bootstrap_95_ci_percentage_points": [
            float(lower) * 100.0,
            float(upper) * 100.0,
        ],
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "cluster_mean_sample_standard_deviation": standard_deviation,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_seed": _stable_seed(label),
    }


def _arm_records(artifact_path: Path) -> list[dict[str, Any]]:
    artifact = json.loads(artifact_path.read_text())
    return _records(Path(artifact["raw_records"]["path"]))


def _paired_contrasts(
    *, benchmark: str, arm_artifacts: dict[str, Path], required_arms: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(len(required_arms) == 2, "Natural paired contrast requires two baseline arms.")
    comparator, candidate = required_arms
    records = {
        arm: {row["example_id"]: row for row in _arm_records(arm_artifacts[arm])}
        for arm in required_arms
    }
    _require(
        set(records[comparator]) == set(records[candidate]),
        "Natural paired contrast example identities drifted.",
    )
    quality_clusters: dict[str, list[float]] = {}
    cluster_unit = "shared-context-row" if benchmark == "SCBench" else "example"
    measurement_values: dict[str, dict[str, list[float]]] = {
        metric: {comparator: [], candidate: []}
        for metric in ("latency_ms", "peak_hbm_bytes", "hot_resident_bytes")
    }
    jointly_scored = 0
    failure_pairing = {
        "both_scored": 0,
        "candidate_only_failed": 0,
        "comparator_only_failed": 0,
        "both_failed": 0,
    }
    for identifier in sorted(records[comparator]):
        reference = records[comparator][identifier]
        treatment = records[candidate][identifier]
        _require(
            reference["raw_prompt_sha256"] == treatment["raw_prompt_sha256"],
            f"Natural paired prompt drifted: {identifier}",
        )
        reference_scored = reference["status"] == "scored"
        treatment_scored = treatment["status"] == "scored"
        if reference_scored and treatment_scored:
            failure_pairing["both_scored"] += 1
            jointly_scored += 1
        elif reference_scored:
            failure_pairing["candidate_only_failed"] += 1
        elif treatment_scored:
            failure_pairing["comparator_only_failed"] += 1
        else:
            failure_pairing["both_failed"] += 1
        reference_score = float(reference["score"]) if reference_scored else 0.0
        treatment_score = float(treatment["score"]) if treatment_scored else 0.0
        difference = treatment_score - reference_score
        if benchmark == "SCBench":
            cluster_id = (
                f'{reference["mode"]}:{reference["task"]}:{reference["row_index"]}'
            )
        else:
            cluster_id = identifier
        quality_clusters.setdefault(cluster_id, []).append(difference)
        for metric in measurement_values:
            measurement_values[metric][comparator].append(float(reference[metric]))
            measurement_values[metric][candidate].append(float(treatment[metric]))
    quality = {
        "candidate": candidate,
        "comparator": comparator,
        "failure_as_zero": True,
        "jointly_scored_examples": jointly_scored,
        "failure_pairing": failure_pairing,
        **_paired_cluster_bootstrap(
            quality_clusters,
            label=f"p3-natural:{benchmark}:quality",
            cluster_unit=cluster_unit,
        ),
    }
    measurements: dict[str, Any] = {}
    for metric, by_arm in measurement_values.items():
        comparator_values = np.asarray(by_arm[comparator], dtype=np.float64)
        candidate_values = np.asarray(by_arm[candidate], dtype=np.float64)
        comparator_mean = float(comparator_values.mean())
        candidate_mean = float(candidate_values.mean())
        measurements[metric] = {
            "paired_examples": len(candidate_values),
            "candidate": candidate,
            "comparator": comparator,
            "candidate_mean": candidate_mean,
            "comparator_mean": comparator_mean,
            "mean_paired_difference": float((candidate_values - comparator_values).mean()),
            "ratio_of_means": (
                candidate_mean / comparator_mean if comparator_mean > 0.0 else None
            ),
            "includes_terminal_failures": True,
        }
    return quality, measurements


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
    expected_revisions: dict[str, str] | None = None,
    expected_seed: int = 42,
    expected_identifiers: set[str] | None = None,
    score_verifier: ScoreVerifier | None = None,
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
    run_identity = artifact.get("run_identity")
    _require(isinstance(run_identity, dict), f"Missing {benchmark}/{arm} run identity.")
    assert isinstance(run_identity, dict)
    identity_contract = {
        "source_commit": artifact["source"]["commit"],
        "implementation_sha256": source_implementation["implementation_sha256"],
        "manifest_sha256": manifest_digest,
        "inventory_sha256": artifact.get("dataset_inventory", {}).get("sha256"),
        "causal_gate_sha256": artifact.get("causal_gate", {}).get("sha256"),
        "fixed_selection_sha256": artifact.get("fixed_baseline_selection", {}).get("sha256"),
        "model_snapshot_digest_set_sha256": artifact.get(
            "model_snapshot_digest_set_sha256"
        ),
        "seed": expected_seed,
    }
    _require(
        all(run_identity.get(key) == value for key, value in identity_contract.items()),
        f"{benchmark}/{arm} run identity drifted.",
    )
    identity_arm_config = run_identity.get("arm_config")
    _require(
        isinstance(identity_arm_config, dict) and bool(identity_arm_config),
        f"{benchmark}/{arm} run arm config is missing.",
    )
    if benchmark == "RULER":
        _require(
            run_identity.get("dataset_manifest_digest_set_sha256")
            == artifact.get("benchmark_dataset_digest_set_sha256"),
            f"{benchmark}/{arm} dataset run identity drifted.",
        )
    if benchmark == "SCBench" and expected_revisions is not None:
        _require(
            run_identity.get("scorer_bundle_sha256")
            == expected_revisions["scorer_sha256"],
            f"{benchmark}/{arm} scorer run identity drifted.",
        )
    if benchmark == "LongMemEval":
        _require(
            run_identity.get("judge_mode") in {"blocked", "openai"},
            f"{benchmark}/{arm} judge-mode run identity drifted.",
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
    scores_recomputed = 0
    score_sum = 0.0
    measurement_values: dict[str, list[float | int]] = {
        "exact_input_tokens": [],
        "latency_ms": [],
        "peak_hbm_bytes": [],
        "hot_resident_bytes": [],
    }
    scored_measurement_values: dict[str, list[float | int]] = {
        key: [] for key in measurement_values
    }
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
            _positive_integer(row.get("exact_input_tokens"))
            and _positive_integer(row.get("generation_reserve_tokens")),
            f"Invalid token accounting: {identifier}",
        )
        _sha256_value(row.get("raw_prompt_sha256"), f"{identifier} prompt")
        paired_input_digests.append(f"{identifier}:{row['raw_prompt_sha256']}")
        _require(
            _finite_nonnegative_number(row.get("latency_ms")),
            f"Invalid latency: {identifier}",
        )
        _require(
            _nonnegative_integer(row.get("peak_hbm_bytes")),
            f"Invalid HBM accounting: {identifier}",
        )
        _require(
            _nonnegative_integer(row.get("hot_resident_bytes")),
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
        if expected_revisions is not None:
            _require(
                revisions == expected_revisions,
                f"Frozen record revisions drifted: {identifier}",
            )
        _require(
            row.get("arm_config") == identity_arm_config,
            f"Record arm config drifted: {identifier}",
        )
        for metric in measurement_values:
            measurement_values[metric].append(row[metric])
        if benchmark in {"RULER", "SCBench", "LongBench-v2", "LongMemEval", "MRCR"}:
            _require(
                _nonnegative_integer(row.get("token_boundary_retreat")),
                f"Invalid exact-token boundary accounting: {identifier}",
            )
            _require(
                row["token_boundary_retreat"] <= row["exact_input_tokens"],
                f"Impossible exact-token boundary accounting: {identifier}",
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
                and _nonnegative_integer(row.get("row_index"))
                and _nonnegative_integer(row.get("turn_index")),
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
                    and _nonnegative_integer(judge.get("created"))
                    and isinstance(judge.get("prompt"), str)
                    and bool(judge["prompt"])
                    and isinstance(judge.get("raw_response"), str)
                    and bool(judge["raw_response"])
                    and _finite_nonnegative_number(judge.get("latency_ms")),
                    f"Incomplete official judge provenance: {identifier}",
                )
            elif row.get("failure_type") == "judge-blocked":
                _require(
                    isinstance(judge, dict)
                    and judge.get("model") == "gpt-4o-2024-08-06"
                    and judge.get("status") == "blocked"
                    and isinstance(judge.get("reason"), str)
                    and bool(judge["reason"])
                    and _nonnegative_integer(row.get("generated_tokens_observed")),
                    f"Incomplete blocked-judge provenance: {identifier}",
                )
        if row["status"] == "scored":
            score = row.get("score")
            if not _unit_interval_number(score):
                raise ValueError(f"Invalid score: {identifier}")
            score_value = cast(int | float, score)
            _require(row.get("failure_type") is None, f"Scored record has failure: {identifier}")
            _require(
                bool(row["raw_response"].strip())
                and _positive_integer(row.get("generated_tokens_observed"))
                and row.get("stop_reason") in {"max-new-tokens", "eos-or-special-token"},
                f"Incomplete scored generation evidence: {identifier}",
            )
            if score_verifier is not None:
                recomputed_score, recomputed_parsed = score_verifier(row)
                _require(
                    _unit_interval_number(recomputed_score)
                    and math.isclose(
                        float(score_value),
                        float(recomputed_score),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    f"Reported natural score drifted from raw response: {identifier}",
                )
                _require(
                    row.get("parsed_response") == recomputed_parsed,
                    f"Parsed natural response drifted: {identifier}",
                )
                scores_recomputed += 1
            scored += 1
            score_sum += float(score_value)
            for metric in scored_measurement_values:
                scored_measurement_values[metric].append(row[metric])
        else:
            failure = row.get("failure_type")
            if not isinstance(failure, str) or failure not in allowed_failures:
                raise ValueError(f"Unregistered failure: {failure}")
            _require(row.get("score") is None, f"Failed record has a score: {identifier}")
            generated = row.get("generated_tokens_observed")
            _require(
                generated is None or _positive_integer(generated),
                f"Invalid failed generation evidence: {identifier}",
            )
            failures[failure] = failures.get(failure, 0) + 1
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        record_digests.append(hashlib.sha256(canonical.encode()).hexdigest())
    _require(scored + sum(failures.values()) == expected_examples, "Arm accounting does not close.")
    if expected_identifiers is not None:
        _require(
            identifiers == expected_identifiers,
            f"{benchmark}/{arm} example identities drifted from the frozen dataset.",
        )
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
    source_inventory_metadata = artifact.get("evaluation_source_inventory")
    if source_inventory_metadata is not None:
        source_inventory = _dependency(source_inventory_metadata, "evaluation source inventory")
        _require(
            run_identity.get("source_inventory_sha256") == source_inventory["sha256"],
            f"{benchmark}/{arm} source-inventory run identity drifted.",
        )
        dependencies["evaluation_source_inventory"] = source_inventory
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
            "failure_rate": sum(failures.values()) / expected_examples,
            "measurements": {
                "all_terminal_attempts": {
                    key: _distribution(values) for key, values in measurement_values.items()
                },
                "scored_only": {
                    key: _distribution(values)
                    for key, values in scored_measurement_values.items()
                },
            },
            "paired_example_prompt_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(paired_input_digests)).encode()
            ).hexdigest(),
            "raw_cell": {"path": str(artifact_path), "sha256": sha256(artifact_path)},
            "raw_record_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(record_digests)).encode()
            ).hexdigest(),
            "source_implementation": source_implementation,
            "run_identity_verified": True,
            "terminal_measurement_schema_verified": True,
            "dataset_example_identities_verified": expected_identifiers is not None,
            "expected_example_identity_set_sha256": _identifier_digest(identifiers),
            "scores_recomputed_from_raw_response": score_verifier is not None
            and scores_recomputed == scored,
            "scores_recomputed": scores_recomputed,
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
    expected_revisions = expected_record_revisions(benchmark, manifest)
    expected_seed = manifest["benchmarks"][benchmark]["generation_seed"]
    first_cell = json.loads(arm_artifacts[required[0]].read_text())
    inventory_metadata = first_cell.get("dataset_inventory")
    inventory = _dependency(inventory_metadata, "dataset inventory")
    expected_identifiers = expected_example_identifiers(
        benchmark=benchmark,
        manifest=manifest,
        inventory_path=Path(inventory["path"]),
    )
    _require(
        len(expected_identifiers) == expected,
        f"{benchmark} frozen dataset identity count drifted.",
    )
    score_verifier = build_score_verifier(
        benchmark=benchmark,
        manifest=manifest,
        manifest_path=manifest_path,
        cell=first_cell,
    )
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
            expected_revisions=expected_revisions,
            expected_seed=expected_seed,
            expected_identifiers=expected_identifiers,
            score_verifier=score_verifier,
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
    paired_quality, paired_measurements = _paired_contrasts(
        benchmark=benchmark,
        arm_artifacts=arm_artifacts,
        required_arms=required,
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
        "generation_seed": expected_seed,
        "experiment_manifest": {"path": str(manifest_path), "sha256": manifest_digest},
        "audit": {
            "all_raw_artifacts_verified": True,
            "all_failure_accounting_complete": True,
            "all_required_arms_input_paired": True,
            "all_source_implementations_verified": True,
            "all_record_revisions_verified": True,
            "all_run_identities_verified": True,
            "all_terminal_measurement_schema_verified": True,
            "all_dataset_example_identities_verified": True,
            "all_reported_scores_recomputed_from_raw_response": True,
            "reported_scores_recomputed": sum(
                row["scores_recomputed"] for row in arms.values()
            ),
            "expected_example_identity_set_sha256": _identifier_digest(
                expected_identifiers
            ),
            "raw_record_digest_set_sha256": hashlib.sha256(
                "\n".join(
                    sorted(row["raw_record_digest_set_sha256"] for row in arms.values())
                ).encode()
            ).hexdigest(),
        },
        "arms": arms,
        "paired_quality_contrast": paired_quality,
        "paired_measurement_contrasts": paired_measurements,
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
