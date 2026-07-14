from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_ORDER = ("RULER", "SCBench", "LongBench-v2", "LongMemEval", "MRCR")
EXPECTED_RULER_LENGTHS = (8192, 16384, 32768, 65536, 131072)
EXPECTED_SCBENCH_TASKS = (
    "scbench_choice_eng",
    "scbench_kv",
    "scbench_many_shot",
    "scbench_mf",
    "scbench_prefix_suffix",
    "scbench_qa_chn",
    "scbench_qa_eng",
    "scbench_repoqa",
    "scbench_repoqa_and_kv",
    "scbench_summary",
    "scbench_summary_with_needles",
    "scbench_vt",
)
EXPECTED_REVISIONS = {
    "model": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "SCBench-data": "283310bb8c5ba6909dd9a6b1be087d2937f76f6d",
    "SCBench-code": "a4eb395f949ea39e871f9bc586d683390692c6be",
    "LongBench-v2-data": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    "LongBench-v2-code": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
    "LongMemEval-data": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
    "LongMemEval-code": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
    "MRCR-data": "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
}


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest.")
    int(value, 16)


def _benchmark(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload["benchmarks"].get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing benchmark contract for {name}.")
    return value


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("Natural-suite schema_version must be 1.")
    if payload.get("status") != "frozen_before_execution":
        raise ValueError("Natural-suite status must remain frozen_before_execution.")
    if tuple(payload.get("execution_order", ())) != EXPECTED_ORDER:
        raise ValueError("Natural benchmark order drifted from the preregistration.")
    if payload["model"]["revision"] != EXPECTED_REVISIONS["model"]:
        raise ValueError("The 128K-compatible model revision drifted.")
    if payload["model"]["maximum_supported_context_tokens"] < 131072:
        raise ValueError("The primary model does not cover the 128K protocol point.")
    if payload["common_protocol"]["overflow_action"] != "report_unsupported_without_truncation":
        raise ValueError("Natural evaluation must not silently truncate prompts.")

    ruler = _benchmark(payload, "RULER")
    if tuple(ruler["lengths_tokens"]) != EXPECTED_RULER_LENGTHS:
        raise ValueError("RULER must retain every preregistered 8K-128K length.")
    if ruler["samples_per_task"] != 500 or ruler["task_count"] != 13:
        raise ValueError("RULER sample or task count drifted.")

    scbench = _benchmark(payload, "SCBench")
    if scbench["dataset"]["revision"] != EXPECTED_REVISIONS["SCBench-data"]:
        raise ValueError("SCBench dataset revision drifted.")
    if scbench["upstream_code"]["revision"] != EXPECTED_REVISIONS["SCBench-code"]:
        raise ValueError("SCBench code revision drifted.")
    tasks = scbench["tasks"]
    if tuple(sorted(tasks)) != EXPECTED_SCBENCH_TASKS:
        raise ValueError("SCBench must include all twelve frozen tasks.")
    if tuple(scbench["modes"]) != ("multi-turn", "multi-request"):
        raise ValueError("SCBench must run both shared-context modes.")
    if sum(task["rows"] for task in tasks.values()) != 922:
        raise ValueError("SCBench row total must be 922.")
    if sum(task["turns"] for task in tasks.values()) != 5143:
        raise ValueError("SCBench turn total must be 5,143 per mode.")
    if scbench["expected_predictions_per_arm"] != 10286:
        raise ValueError("SCBench prediction total must include both modes.")

    longbench = _benchmark(payload, "LongBench-v2")
    if longbench["dataset"]["revision"] != EXPECTED_REVISIONS["LongBench-v2-data"]:
        raise ValueError("LongBench v2 dataset revision drifted.")
    if longbench["upstream_code"]["revision"] != EXPECTED_REVISIONS["LongBench-v2-code"]:
        raise ValueError("LongBench v2 code revision drifted.")
    if longbench["expected_examples"] != 503:
        raise ValueError("LongBench v2 must retain all 503 examples.")
    if longbench["overflow_action"] != "report_unsupported_without_truncation":
        raise ValueError("LongBench v2 cannot use the upstream head-tail truncation path.")

    longmem = _benchmark(payload, "LongMemEval")
    if longmem["dataset"]["revision"] != EXPECTED_REVISIONS["LongMemEval-data"]:
        raise ValueError("LongMemEval dataset revision drifted.")
    if longmem["upstream_code"]["revision"] != EXPECTED_REVISIONS["LongMemEval-code"]:
        raise ValueError("LongMemEval code revision drifted.")
    if longmem["expected_examples"] != 500:
        raise ValueError("LongMemEval_S must retain all 500 questions.")
    if longmem["official_judge"]["model"] != "gpt-4o-2024-08-06":
        raise ValueError("LongMemEval official judge revision must be explicit.")

    mrcr = _benchmark(payload, "MRCR")
    if mrcr["dataset"]["revision"] != EXPECTED_REVISIONS["MRCR-data"]:
        raise ValueError("MRCR dataset revision drifted.")
    if tuple(mrcr["needle_counts"]) != (2, 4, 8):
        raise ValueError("MRCR must retain 2-, 4-, and 8-needle variants.")
    if mrcr["samples_per_bin_per_needle_count"] != 100:
        raise ValueError("MRCR must retain 100 samples per bin and needle count.")
    if len(mrcr["bin_boundaries_tokens"]) != 8:
        raise ValueError("MRCR must record all eight official bins.")
    if mrcr["expected_examples_all_bins"] != 2400:
        raise ValueError("MRCR all-bin total must be 2,400.")
    if mrcr["expected_examples_through_128k"] != 1500:
        raise ValueError("MRCR <=128K total must be 1,500.")

    all_data_files: list[dict[str, Any]] = []
    for benchmark in (scbench, longbench, longmem, mrcr):
        all_data_files.extend(benchmark["dataset"]["files"])
    for entry in all_data_files:
        _require_sha256(entry["sha256"], entry["path"])
        if entry["bytes"] <= 0 or entry["rows"] <= 0:
            raise ValueError(f"Invalid file inventory entry: {entry['path']}.")
    expected_bytes = sum(entry["bytes"] for entry in all_data_files)
    if payload["execution_totals"]["dataset_files_to_verify"] != len(all_data_files):
        raise ValueError("Dataset file total drifted from the frozen inventory.")
    if payload["execution_totals"]["dataset_bytes_to_acquire"] != expected_bytes:
        raise ValueError("Dataset byte total drifted from the frozen inventory.")
    if payload["execution_totals"]["minimum_predictions_per_arm"] != 45289:
        raise ValueError("Natural-suite primary prediction total drifted.")

    return {
        "benchmarks": list(EXPECTED_ORDER),
        "model_revision": payload["model"]["revision"],
        "scbench_contexts": 922,
        "scbench_turns_per_mode": 5143,
        "longbench_v2_examples": 503,
        "longmemeval_examples": 500,
        "mrcr_examples_all_bins": 2400,
        "mrcr_examples_through_128k": 1500,
        "dataset_files": len(all_data_files),
        "dataset_bytes": expected_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen P3 natural suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    payload = json.loads(raw)
    summary = validate_manifest(payload)
    summary["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
