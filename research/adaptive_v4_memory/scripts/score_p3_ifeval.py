from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from p3_source_provenance import verify_git_implementation
from prepare_p3_natural_safety_assets import (
    sha256,
    tree_sha256,
    validate_ifeval_rows,
    verify_file,
)
from validate_p3_natural_safety_manifest import validate_manifest

ARMS = ("native-dense", "strongest-memory-matched-fixed")
GENERATION_RUNNER_PATH = "research/adaptive_v4_memory/scripts/run_p3_natural_safety_generation.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def score_arm(
    *, inputs: list[dict[str, Any]], records: list[dict[str, Any]], official: Any
) -> list[dict[str, Any]]:
    by_key = {record.get("source_id"): record for record in records}
    _require(
        len(by_key) == len(records) == len(inputs),
        "IFEval generation records are missing or duplicated.",
    )
    outputs: list[dict[str, Any]] = []
    for row in inputs:
        key = row["key"]
        _require(key in by_key, f"IFEval generation is missing key {key}.")
        record = by_key[key]
        response = record.get("raw_response") if record.get("status") == "generated" else ""
        _require(isinstance(response, str), f"IFEval response is not text: {key}.")
        assert isinstance(response, str)
        inp = official.InputExample(
            key=key,
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        try:
            strict = official.test_instruction_following_strict(inp, {row["prompt"]: response})
            loose = official.test_instruction_following_loose(inp, {row["prompt"]: response})
            _require(
                len(strict.follow_instruction_list)
                == len(loose.follow_instruction_list)
                == len(row["instruction_id_list"]),
                f"IFEval official instruction result length drifted: {key}.",
            )
            outputs.append(
                {
                    "source_id": key,
                    "example_id": record["example_id"],
                    "generation_status": record["status"],
                    "generation_failure_type": record.get("failure_type"),
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "instruction_id_list": row["instruction_id_list"],
                    "strict_follow_instruction_list": list(strict.follow_instruction_list),
                    "strict_follow_all_instructions": bool(strict.follow_all_instructions),
                    "loose_follow_instruction_list": list(loose.follow_instruction_list),
                    "loose_follow_all_instructions": bool(loose.follow_all_instructions),
                    "scorer_status": "scored",
                    "scorer_failure_type": None,
                }
            )
        except Exception as error:
            outputs.append(
                {
                    "source_id": key,
                    "example_id": record["example_id"],
                    "generation_status": record["status"],
                    "generation_failure_type": record.get("failure_type"),
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "instruction_id_list": row["instruction_id_list"],
                    "strict_follow_instruction_list": [False] * len(row["instruction_id_list"]),
                    "strict_follow_all_instructions": False,
                    "loose_follow_instruction_list": [False] * len(row["instruction_id_list"]),
                    "loose_follow_all_instructions": False,
                    "scorer_status": "failure",
                    "scorer_failure_type": "official-scorer-error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return outputs


def aggregate(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(outputs), "IFEval official outputs are empty.")
    instruction_total = sum(len(row["instruction_id_list"]) for row in outputs)
    _require(instruction_total > 0, "IFEval contains no instructions.")
    return {
        "expected_prompts": len(outputs),
        "scored_prompts": sum(row["scorer_status"] == "scored" for row in outputs),
        "scorer_failures": sum(row["scorer_status"] == "failure" for row in outputs),
        "generation_failures": sum(row["generation_status"] != "generated" for row in outputs),
        "prompt_level_strict_accuracy": sum(
            row["strict_follow_all_instructions"] for row in outputs
        )
        / len(outputs),
        "instruction_level_strict_accuracy": sum(
            sum(row["strict_follow_instruction_list"]) for row in outputs
        )
        / instruction_total,
        "prompt_level_loose_accuracy": sum(row["loose_follow_all_instructions"] for row in outputs)
        / len(outputs),
        "instruction_level_loose_accuracy": sum(
            sum(row["loose_follow_instruction_list"]) for row in outputs
        )
        / instruction_total,
        "instruction_total": instruction_total,
    }


def paired_effect(
    fixed: list[dict[str, Any]], native: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    fixed_map = {row["source_id"]: row for row in fixed}
    native_map = {row["source_id"]: row for row in native}
    _require(set(fixed_map) == set(native_map), "IFEval paired identities diverged.")
    differences = np.array(
        [
            float(fixed_map[key]["strict_follow_all_instructions"])
            - float(native_map[key]["strict_follow_all_instructions"])
            for key in sorted(fixed_map)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bootstrap[index] = float(
            np.mean(differences[rng.integers(0, len(differences), len(differences))])
        )
    return {
        "estimand": "strongest-memory-matched-fixed minus native-dense prompt-level strict accuracy",
        "paired_prompts": len(differences),
        "mean_difference": float(np.mean(differences)),
        "paired_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "losses": int(np.sum(differences < 0)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def _records(
    cell_path: Path,
    arm: str,
    expected: int,
    *,
    manifest_digest: str,
    inventory_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell = json.loads(cell_path.read_text())
    _require(
        cell.get("experiment_id") == "p3-natural-safety-generation-arm-cell-v1"
        and cell.get("benchmark") == "IFEval"
        and cell.get("arm") == arm
        and cell.get("status") == "terminal"
        and cell.get("source", {}).get("dirty") is False
        and cell.get("expected_generations") == expected,
        f"IFEval generation cell is invalid: {arm}.",
    )
    _require(
        cell.get("manifest", {}).get("sha256") == manifest_digest
        and cell.get("asset_inventory", {}).get("sha256") == inventory_digest,
        f"IFEval generation provenance drifted: {arm}.",
    )
    cell["verified_source_implementation"] = verify_git_implementation(
        cell.get("source"),
        expected_path=GENERATION_RUNNER_PATH,
        label=f"IFEval/{arm}",
    )
    path = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        path.is_file() and cell["raw_records"]["sha256"] == sha256(path), "IFEval records drifted."
    )
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    _require(len(records) == expected, f"IFEval record count drifted: {arm}.")
    _require(
        all(
            record.get("benchmark") == "IFEval"
            and record.get("arm") == arm
            and _is_sha256(record.get("raw_prompt_sha256"))
            and _is_sha256(record.get("input_token_ids_sha256"))
            for record in records
        ),
        f"IFEval record provenance drifted: {arm}.",
    )
    return records, cell


def _official_module(source_root: Path, contract: dict[str, Any]) -> Any:
    for entry in contract["upstream_code"]["files"]:
        verify_file(source_root / entry["path"], entry)
    for name in tuple(sys.modules):
        if name == "instruction_following_eval" or name.startswith("instruction_following_eval."):
            del sys.modules[name]
    sys.path.insert(0, str(source_root))
    try:
        return importlib.import_module("instruction_following_eval.evaluation_lib")
    finally:
        sys.path.pop(0)


def configure_nltk_runtime(
    *, inventory: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    expected_runtime = contract["runtime_requirements"]
    observed_runtime = inventory["runtime_requirements"]
    _require(
        observed_runtime.get("packages") == expected_runtime["packages"],
        "IFEval package runtime contract drifted.",
    )
    expected_data = expected_runtime["nltk_data"]
    observed_data = observed_runtime["nltk_data"]
    _require(
        observed_data.get("repository") == expected_data["repository"]
        and observed_data.get("revision") == expected_data["revision"],
        "IFEval NLTK data revision drifted.",
    )
    observed_archives = observed_data.get("archives", [])
    _require(
        isinstance(observed_archives, list)
        and len(observed_archives) == len(expected_data["files"]),
        "IFEval NLTK archive inventory drifted.",
    )
    for entry in expected_data["files"]:
        match = next(
            (
                observed
                for observed in observed_archives
                if Path(observed.get("path", "")).as_posix().endswith(entry["path"])
            ),
            None,
        )
        _require(match is not None, f"IFEval NLTK archive is missing: {entry['path']}.")
        assert match is not None
        verify_file(Path(match["path"]), entry)
    data_root = Path(observed_data.get("data_root", {}).get("path", ""))
    observed_tree = tree_sha256(data_root)
    _require(
        observed_tree["files"] == expected_data["extracted_file_count"]
        and observed_tree["sha256"] == expected_data["extracted_tree_sha256"]
        and observed_data["data_root"].get("files") == observed_tree["files"]
        and observed_data["data_root"].get("sha256") == observed_tree["sha256"],
        "IFEval NLTK extracted tree drifted.",
    )
    nltk = importlib.import_module("nltk")
    _require(
        importlib.metadata.version("nltk") == "3.10.0",
        "IFEval requires the frozen nltk==3.10.0 runtime.",
    )
    nltk.data.path[:] = [str(data_root.resolve())]
    nltk.tokenize._get_punkt_tokenizer.cache_clear()
    nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    nltk.word_tokenize("Frozen IFEval tokenizer preflight.")
    return {
        "repository": expected_data["repository"],
        "revision": expected_data["revision"],
        "data_root": str(data_root.resolve()),
        "files": observed_tree["files"],
        "tree_sha256": observed_tree["sha256"],
        "preflight": "passed",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def runtime_environment(nltk_data: dict[str, Any]) -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("absl-py", "immutabledict", "langdetect", "nltk")
        },
        "nltk_data": nltk_data,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score P3 IFEval with pinned official code.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    )
    parser.add_argument(
        "--asset-inventory",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/assets/inventory.json"
        ),
    )
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/ifeval"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/ifeval.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "IFEval scoring requires a clean source tree.")
    raw_manifest = args.manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    validation = validate_manifest(manifest)
    contract = manifest["benchmarks"]["IFEval"]
    expected = contract["protocol"]["expected_prompts_per_arm"]
    inventory = json.loads(args.asset_inventory.read_text())
    manifest_digest = hashlib.sha256(raw_manifest).hexdigest()
    inventory_digest = sha256(args.asset_inventory)
    _require(
        inventory.get("experiment_id") == "p3-natural-safety-asset-inventory-v1"
        and inventory.get("status") == "verified"
        and inventory.get("source", {}).get("dirty") is False
        and inventory.get("manifest", {}).get("sha256") == manifest_digest,
        "IFEval asset inventory drifted.",
    )
    dataset = inventory["benchmarks"]["IFEval"]["dataset"]
    row_entry = next(entry for entry in contract["dataset"]["files"] if "rows" in entry)
    row_path = Path(
        next(
            entry["path"]
            for entry in dataset["files"]
            if Path(entry["path"]).name == row_entry["path"]
        )
    )
    verify_file(row_path, row_entry)
    inputs = [json.loads(line) for line in row_path.read_text().splitlines() if line]
    validate_ifeval_rows(
        inputs,
        expected_rows=expected,
        required_fields=contract["dataset"]["required_fields"],
    )
    source_entry = inventory["benchmarks"]["IFEval"]["upstream_code"]["files"]
    eval_path = Path(
        next(entry["path"] for entry in source_entry if entry["path"].endswith("evaluation_lib.py"))
    )
    source_root = eval_path.parents[1]
    nltk_runtime = configure_nltk_runtime(
        inventory=inventory["benchmarks"]["IFEval"], contract=contract
    )
    official = _official_module(source_root, contract)
    generation: dict[str, list[dict[str, Any]]] = {}
    cells: dict[str, Any] = {}
    for arm in ARMS:
        generation[arm], cells[arm] = _records(
            args.generation_root / arm / "cell.json",
            arm,
            expected,
            manifest_digest=manifest_digest,
            inventory_digest=inventory_digest,
        )
    _require(
        all(
            generation[ARMS[0]][index]["source_id"] == generation[ARMS[1]][index]["source_id"]
            and generation[ARMS[0]][index]["raw_prompt_sha256"]
            == generation[ARMS[1]][index]["raw_prompt_sha256"]
            and generation[ARMS[0]][index]["input_token_ids_sha256"]
            == generation[ARMS[1]][index]["input_token_ids_sha256"]
            for index in range(expected)
        ),
        "IFEval generation arms are not input paired.",
    )
    _require(
        len(
            {
                json.dumps(cells[arm]["verified_source_implementation"], sort_keys=True)
                for arm in ARMS
            }
        )
        == 1,
        "IFEval generation arms used different source implementations.",
    )
    scored = {
        arm: score_arm(inputs=inputs, records=generation[arm], official=official) for arm in ARMS
    }
    output_root = args.output.parent / "ifeval-official"
    raw_outputs: dict[str, Any] = {}
    for arm in ARMS:
        path = output_root / arm / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in scored[arm]))
        temporary.replace(path)
        raw_outputs[arm] = {"path": str(path), "sha256": sha256(path)}
    statistics = manifest["statistics"]
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-safety-ifeval-official-audit-v1",
        "source": {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "dirty": False,
            "implementation_sha256": sha256(Path(__file__)),
        },
        "manifest": {
            "path": str(args.manifest),
            "sha256": manifest_digest,
            "validation": validation,
        },
        "asset_inventory": {"path": str(args.asset_inventory), "sha256": inventory_digest},
        "generation_cells": {
            arm: {
                "path": str(args.generation_root / arm / "cell.json"),
                "sha256": sha256(args.generation_root / arm / "cell.json"),
            }
            for arm in ARMS
        },
        "official_source_revision": contract["upstream_code"]["revision"],
        "environment": runtime_environment(nltk_runtime),
        "input_pairing_verified": True,
        "audit": {
            "required_arms_terminal": True,
            "input_pairing_verified": True,
            "official_scoring_accounted": True,
            "source_implementations_verified": True,
            "expected_prompts_per_arm": expected,
        },
        "arms": {
            arm: {"metrics": aggregate(scored[arm]), "raw_official_results": raw_outputs[arm]}
            for arm in ARMS
        },
        "paired_fixed_minus_native": paired_effect(
            scored[ARMS[1]],
            scored[ARMS[0]],
            seed=9171403,
            replicates=statistics["paired_bootstrap_replicates"],
        ),
        "claim_boundary": contract["protocol"]["boundary"],
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "expected_prompts": expected}, sort_keys=True))


if __name__ == "__main__":
    main()
