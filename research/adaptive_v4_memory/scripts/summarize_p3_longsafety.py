from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_p3_natural_safety_assets import sha256
from validate_p3_natural_safety_manifest import validate_manifest

ARMS = ("native-dense", "strongest-memory-matched-fixed")
POSITIONS = ("front", "end")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def audit_arm(
    *,
    arm: str,
    cell_path: Path,
    expected: int,
    failures: set[str],
    manifest_digest: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cell = json.loads(cell_path.read_text())
    _require(
        cell.get("experiment_id") == "p3-natural-safety-generation-arm-cell-v1"
        and cell.get("benchmark") == "LongSafety"
        and cell.get("arm") == arm
        and cell.get("status") == "terminal"
        and cell.get("source", {}).get("dirty") is False
        and cell.get("expected_generations") == expected,
        f"LongSafety generation cell is invalid: {arm}.",
    )
    inventory_digest = cell.get("asset_inventory", {}).get("sha256")
    _require(
        cell.get("manifest", {}).get("sha256") == manifest_digest
        and _is_sha256(inventory_digest),
        f"LongSafety generation provenance drifted: {arm}.",
    )
    records_path = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records_path.is_file() and cell["raw_records"]["sha256"] == sha256(records_path),
        f"LongSafety raw records drifted: {arm}.",
    )
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    _require(len(records) == expected, f"LongSafety record count drifted: {arm}.")
    by_id: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    failures_by_type: dict[str, int] = defaultdict(int)
    source_positions: dict[int, set[str]] = defaultdict(set)
    for record in records:
        identifier = record.get("example_id")
        source_id = record.get("source_id")
        position = record.get("prompt_position")
        metadata = record.get("metadata")
        _require(
            isinstance(identifier, str)
            and identifier not in by_id
            and isinstance(source_id, int)
            and position in POSITIONS
            and isinstance(metadata, dict)
            and isinstance(metadata.get("safety_type"), str)
            and isinstance(metadata.get("task_type"), str)
            and record.get("benchmark") == "LongSafety"
            and record.get("arm") == arm
            and _is_sha256(record.get("raw_prompt_sha256"))
            and _is_sha256(record.get("input_token_ids_sha256")),
            f"LongSafety record coordinates drifted: {identifier}.",
        )
        assert isinstance(identifier, str) and isinstance(source_id, int)
        by_id[identifier] = record
        source_positions[source_id].add(position)
        status = record.get("status")
        if status == "generated":
            _require(
                record.get("failure_type") is None
                and isinstance(record.get("raw_response"), str)
                and bool(record["raw_response"].strip()),
                f"LongSafety generated record drifted: {identifier}.",
            )
        else:
            failure = record.get("failure_type")
            _require(
                status == "failure" and isinstance(failure, str) and failure in failures,
                f"LongSafety failure is unregistered: {identifier}.",
            )
            assert isinstance(failure, str)
            failures_by_type[failure] += 1
        groups[(position, metadata["safety_type"], metadata["task_type"])].append(record)
    _require(
        len(source_positions) * len(POSITIONS) == expected
        and all(positions == set(POSITIONS) for positions in source_positions.values()),
        f"LongSafety front/end coverage drifted: {arm}.",
    )
    slices = [
        {
            "prompt_position": key[0],
            "safety_type": key[1],
            "task_type": key[2],
            "expected_generations": len(rows),
            "generated": sum(row.get("status") == "generated" for row in rows),
            "failures": sum(row.get("status") == "failure" for row in rows),
            "judge_status": "blocked",
            "safety_rate": None,
        }
        for key, rows in sorted(groups.items())
    ]
    return (
        {
            "terminal_generation": True,
            "expected_generations": expected,
            "generated": sum(row.get("status") == "generated" for row in records),
            "failures_by_type": dict(sorted(failures_by_type.items())),
            "prompt_positions": list(POSITIONS),
            "source_examples": len(source_positions),
            "asset_inventory_sha256": inventory_digest,
            "slices": slices,
            "raw_cell": {"path": str(cell_path), "sha256": sha256(cell_path)},
        },
        by_id,
    )


def summarize(manifest_path: Path, arm_paths: dict[str, Path]) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    validation = validate_manifest(manifest)
    contract = manifest["benchmarks"]["LongSafety"]
    expected = contract["prompt_protocol"]["expected_predictions_per_arm"]
    _require(set(arm_paths) == set(ARMS), "LongSafety audit arm set drifted.")
    arms: dict[str, Any] = {}
    records: dict[str, dict[str, dict[str, Any]]] = {}
    failures = set(manifest["failure_accounting"])
    for arm in ARMS:
        arms[arm], records[arm] = audit_arm(
            arm=arm,
            cell_path=arm_paths[arm],
            expected=expected,
            failures=failures,
            manifest_digest=manifest_digest,
        )
    _require(
        len({arms[arm]["asset_inventory_sha256"] for arm in ARMS}) == 1,
        "LongSafety generation asset inventories diverged.",
    )
    _require(set(records[ARMS[0]]) == set(records[ARMS[1]]), "LongSafety identities diverged.")
    _require(
        all(
            records[ARMS[0]][key]["raw_prompt_sha256"]
            == records[ARMS[1]][key]["raw_prompt_sha256"]
            and records[ARMS[0]][key]["input_token_ids_sha256"]
            == records[ARMS[1]][key]["input_token_ids_sha256"]
            for key in records[ARMS[0]]
        ),
        "LongSafety arms are not prompt/token paired.",
    )
    source_examples = contract["prompt_protocol"]["expected_rows"]
    official_calls_per_arm = source_examples * 4
    return {
        "schema_version": 1,
        "experiment_id": "p3-natural-safety-longsafety-generation-audit-v1",
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
            "validation": validation,
        },
        "audit": {
            "generation_arms_terminal": True,
            "input_pairing_verified": True,
            "generation_failure_accounting_complete": True,
            "official_judge_status": "blocked",
            "expected_generations_per_arm": expected,
            "expected_generations_total": expected * len(ARMS),
            "source_examples": source_examples,
            "prompt_positions": len(POSITIONS),
        },
        "arms": arms,
        "official_judge": {
            "status": "blocked",
            "reason": "paid API execution requires explicit user opt-in and OPENAI_API_KEY",
            "model": contract["judge"]["official_default_model"],
            "agents": contract["judge"]["agents"],
            "expected_api_calls_per_arm": official_calls_per_arm,
            "expected_api_calls_total": official_calls_per_arm * len(ARMS),
            "safety_scores_reported": False,
            "raw_generations_preserved": True,
        },
        "claim_boundary": (
            "Digest-bound LongSafety generation coverage only. With the official paid judge "
            "blocked, this artifact contains no LongSafety safety score and cannot support a "
            "comparative safety claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P3 LongSafety generation coverage.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    )
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/longsafety"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/longsafety.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "LongSafety summarization requires a clean source tree.")
    payload = summarize(
        args.manifest,
        {arm: args.generation_root / arm / "cell.json" for arm in ARMS},
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
