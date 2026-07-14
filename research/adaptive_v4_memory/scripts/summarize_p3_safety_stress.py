from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from p3_safety_workloads import FAMILIES
from verify_p3_natural_model import sha256

ARMS = (
    "native-dense",
    "strongest-memory-matched-fixed",
    "strongest-memory-matched-fixed+protected-prefix",
)
FIXED_ARM = "strongest-memory-matched-fixed"
PROTECTED_ARM = "strongest-memory-matched-fixed+protected-prefix"
RUNNER_PATH = "research/adaptive_v4_memory/scripts/run_p3_safety_stress.py"
WORKLOAD_PATH = "research/adaptive_v4_memory/scripts/p3_safety_workloads.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Safety {label} drifted.")
    int(value, 16)


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _dependency(metadata: Any, label: str) -> str:
    _require(isinstance(metadata, dict), f"Missing safety {label} metadata.")
    path = Path(metadata.get("path", ""))
    _require(path.is_file() and metadata.get("sha256") == sha256(path), f"Safety {label} drifted.")
    return metadata["sha256"]


def _verify_source_implementation(cell: dict[str, Any]) -> dict[str, str]:
    source = cell.get("source", {})
    identity = cell.get("run_identity", {})
    commit = source.get("commit")
    _require(
        isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit),
        "Invalid safety source commit.",
    )
    assert isinstance(commit, str)
    digests: dict[str, str] = {}
    for name, path in (("implementation_sha256", RUNNER_PATH), ("workload_sha256", WORKLOAD_PATH)):
        blob = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True)
        _require(blob.returncode == 0, f"Missing safety source at commit: {path}")
        digests[name] = hashlib.sha256(blob.stdout).hexdigest()
    _require(
        source.get("implementation_sha256") == digests["implementation_sha256"]
        and identity.get("source_commit") == commit
        and identity.get("implementation_sha256") == digests["implementation_sha256"]
        and identity.get("workload_sha256") == digests["workload_sha256"],
        "Safety source implementation does not match its commit.",
    )
    return {"commit": commit, "runner_path": RUNNER_PATH, "workload_path": WORKLOAD_PATH, **digests}


def _audit_protected_record(record: dict[str, Any], identifier: str) -> None:
    span = record.get("protected_prefix_token_span")
    _require(
        isinstance(span, dict)
        and span.get("start") == 0
        and _positive_integer(span.get("end"))
        and span.get("tokens") == span.get("end"),
        f"Protected token span drifted: {identifier}.",
    )
    assert isinstance(span, dict)
    _require(
        _nonnegative_integer(span.get("stable_boundary_retreat")),
        f"Protected token boundary drifted: {identifier}.",
    )
    if record.get("arm") != PROTECTED_ARM:
        _require(
            record.get("protected_prefix_physical_audit") is None,
            f"Unexpected protected physical audit: {identifier}.",
        )
    if record.get("arm") != PROTECTED_ARM or record.get("status") != "scored":
        return
    audit = record.get("protected_prefix_physical_audit")
    _require(
        isinstance(audit, dict)
        and audit.get("same_budget_verified") is True
        and audit.get("protected_start") == span["start"]
        and audit.get("protected_end") == span["end"]
        and audit.get("protected_tokens") == span["tokens"]
        and isinstance(audit.get("layers"), list)
        and audit.get("layer_count") == len(audit["layers"])
        and audit.get("layer_count", 0) > 0,
        f"Protected physical audit drifted: {identifier}.",
    )
    assert isinstance(audit, dict) and isinstance(audit.get("layers"), list)
    for layer in audit["layers"]:
        _require(
            isinstance(layer, dict)
            and layer.get("protected_start") == span["start"]
            and layer.get("protected_end") == span["end"]
            and layer.get("protected_tokens") == span["tokens"]
            and _nonnegative_integer(layer.get("layer_index"))
            and _positive_integer(layer.get("input_tokens"))
            and _nonnegative_integer(layer.get("kept_tokens"))
            and layer["kept_tokens"] <= layer["input_tokens"]
            and span["tokens"] <= layer["kept_tokens"]
            and _finite_nonnegative_number(layer.get("compression_ratio"))
            and layer["compression_ratio"] < 1.0
            and layer["kept_tokens"]
            == int(layer["input_tokens"] * (1.0 - layer["compression_ratio"])),
            f"Protected layer budget drifted: {identifier}.",
        )
    _require(
        len({layer["layer_index"] for layer in audit["layers"]}) == audit["layer_count"],
        f"Protected layer identities drifted: {identifier}.",
    )


def _record_map(cell_path: Path) -> dict[str, dict[str, Any]]:
    cell = json.loads(cell_path.read_text())
    records = _records(Path(cell["raw_records"]["path"]))
    return {str(record["example_id"]): record for record in records}


def _exact_paired_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    smaller = min(wins, losses)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(smaller + 1)
    ]
    maximum = max(log_probabilities)
    lower_tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in log_probabilities)
    return min(1.0, 2.0 * lower_tail)


def paired_protected_effect(
    fixed: dict[str, dict[str, Any]],
    protected: dict[str, dict[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    _require(set(fixed) == set(protected), "Protected contrast example identities diverged.")
    identifiers = sorted(fixed)
    differences = np.array(
        [
            float(protected[key].get("score") or 0.0) - float(fixed[key].get("score") or 0.0)
            for key in identifiers
        ],
        dtype=np.float64,
    )
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        bootstrap[index] = float(
            np.mean(differences[rng.integers(0, len(differences), len(differences))])
        )
    comparable = [
        key
        for key in identifiers
        if fixed[key].get("status") == protected[key].get("status") == "scored"
    ]
    resident_equal = all(
        fixed[key].get("hot_resident_bytes") == protected[key].get("hot_resident_bytes")
        for key in comparable
    )
    _require(resident_equal, "Protected contrast physical resident bytes diverged.")
    return {
        "estimand": f"{PROTECTED_ARM} minus {FIXED_ARM}; operational failures score zero",
        "paired_examples": len(identifiers),
        "mean_success_rate_difference": float(np.mean(differences)),
        "paired_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": wins,
        "ties": int(np.sum(differences == 0)),
        "losses": losses,
        "exact_two_sided_paired_pvalue": _exact_paired_pvalue(wins, losses),
        "physically_comparable_scored_pairs": len(comparable),
        "resident_bytes_equal_for_comparable_pairs": resident_equal,
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
    }


def audit_arm(
    *, arm: str, path: Path, manifest: dict[str, Any], manifest_digest: str
) -> tuple[dict[str, Any], dict[str, str]]:
    cell = json.loads(path.read_text())
    _require(
        cell.get("experiment_id") == "p3-safety-stress-arm-cell-v1"
        and cell.get("benchmark") == "SafetyStress"
        and cell.get("arm") == arm
        and cell.get("status") == "terminal"
        and cell.get("source", {}).get("dirty") is False,
        f"Invalid safety arm cell: {arm}.",
    )
    source_implementation = _verify_source_implementation(cell)
    _require(cell.get("manifest", {}).get("sha256") == manifest_digest, "Safety manifest drifted.")
    records_path = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records_path.is_file() and cell["raw_records"]["sha256"] == sha256(records_path),
        f"Safety raw records drifted: {arm}.",
    )
    records = _records(records_path)
    expected = manifest["expected_examples_per_arm"]
    _require(len(records) == expected, f"Safety record count drifted: {arm}.")
    identity = cell.get("run_identity", {})
    arm_config = identity.get("arm_config")
    _require(
        isinstance(identity, dict)
        and isinstance(arm_config, dict)
        and identity.get("manifest_sha256") == manifest_digest
        and identity.get("seed") == manifest["seed"],
        f"Safety run identity drifted: {arm}.",
    )
    model_snapshot_digest = cell.get("model_snapshot_digest_set_sha256")
    _sha256_value(model_snapshot_digest, "model snapshot")
    _require(
        identity.get("model_snapshot_digest_set_sha256") == model_snapshot_digest,
        f"Safety model snapshot identity drifted: {arm}.",
    )
    identities: set[str] = set()
    coordinates_seen: set[tuple[str, int, int]] = set()
    prompt_pairs: list[str] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)
    for record in records:
        family_value, context_value = record.get("family"), record.get("context_target")
        coordinate = record.get("coordinates", {})
        index = coordinate.get("index") if isinstance(coordinate, dict) else None
        _require(
            isinstance(family_value, str)
            and family_value in FAMILIES
            and type(context_value) is int
            and context_value in manifest["context_targets_tokens"]
            and _nonnegative_integer(index)
            and index < manifest["examples_per_family_context"]
            and coordinate
            == {"family": family_value, "context_target": context_value, "index": index}
            and _nonnegative_integer(record.get("exact_input_tokens"))
            and int(context_value * manifest["target_fill_tolerance"]["minimum_fraction"])
            <= record["exact_input_tokens"]
            <= context_value - manifest["generation_reserve_tokens"],
            "Safety slice, coordinate, or token accounting drifted.",
        )
        assert isinstance(family_value, str) and isinstance(context_value, int)
        assert isinstance(index, int)
        family, context = family_value, context_value
        coordinate_key = (family, context, index)
        expected_identifier = (
            f"{family}:{context}:{index}:"
            + hashlib.sha256(
                f"{manifest['seed']}:{family}:{context}:{index}".encode()
            ).hexdigest()
        )
        identifier = record.get("example_id")
        _require(
            identifier == expected_identifier
            and identifier not in identities
            and coordinate_key not in coordinates_seen,
            "Safety example identity or coordinate is missing or duplicated.",
        )
        assert isinstance(identifier, str)
        identities.add(identifier)
        coordinates_seen.add(coordinate_key)
        _require(
            record.get("benchmark") == "SafetyStress"
            and record.get("arm") == arm
            and record.get("generation_reserve_tokens")
            == manifest["generation_reserve_tokens"]
            and record.get("arm_config") == arm_config,
            "Safety record or arm configuration drifted.",
        )
        _audit_protected_record(record, identifier)
        for key in ("raw_prompt_sha256", "input_token_ids_sha256", "expected_response_sha256"):
            value = record.get(key)
            _sha256_value(value, key)
        canary = record.get("canary_sha256")
        if family in {"refusal-retention", "prompt-injection-leakage"}:
            _sha256_value(canary, "canary")
        else:
            _require(canary is None, f"Unexpected safety canary: {identifier}.")
        _require(
            record.get("revisions")
            == {
                "model_revision": manifest["model"]["revision"],
                "workload_sha256": source_implementation["workload_sha256"],
                "scorer_sha256": source_implementation["workload_sha256"],
            },
            f"Safety record revisions drifted: {identifier}.",
        )
        _require(
            isinstance(record.get("raw_response"), str)
            and (
                record.get("parsed_response") is None
                or isinstance(record.get("parsed_response"), str)
            )
            and isinstance(record.get("stop_reason"), str)
            and bool(record["stop_reason"])
            and _finite_nonnegative_number(record.get("latency_ms"))
            and _nonnegative_integer(record.get("peak_hbm_bytes"))
            and _nonnegative_integer(record.get("hot_resident_bytes"))
            and isinstance(record.get("leakage_event"), bool)
            and isinstance(record.get("exact_required_response"), bool)
            and (
                "generated_tokens_observed" not in record
                or _nonnegative_integer(record.get("generated_tokens_observed"))
            ),
            f"Invalid safety terminal measurements: {identifier}.",
        )
        prompt_pairs.append(
            f"{identifier}:{record['raw_prompt_sha256']}:{record['input_token_ids_sha256']}"
        )
        status = record.get("status")
        if status == "scored":
            _require(
                record.get("failure_type") is None
                and type(record.get("score")) in (int, float)
                and record["score"] in {0.0, 1.0}
                and record.get("parsed_response") is not None
                and _nonnegative_integer(record.get("generated_tokens_observed")),
                f"Invalid scored safety record: {identifier}.",
            )
        else:
            failure = record.get("failure_type")
            _require(
                status == "failure"
                and isinstance(failure, str)
                and failure in manifest["failure_accounting"]
                and record.get("score") is None,
                f"Unregistered safety failure: {identifier}.",
            )
            assert isinstance(failure, str)
            failures[failure] += 1
        groups[(family, context)].append(record)
    _require(
        coordinates_seen
        == {
            (family, context, index)
            for family in FAMILIES
            for context in manifest["context_targets_tokens"]
            for index in range(manifest["examples_per_family_context"])
        },
        f"Safety coordinate grid drifted: {arm}.",
    )
    slices: list[dict[str, Any]] = []
    for (family, context), rows in sorted(groups.items()):
        score_sum = sum(float(row.get("score") or 0.0) for row in rows)
        leakage = sum(row.get("leakage_event") is True for row in rows)
        scored = sum(row.get("status") == "scored" for row in rows)
        slices.append(
            {
                "family": family,
                "context_target": context,
                "expected_examples": len(rows),
                "scored_examples": scored,
                "failures": len(rows) - scored,
                "success_rate_failures_zero": score_sum / len(rows),
                "leakage_events": leakage,
                "leakage_rate_all_expected": leakage / len(rows),
            }
        )
    _require(
        len(slices) == len(FAMILIES) * len(manifest["context_targets_tokens"])
        and all(
            row["expected_examples"] == manifest["examples_per_family_context"] for row in slices
        ),
        "Safety slice Cartesian coverage drifted.",
    )
    all_scores = [float(row.get("score") or 0.0) for row in records]
    dependencies = {
        "causal_gate": _dependency(cell.get("causal_gate"), "causal gate"),
        "fixed_selection": _dependency(cell.get("fixed_baseline_selection"), "fixed selection"),
        "natural_manifest": _dependency(cell.get("natural_manifest"), "natural manifest"),
        "model_snapshot": cell["model_snapshot_digest_set_sha256"],
    }
    _require(
        identity.get("causal_gate_sha256") == dependencies["causal_gate"]
        and identity.get("fixed_selection_sha256") == dependencies["fixed_selection"]
        and identity.get("natural_manifest_sha256") == dependencies["natural_manifest"],
        f"Safety dependency identity drifted: {arm}.",
    )
    return (
        {
            "terminal": True,
            "expected_examples": expected,
            "scored_examples": sum(row.get("status") == "scored" for row in records),
            "failures_by_type": dict(sorted(failures.items())),
            "macro_success_rate_failures_zero": float(np.mean(all_scores)),
            "total_leakage_events": sum(row.get("leakage_event") is True for row in records),
            "slices": slices,
            "worst_slice": min(slices, key=lambda row: row["success_rate_failures_zero"]),
            "paired_prompt_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_pairs)).encode()
            ).hexdigest(),
            "raw_cell": {"path": str(path), "sha256": sha256(path)},
            "source_implementation": source_implementation,
        },
        dependencies,
    )


def summarize(manifest_path: Path, arm_paths: dict[str, Path]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    _require(
        manifest.get("experiment_id") == "p3-qwen3-4b-safety-stress-v1"
        and set(arm_paths) == set(ARMS),
        "Safety summary manifest or arm set drifted.",
    )
    manifest_digest = sha256(manifest_path)
    arms: dict[str, Any] = {}
    dependencies: list[dict[str, str]] = []
    for arm in ARMS:
        arms[arm], observed = audit_arm(
            arm=arm,
            path=arm_paths[arm],
            manifest=manifest,
            manifest_digest=manifest_digest,
        )
        dependencies.append(observed)
    _require(
        len({json.dumps(row, sort_keys=True) for row in dependencies}) == 1,
        "Safety arm dependencies diverged.",
    )
    _require(
        len({row["paired_prompt_digest_set_sha256"] for row in arms.values()}) == 1,
        "Safety arms are not prompt/token paired.",
    )
    _require(
        len({json.dumps(row["source_implementation"], sort_keys=True) for row in arms.values()})
        == 1,
        "Safety arms used different source implementations.",
    )
    contrast = paired_protected_effect(
        _record_map(arm_paths[FIXED_ARM]),
        _record_map(arm_paths[PROTECTED_ARM]),
        seed=int(manifest["seed"]),
    )
    return {
        "schema_version": 1,
        "experiment_id": "p3-safety-stress-audit-v1",
        "manifest": {"path": str(manifest_path), "sha256": manifest_digest},
        "audit": {
            "required_arms_terminal": True,
            "failure_accounting_complete": True,
            "input_pairing_verified": True,
            "source_implementations_verified": True,
            "protected_prefix_physical_budget_verified": True,
            "examples_accounted_per_arm": manifest["expected_examples_per_arm"],
            "families_terminal": len(FAMILIES),
            "contexts_terminal": len(manifest["context_targets_tokens"]),
        },
        "dependencies": dependencies[0],
        "arms": arms,
        "protected_prefix_causal_contrast": contrast,
        "claim_boundary": manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen P3 safety stress suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-safety-stress-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/safety-stress.summary.json"),
    )
    args = parser.parse_args()
    root = Path("artifacts/adaptive_v4_memory/paper_grade/p3/safety-stress")
    arm_paths = {arm: root / arm / "cell.json" for arm in ARMS}
    payload = summarize(args.manifest, arm_paths)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Safety summarization requires a clean source tree.")
    payload["source"] = {"commit": commit, "dirty": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
