from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from score_p3_ifeval import (  # noqa: E402
    _records as load_cell_records,
)
from score_p3_ifeval import (  # noqa: E402
    aggregate,
    audit_official_outputs,
    paired_effect,
    score_arm,
)


class FakeOfficial:
    InputExample = SimpleNamespace

    @staticmethod
    def test_instruction_following_strict(inp, prompt_to_response):
        followed = prompt_to_response[inp.prompt] == "PASS"
        return SimpleNamespace(
            follow_instruction_list=[followed] * len(inp.instruction_id_list),
            follow_all_instructions=followed,
        )

    test_instruction_following_loose = test_instruction_following_strict


def _generation_source() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    path = "research/adaptive_v4_memory/scripts/run_p3_natural_safety_generation.py"
    blob = subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout
    return {
        "commit": commit,
        "dirty": False,
        "implementation_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _inputs() -> list[dict[str, object]]:
    return [
        {
            "key": 1,
            "prompt": "one",
            "instruction_id_list": ["a", "b"],
            "kwargs": [{}, {}],
        },
        {
            "key": 2,
            "prompt": "two",
            "instruction_id_list": ["c"],
            "kwargs": [{}],
        },
    ]


def _records(first: str, second: str) -> list[dict[str, object]]:
    return [
        {
            "source_id": 1,
            "example_id": "ifeval:1",
            "status": "generated",
            "failure_type": None,
            "raw_response": first,
            "metadata": {"instruction_id_list": ["a", "b"], "kwargs": [{}, {}]},
        },
        {
            "source_id": 2,
            "example_id": "ifeval:2",
            "status": "generated",
            "failure_type": None,
            "raw_response": second,
            "metadata": {"instruction_id_list": ["c"], "kwargs": [{}]},
        },
    ]


def test_official_ifeval_scoring_and_aggregation_are_failure_conservative() -> None:
    outputs = score_arm(inputs=_inputs(), records=_records("PASS", "FAIL"), official=FakeOfficial)

    assert aggregate(outputs) == {
        "expected_prompts": 2,
        "scored_prompts": 2,
        "scorer_failures": 0,
        "generation_failures": 0,
        "prompt_level_strict_accuracy": 0.5,
        "instruction_level_strict_accuracy": 2 / 3,
        "prompt_level_loose_accuracy": 0.5,
        "instruction_level_loose_accuracy": 2 / 3,
        "instruction_total": 3,
    }
    failed = _records("PASS", "PASS")
    failed[1] = {
        "source_id": 2,
        "example_id": "ifeval:2",
        "status": "failure",
        "failure_type": "oom",
        "metadata": {"instruction_id_list": ["c"], "kwargs": [{}]},
    }
    conservative = aggregate(score_arm(inputs=_inputs(), records=failed, official=FakeOfficial))
    assert conservative["generation_failures"] == 1
    assert conservative["prompt_level_strict_accuracy"] == 0.5


def test_ifeval_paired_effect_preserves_prompt_pairing() -> None:
    native = score_arm(inputs=_inputs(), records=_records("FAIL", "FAIL"), official=FakeOfficial)
    fixed = score_arm(inputs=_inputs(), records=_records("PASS", "FAIL"), official=FakeOfficial)

    result = paired_effect(fixed, native, seed=3, replicates=100)

    assert result["paired_prompts"] == 2
    assert result["mean_difference"] == 0.5
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 0)


def test_ifeval_main_contract_exposes_p5_audit_fields() -> None:
    source = (SCRIPTS / "score_p3_ifeval.py").read_text()
    for field in (
        "required_arms_terminal",
        "input_pairing_verified",
        "official_scoring_accounted",
        "source_implementations_verified",
        "generation_dependency_digests_verified",
        "generation_record_revisions_verified",
        "generation_terminal_measurement_schema_verified",
        "generation_seed_verified",
        "official_result_schema_verified",
        "expected_prompts_per_arm",
    ):
        assert f'"{field}"' in source


def _cell_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"
    manifest = json.loads(manifest_path.read_text())
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    natural = tmp_path / "natural.json"
    natural.write_text(
        json.dumps(
            {
                "model": {
                    "revision": manifest["model"]["revision"],
                    "snapshot_digest_set_sha256": manifest["model"][
                        "snapshot_digest_set_sha256"
                    ],
                }
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "experiment_id": "p3-fixed-baseline-selection-v1",
                "source": {"dirty": False},
                "selected_arm": "snapkv",
                "selected_compression_ratio": 0.5,
            }
        )
    )
    source = _generation_source()
    arm_config = {"press_name": "no_press", "compression_ratio": 0.0}
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "source_id": 1,
                "example_id": "ifeval:1",
                "benchmark": "IFEval",
                "arm": "native-dense",
                "prompt_position": "official-short",
                "exact_input_tokens": 10,
                "generation_reserve_tokens": 2048,
                "raw_prompt_sha256": "a" * 64,
                "input_token_ids_sha256": "b" * 64,
                "token_boundary_retreat": 0,
                "arm_config": arm_config,
                "metadata": {"instruction_id_list": ["a"], "kwargs": [{}]},
                "revisions": {
                    "model_revision": manifest["model"]["revision"],
                    "dataset_revision": manifest["benchmarks"]["IFEval"]["dataset"][
                        "revision"
                    ],
                    "code_revision": manifest["benchmarks"]["IFEval"]["upstream_code"][
                        "revision"
                    ],
                    "runner_sha256": source["implementation_sha256"],
                },
                "status": "generated",
                "raw_response": "PASS",
                "score": None,
                "evaluation_status": "pending-official-deterministic-scorer",
                "failure_type": None,
                "stop_reason": "eos-or-special-token",
                "generated_tokens_observed": 1,
                "latency_ms": 1.0,
                "peak_hbm_bytes": 100,
                "hot_resident_bytes": 50,
            },
            sort_keys=True,
        )
        + "\n"
    )
    digest = hashlib.sha256(records.read_bytes()).hexdigest()
    cell = tmp_path / "cell.json"
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    inventory_digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
    natural_digest = hashlib.sha256(natural.read_bytes()).hexdigest()
    selection_digest = hashlib.sha256(selection.read_bytes()).hexdigest()
    cell.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-generation-arm-cell-v1",
                "benchmark": "IFEval",
                "arm": "native-dense",
                "status": "terminal",
                "source": source,
                "run_identity": {
                    "source_commit": source["commit"],
                    "implementation_sha256": source["implementation_sha256"],
                    "manifest_sha256": manifest_digest,
                    "asset_inventory_sha256": inventory_digest,
                    "natural_manifest_sha256": natural_digest,
                    "fixed_selection_sha256": selection_digest,
                    "model_snapshot_digest_set_sha256": manifest["model"][
                        "snapshot_digest_set_sha256"
                    ],
                    "benchmark": "IFEval",
                    "arm_config": arm_config,
                    "seed": manifest["statistics"]["generation_seed"],
                },
                "expected_generations": 1,
                "manifest": {"path": str(manifest_path), "sha256": manifest_digest},
                "asset_inventory": {"path": str(inventory), "sha256": inventory_digest},
                "natural_manifest": {"path": str(natural), "sha256": natural_digest},
                "fixed_selection": {"path": str(selection), "sha256": selection_digest},
                "raw_records": {"path": str(records), "sha256": digest},
            }
        )
    )
    return cell, manifest


def test_ifeval_cell_accepts_fully_bound_generation_evidence(tmp_path: Path) -> None:
    cell, manifest = _cell_fixture(tmp_path)
    manifest_digest = audited_manifest_digest(manifest)

    records, audited = load_cell_records(
        cell,
        "native-dense",
        1,
        manifest_digest=manifest_digest,
        inventory_digest=json.loads(cell.read_text())["asset_inventory"]["sha256"],
        manifest=manifest,
    )

    assert len(records) == 1
    assert audited["run_identity"]["manifest_sha256"] == manifest_digest


def audited_manifest_digest(manifest: dict[str, object]) -> str:
    root = Path(__file__).resolve().parents[1]
    path = root / "research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"
    assert json.loads(path.read_text()) == manifest
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("arm", "wrong", "record provenance drifted"),
        ("latency_ms", float("nan"), "terminal measurements drifted"),
        ("peak_hbm_bytes", True, "terminal measurements drifted"),
    ],
)
def test_ifeval_cell_rejects_invalid_raw_generation(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    cell, manifest = _cell_fixture(tmp_path)
    payload = json.loads(cell.read_text())
    records_path = Path(payload["raw_records"]["path"])
    record = json.loads(records_path.read_text())
    record[field] = value
    records_path.write_text(json.dumps(record) + "\n")
    payload["raw_records"]["sha256"] = hashlib.sha256(records_path.read_bytes()).hexdigest()
    cell.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_cell_records(
            cell,
            "native-dense",
            1,
            manifest_digest=audited_manifest_digest(manifest),
            inventory_digest=payload["asset_inventory"]["sha256"],
            manifest=manifest,
        )


def test_ifeval_official_output_audit_rejects_non_boolean_metrics() -> None:
    outputs = score_arm(inputs=_inputs(), records=_records("PASS", "PASS"), official=FakeOfficial)
    outputs[0]["strict_follow_all_instructions"] = 1

    with pytest.raises(ValueError, match="official result schema drifted"):
        audit_official_outputs(outputs, expected=2)
