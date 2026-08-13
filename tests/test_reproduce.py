from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from nano_deepseek_v4 import reproduce
from nano_deepseek_v4.checkpoint import PretrainedBundleReport
from nano_deepseek_v4.train_text import TinyTextTrainingResult

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGED_CONTRACT = (
    _ROOT / "nano_deepseek_v4" / "_receipts" / "tiny-text-training-baseline.json"
)
_REFERENCE_CONTRACT = _ROOT / "references" / "tiny-text-training-baseline.json"


def _contract() -> tuple[dict[str, Any], str]:
    return reproduce._load_contract(_PACKAGED_CONTRACT.read_bytes())


def _fake_result(
    staging_bundle: Path,
    *,
    contract: dict[str, Any] | None = None,
    **changes: object,
) -> TinyTextTrainingResult:
    if contract is None:
        contract, _ = _contract()
    recipe = contract["recipe"]
    expected = contract["expected_result"]
    historical = contract["historical_runs"][0]
    observations = historical["observations"]
    result = TinyTextTrainingResult(
        schema_version=expected["result_schema_version"],
        source="builtin",
        corpus_sha256=expected["corpus_sha256"],
        config_source="tiny_text_config",
        config_sha256=expected["config_sha256"],
        implementation_sha256=historical["implementation_sha256"],
        python_version=historical["environment"]["python_version"],
        torch_version=historical["environment"]["torch_version"],
        machine=historical["environment"]["machine"],
        device=recipe["device"],
        seed=recipe["seed"],
        steps=recipe["steps"],
        context_length=recipe["context_length"],
        batch_size=recipe["batch_size"],
        eval_batches=recipe["eval_batches"],
        learning_rate=recipe["learning_rate"],
        max_new_tokens=recipe["max_new_tokens"],
        prompt_text=expected["prompt_text"],
        generation_temperature=recipe["temperature"],
        generation_top_p=recipe["top_p"],
        optimizer_names=tuple(expected["optimizer_names"]),
        parameter_count=expected["parameter_count"],
        trainable_parameter_count=expected["trainable_parameter_count"],
        initial_eval_loss=observations["initial_eval_loss"],
        final_eval_loss=observations["final_eval_loss"],
        final_train_loss=observations["final_train_loss"],
        loss_improved=True,
        elapsed_seconds=1.0,
        trained_tokens=expected["trained_tokens"],
        tokens_per_second=float(expected["trained_tokens"]),
        sample_text=observations["sample_text"],
        save_directory=str(staging_bundle),
        bundle_format_version=observations["bundle_format_version"],
        bundle_manifest_sha256=observations["bundle_manifest_sha256"],
        tokenizer_sha256=observations["tokenizer_sha256"],
        checkpoint_round_trip_match=observations["checkpoint_round_trip_match"],
        tokenizer_round_trip_match=observations["tokenizer_round_trip_match"],
    )
    return replace(result, **cast(Any, changes))


def _fake_bundle_report(
    staging_bundle: Path,
    result: TinyTextTrainingResult,
    **changes: object,
) -> PretrainedBundleReport:
    report = PretrainedBundleReport(
        is_complete=True,
        bundle_path=str(staging_bundle),
        format="nano-deepseek-v4-pretrained",
        format_version=2,
        model_class="DeepSeekV4ForCausalLM",
        tokenizer_file="tokenizer.json",
        tokenizer_format="nano-deepseek-v4-tokenizer",
        tokenizer_format_version=1,
        tokenizer_type="byte-v1",
        tokenizer_sha256=str(result.tokenizer_sha256),
        tokenizer_verified=True,
        generation_ready=True,
        config_sha256=result.config_sha256,
        manifest_sha256=str(result.bundle_manifest_sha256),
        index_sha256="a" * 64,
        checksums_verified=True,
        shard_headers_verified=True,
        file_count=4,
        shard_count=1,
        tensor_count=10,
        tensor_bytes=1024,
        index_total_size_bytes=1024,
        shard_files=["model-00001-of-00001.safetensors"],
        errors=[],
    )
    return replace(report, **cast(Any, changes))


def _evaluate(
    staging_bundle: Path,
    *,
    result_changes: dict[str, object] | None = None,
    report_changes: dict[str, object] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    TinyTextTrainingResult,
    PretrainedBundleReport,
    dict[str, Any],
    str,
]:
    contract, contract_sha256 = _contract()
    result = _fake_result(staging_bundle, contract=contract, **(result_changes or {}))
    bundle_report = _fake_bundle_report(staging_bundle, result, **(report_changes or {}))
    acceptance, historical = reproduce.evaluate_reproduction(
        result,
        bundle_report,
        contract,
        staging_bundle=staging_bundle,
    )
    return acceptance, historical, result, bundle_report, contract, contract_sha256


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_changes: dict[str, object] | None = None,
    report_changes: dict[str, object] | None = None,
) -> list[Path]:
    staged_paths: list[Path] = []

    def run(staging_bundle: Path) -> TinyTextTrainingResult:
        staging_bundle.mkdir(parents=True)
        (staging_bundle / "marker.txt").write_text("complete\n", encoding="utf-8")
        staged_paths.append(staging_bundle)
        return _fake_result(staging_bundle, **cast(Any, result_changes or {}))

    def verify(staging_bundle: Path, *, verify_checksums: bool) -> PretrainedBundleReport:
        assert verify_checksums is True
        result = _fake_result(staging_bundle, **cast(Any, result_changes or {}))
        changes = dict(report_changes or {})
        changes.setdefault("errors", [f"diagnostic for {staging_bundle}"])
        return _fake_bundle_report(staging_bundle, result, **changes)

    monkeypatch.setattr(reproduce, "_run_with_bundle", run)
    monkeypatch.setattr(reproduce, "verify_deepseek_v4_pretrained_bundle", verify)
    return staged_paths


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current: dict[str, Any] = payload
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value


def test_packaged_contract_is_exact_reference_bytes_and_loads_by_digest():
    packaged = _PACKAGED_CONTRACT.read_bytes()
    reference = _REFERENCE_CONTRACT.read_bytes()

    contract, digest = reproduce._load_contract()

    assert packaged == reference
    assert digest == hashlib.sha256(packaged).hexdigest()
    assert contract == json.loads(reference)
    assert contract["contract_id"] == reproduce.CONTRACT_ID


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 4),
        (("kind",), "similar-but-not-the-contract"),
        (("recipe", "steps"), 19),
        (("recipe", "temperature"), 0.8),
        (("acceptance", "minimum_validation_loss_reduction"), 1.99),
        (("expected_result", "config_sha256"), "0" * 64),
        (("expected_result", "optimizer_names"), ["AdamW", "Muon"]),
        (("expected_result", "bundle_verification", "is_complete"), False),
    ],
)
def test_contract_rejects_protocol_or_expected_result_drift(
    path: tuple[str, ...], value: object
):
    payload = json.loads(_PACKAGED_CONTRACT.read_bytes())
    _set_nested(payload, path, value)

    with pytest.raises(reproduce.ReproductionContractError) as raised:
        reproduce._load_contract(json.dumps(payload).encode())

    assert raised.value.reason_code == "contract_invalid"


def test_contract_rejects_unexpected_keys_and_duplicate_history():
    payload = json.loads(_PACKAGED_CONTRACT.read_bytes())
    payload["unexpected"] = True
    with pytest.raises(reproduce.ReproductionContractError, match="keys drifted"):
        reproduce._load_contract(json.dumps(payload).encode())

    payload = json.loads(_PACKAGED_CONTRACT.read_bytes())
    payload["historical_runs"].append(payload["historical_runs"][0])
    with pytest.raises(reproduce.ReproductionContractError, match="repeats historical run"):
        reproduce._load_contract(json.dumps(payload).encode())


def test_pure_evaluator_accepts_contract_and_matches_exact_history(tmp_path: Path):
    acceptance, historical, _, _, _, _ = _evaluate(tmp_path / "staged")

    assert acceptance["passed"] is True
    assert acceptance["reason_codes"] == []
    assert acceptance["summary"]["failed"] == 0
    assert historical["informational_only"] is True
    assert historical["status"] == "exact_match"
    assert historical["different_check_ids"] == []


def test_historical_drift_is_informational_and_preserves_hard_pass(tmp_path: Path):
    (
        acceptance,
        historical,
        result,
        bundle_report,
        contract,
        contract_sha256,
    ) = _evaluate(tmp_path / "staged", result_changes={"sample_text": "different sample"})

    report, exit_code = reproduce._build_report(
        result,
        bundle_report,
        contract=contract,
        contract_sha256=contract_sha256,
        acceptance=acceptance,
        historical=historical,
        requested_path=None,
        published=False,
    )

    assert acceptance["passed"] is True
    assert historical["status"] == "different"
    assert historical["different_check_ids"] == ["sample_text"]
    assert report["status"] == "pass"
    assert exit_code == reproduce.EXIT_PASS


def test_unknown_implementation_is_informational_and_preserves_hard_pass(tmp_path: Path):
    (
        acceptance,
        historical,
        result,
        bundle_report,
        contract,
        contract_sha256,
    ) = _evaluate(tmp_path / "staged", result_changes={"implementation_sha256": "f" * 64})

    report, exit_code = reproduce._build_report(
        result,
        bundle_report,
        contract=contract,
        contract_sha256=contract_sha256,
        acceptance=acceptance,
        historical=historical,
        requested_path=None,
        published=False,
    )

    assert acceptance["passed"] is True
    assert historical["status"] == "unverified_implementation"
    assert historical["checks"] == []
    assert report["status"] == "pass"
    assert exit_code == reproduce.EXIT_PASS


@pytest.mark.parametrize(
    ("final_eval_loss", "expected_pass"),
    [(5.0, False), (4.999, True)],
)
def test_loss_reduction_threshold_is_strictly_greater_than_two(
    tmp_path: Path,
    final_eval_loss: float,
    expected_pass: bool,
):
    acceptance, _, _, _, _, _ = _evaluate(
        tmp_path / "staged",
        result_changes={
            "implementation_sha256": "f" * 64,
            "initial_eval_loss": 7.0,
            "final_eval_loss": final_eval_loss,
        },
    )

    assert acceptance["passed"] is expected_pass
    assert (
        "insufficient_validation_loss_reduction" in acceptance["reason_codes"]
    ) is not expected_pass


def test_non_finite_loss_is_a_serializable_contract_failure(tmp_path: Path):
    (
        acceptance,
        historical,
        result,
        bundle_report,
        contract,
        contract_sha256,
    ) = _evaluate(
        tmp_path / "staged",
        result_changes={
            "implementation_sha256": "f" * 64,
            "final_eval_loss": float("nan"),
        },
    )

    report, exit_code = reproduce._build_report(
        result,
        bundle_report,
        contract=contract,
        contract_sha256=contract_sha256,
        acceptance=acceptance,
        historical=historical,
        requested_path=None,
        published=False,
    )
    payload = reproduce._serialize_report(report)

    assert exit_code == reproduce.EXIT_CONTRACT_FAILED
    assert "non_finite_loss" in report["reason_codes"]
    assert report["training_result"]["final_eval_loss"] == "NaN"
    assert json.loads(payload)["status"] == "fail"


@pytest.mark.parametrize(
    "field",
    [
        "is_complete",
        "generation_ready",
        "checksums_verified",
        "tokenizer_verified",
        "shard_headers_verified",
    ],
)
def test_independent_bundle_verifier_failure_fails_acceptance(tmp_path: Path, field: str):
    acceptance, _, _, _, _, _ = _evaluate(
        tmp_path / "staged",
        report_changes={field: False},
    )

    assert acceptance["passed"] is False
    assert "bundle_verification_failed" in acceptance["reason_codes"]


@pytest.mark.parametrize(
    ("report_changes", "reason_code"),
    [
        ({"manifest_sha256": "b" * 64}, "bundle_manifest_mismatch"),
        ({"tokenizer_sha256": "b" * 64}, "tokenizer_digest_mismatch"),
        ({"config_sha256": "b" * 64}, "bundle_config_digest_mismatch"),
    ],
)
def test_runner_and_verifier_digests_must_be_consistent(
    tmp_path: Path,
    report_changes: dict[str, object],
    reason_code: str,
):
    acceptance, _, _, _, _, _ = _evaluate(
        tmp_path / "staged",
        report_changes=report_changes,
    )

    assert acceptance["passed"] is False
    assert reason_code in acceptance["reason_codes"]


def test_temporary_bundle_is_deleted_and_private_path_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
):
    staged_paths = _install_fake_execution(monkeypatch)

    report, exit_code = reproduce.run_reproduction()
    serialized = reproduce._serialize_report(report)

    assert exit_code == reproduce.EXIT_PASS
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()
    assert str(staged_paths[0]) not in serialized
    assert "<staged-bundle>" in serialized
    assert report["artifact"] == {
        "disposition": "temporary_deleted",
        "requested_path": None,
        "published": False,
        "bundle_path": None,
        "bundle_format_version": 2,
        "manifest_sha256": _fake_result(Path("unused")).bundle_manifest_sha256,
        "verification_complete": True,
    }
    assert report["training_result"]["save_directory"] is None


def test_passing_bundle_is_published_with_one_atomic_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    staged_paths = _install_fake_execution(monkeypatch)
    destination = tmp_path / "published" / "bundle"
    real_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def observed_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.is_dir()
        assert target_path == destination
        assert not target_path.exists()
        replace_calls.append((source_path, target_path))
        real_replace(source_path, target_path)

    monkeypatch.setattr(reproduce.os, "replace", observed_replace)

    report, exit_code = reproduce.run_reproduction(save_directory=destination)

    assert exit_code == reproduce.EXIT_PASS
    assert replace_calls == [(staged_paths[0], destination)]
    assert (destination / "marker.txt").read_text(encoding="utf-8") == "complete\n"
    assert report["artifact"]["published"] is True
    assert report["artifact"]["bundle_path"] == str(destination)
    assert report["training_result"]["save_directory"] == str(destination)


def test_existing_empty_save_directory_is_replaced_only_after_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _install_fake_execution(monkeypatch)
    destination = tmp_path / "empty-bundle"
    destination.mkdir()

    report, exit_code = reproduce.run_reproduction(save_directory=destination)

    assert exit_code == reproduce.EXIT_PASS
    assert report["artifact"]["published"] is True
    assert (destination / "marker.txt").read_text(encoding="utf-8") == "complete\n"


def test_publish_failure_restores_an_existing_empty_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    staged_paths = _install_fake_execution(monkeypatch)
    destination = tmp_path / "empty-bundle"
    destination.mkdir()
    monkeypatch.setattr(
        reproduce.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("publish failed")),
    )

    with pytest.raises(reproduce.ReproductionOperationalError) as raised:
        reproduce.run_reproduction(save_directory=destination)

    assert raised.value.reason_code == "bundle_publish_failed"
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not staged_paths[0].exists()


def test_failing_bundle_is_never_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    staged_paths = _install_fake_execution(
        monkeypatch,
        result_changes={"initial_eval_loss": 7.0, "final_eval_loss": 5.0},
    )
    destination = tmp_path / "must-not-exist"

    def forbidden_replace(*_args: object) -> None:
        raise AssertionError("a failing reproduction must not be published")

    monkeypatch.setattr(reproduce.os, "replace", forbidden_replace)

    report, exit_code = reproduce.run_reproduction(save_directory=destination)

    assert exit_code == reproduce.EXIT_CONTRACT_FAILED
    assert not destination.exists()
    assert not staged_paths[0].exists()
    assert report["artifact"]["disposition"] == "not_published"
    assert report["artifact"]["published"] is False


def test_existing_nonempty_save_directory_is_rejected_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    destination = tmp_path / "nonempty"
    destination.mkdir()
    (destination / "owned.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        reproduce,
        "_run_with_bundle",
        lambda _path: pytest.fail("training must not start"),
    )

    with pytest.raises(reproduce.ReproductionUsageError, match="absent or empty"):
        reproduce.run_reproduction(save_directory=destination)

    assert (destination / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_existing_file_save_destination_is_rejected_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    destination = tmp_path / "file"
    destination.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        reproduce,
        "_run_with_bundle",
        lambda _path: pytest.fail("training must not start"),
    )

    with pytest.raises(reproduce.ReproductionUsageError, match="directory path"):
        reproduce.run_reproduction(save_directory=destination)

    assert destination.read_text(encoding="utf-8") == "keep"


def test_symlink_save_destination_is_rejected_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    destination = tmp_path / "link"
    destination.symlink_to(real_directory, target_is_directory=True)
    monkeypatch.setattr(
        reproduce,
        "_run_with_bundle",
        lambda _path: pytest.fail("training must not start"),
    )

    with pytest.raises(reproduce.ReproductionUsageError, match="symbolic link"):
        reproduce.run_reproduction(save_directory=destination)

    assert destination.is_symlink()


@pytest.mark.parametrize("output_relation", ["equal", "inside", "parent"])
def test_cli_rejects_overlapping_output_and_saved_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    output_relation: str,
):
    destination = tmp_path / "bundle"
    if output_relation == "equal":
        output = destination
    elif output_relation == "inside":
        output = destination / "receipt.json"
    else:
        output = destination.parent
    monkeypatch.setattr(
        reproduce,
        "run_reproduction",
        lambda **_kwargs: pytest.fail("execution must not start"),
    )

    with pytest.raises(SystemExit) as raised:
        reproduce.main(
            ["--save-directory", str(destination), "--output", str(output), "--json"]
    )

    assert raised.value.code == 2
    assert "--output and --save-directory must not overlap" in capsys.readouterr().err


def test_cli_emits_structured_contract_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def fail(**_kwargs: object) -> tuple[dict[str, Any], int]:
        raise reproduce.ReproductionContractError("contract_invalid", "bad contract")

    monkeypatch.setattr(reproduce, "run_reproduction", fail)

    exit_code = reproduce.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == reproduce.EXIT_EXECUTION_ERROR
    assert payload["status"] == "error"
    assert payload["reason_codes"] == ["contract_invalid"]
    assert payload["error"] == {
        "reason_code": "contract_invalid",
        "type": "ReproductionContractError",
        "detail": "bad contract",
    }


def test_cli_emits_structured_training_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        reproduce,
        "_run_with_bundle",
        lambda _path: (_ for _ in ()).throw(RuntimeError("private details")),
    )

    exit_code = reproduce.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == reproduce.EXIT_EXECUTION_ERROR
    assert payload["reason_codes"] == ["training_execution_error"]
    assert payload["error"] == {
        "reason_code": "training_execution_error",
        "type": "RuntimeError",
    }
    assert payload["artifact"]["disposition"] == "temporary_deleted"
    assert "private details" not in json.dumps(payload)


def test_cli_emits_structured_output_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        reproduce,
        "run_reproduction",
        lambda **_kwargs: (
            {
                "schema_version": 1,
                "kind": "tiny-text-reproduction",
                "status": "pass",
                "passed": True,
                "complete": True,
                "reason_codes": [],
                "artifact": {"published": False},
            },
            reproduce.EXIT_PASS,
        ),
    )
    monkeypatch.setattr(
        reproduce,
        "_write_json_receipt",
        lambda _path, _payload: (_ for _ in ()).throw(OSError("disk detail")),
    )

    exit_code = reproduce.main(["--json", "--output", str(tmp_path / "receipt.json")])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == reproduce.EXIT_EXECUTION_ERROR
    assert payload["reason_codes"] == ["output_write_failed"]
    assert payload["error"] == {"reason_code": "output_write_failed", "type": "OSError"}
    assert payload["artifact"] == {"published": False}
    assert "disk detail" not in json.dumps(payload)


def test_cli_json_stdout_is_byte_identical_to_atomic_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    acceptance, historical, result, bundle_report, contract, contract_sha256 = _evaluate(
        tmp_path / "staged"
    )
    report, expected_exit = reproduce._build_report(
        result,
        bundle_report,
        contract=contract,
        contract_sha256=contract_sha256,
        acceptance=acceptance,
        historical=historical,
        requested_path=None,
        published=False,
    )
    monkeypatch.setattr(
        reproduce,
        "run_reproduction",
        lambda **_kwargs: (report, expected_exit),
    )
    output = tmp_path / "nested" / "receipt.json"

    exit_code = reproduce.main(["--json", "--output", str(output)])
    stdout = capsys.readouterr().out

    assert exit_code == reproduce.EXIT_PASS
    assert stdout.encode() == output.read_bytes()
    assert json.loads(stdout)["passed"] is True
    assert list(output.parent.glob(f".{output.name}-*")) == []


def test_real_fixed_reproduction_passes_and_publishes_verified_bundle(tmp_path: Path):
    destination = tmp_path / "real-bundle"

    report, exit_code = reproduce.run_reproduction(save_directory=destination)

    assert exit_code == reproduce.EXIT_PASS
    assert report["status"] == "pass"
    assert report["acceptance"]["passed"] is True
    assert report["acceptance"]["summary"]["failed"] == 0
    assert report["artifact"]["published"] is True
    assert report["artifact"]["verification_complete"] is True
    assert report["bundle_verification"]["checksums_verified"] is True
    assert report["bundle_verification"]["generation_ready"] is True
    assert destination.is_dir()
    assert (destination / "nano_deepseek_v4.json").is_file()
