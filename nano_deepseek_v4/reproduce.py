from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from . import __version__
from .checkpoint import PretrainedBundleReport, verify_deepseek_v4_pretrained_bundle
from .train_text import TinyTextTrainingResult, _write_json_receipt, run_tiny_text_training

PACKAGE_NAME = "nano-deepseek-v4"
RECEIPT_SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 3
CONTRACT_KIND = "tiny-text-cpu-reproduction-contract"
CONTRACT_ID = "tiny-text-cpu-v1"
CONTRACT_RESOURCE = "_receipts/tiny-text-training-baseline.json"

EXIT_PASS = 0
EXIT_CONTRACT_FAILED = 1
EXIT_EXECUTION_ERROR = 4

_FIXED_RECIPE: dict[str, object] = {
    "batch_size": 4,
    "context_length": 32,
    "device": "cpu",
    "eval_batches": 4,
    "learning_rate": 2e-3,
    "max_new_tokens": 32,
    "prompt": None,
    "seed": 0,
    "steps": 20,
    "temperature": None,
    "top_p": 1.0,
}

_FIXED_MINIMUM_VALIDATION_LOSS_REDUCTION = 2.0

_FIXED_EXPECTED_RESULT: dict[str, object] = {
    "bundle_format": "nano-deepseek-v4-pretrained",
    "bundle_format_version": 2,
    "checkpoint_round_trip_match": True,
    "config_sha256": "3829a174d4f1102e001c66cc1e29a885b2f9960a5fde8499dd629cd63f1ca623",
    "corpus_sha256": "8c6c2bce0af2b9ec6936306952fc3e8c74b15dcdc140d6255dd3d5124d17a846",
    "optimizer_names": ["Muon", "AdamW"],
    "parameter_count": 246_590,
    "prompt_text": ".\nHeavily compre",
    "result_schema_version": 3,
    "tokenizer_format": "nano-deepseek-v4-tokenizer",
    "tokenizer_format_version": 1,
    "tokenizer_round_trip_match": True,
    "tokenizer_type": "byte-v1",
    "trainable_parameter_count": 246_590,
    "trained_tokens": 2_560,
}

_FIXED_BUNDLE_VERIFICATION = {
    "checksums_verified": True,
    "generation_ready": True,
    "is_complete": True,
    "shard_headers_verified": True,
    "tokenizer_verified": True,
}

_HISTORICAL_OBSERVATION_KEYS = {
    "bundle_format_version",
    "bundle_manifest_sha256",
    "checkpoint_round_trip_match",
    "final_eval_loss",
    "final_train_loss",
    "initial_eval_loss",
    "sample_text",
    "tokenizer_round_trip_match",
    "tokenizer_sha256",
}


class ReproductionContractError(RuntimeError):
    """The wheel-bundled reproduction contract cannot be trusted or evaluated."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


class ReproductionUsageError(ValueError):
    """A requested output path cannot satisfy the publication contract."""


class ReproductionOperationalError(RuntimeError):
    """The harness failed before it could return a completed acceptance result."""

    def __init__(self, reason_code: str, cause_type: str) -> None:
        super().__init__(f"{reason_code}: {cause_type}")
        self.reason_code = reason_code
        self.cause_type = cause_type


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReproductionContractError("contract_invalid", f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ReproductionContractError("contract_invalid", f"{label} must be a list")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ReproductionContractError(
            "contract_invalid",
            f"{label} keys drifted (missing={missing}, unexpected={unexpected})",
        )


def _string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ReproductionContractError(
            "contract_invalid", f"{label}.{key} must be a non-empty string"
        )
    return value


def _integer(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReproductionContractError(
            "contract_invalid", f"{label}.{key} must be an integer"
        )
    return value


def _number(mapping: Mapping[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ReproductionContractError(
            "contract_invalid", f"{label}.{key} must be a finite number"
        )
    return float(value)


def _boolean(mapping: Mapping[str, Any], key: str, label: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ReproductionContractError(
            "contract_invalid", f"{label}.{key} must be a boolean"
        )
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _string(mapping, key, label)
    if not _is_sha256(value):
        raise ReproductionContractError(
            "contract_invalid", f"{label}.{key} must be a lowercase SHA-256"
        )
    return value


def _require_exact_value(
    mapping: Mapping[str, Any], key: str, expected: object, label: str
) -> None:
    actual = mapping.get(key)
    if actual != expected or type(actual) is not type(expected):
        raise ReproductionContractError(
            "contract_invalid",
            f"{label}.{key} must remain {expected!r}",
        )


def _validate_historical_runs(
    raw_runs: object,
    *,
    minimum_reduction: float,
) -> list[Mapping[str, Any]]:
    runs = _list(raw_runs, "contract.historical_runs")
    if not runs:
        raise ReproductionContractError(
            "contract_invalid", "contract.historical_runs must not be empty"
        )

    run_ids: set[str] = set()
    implementation_ids: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for index, raw_run in enumerate(runs):
        label = f"contract.historical_runs[{index}]"
        run = _mapping(raw_run, label)
        _exact_keys(run, {"environment", "id", "implementation_sha256", "observations"}, label)
        run_id = _string(run, "id", label)
        implementation_id = _require_sha256(run, "implementation_sha256", label)
        if run_id in run_ids:
            raise ReproductionContractError(
                "contract_invalid", f"contract repeats historical run id {run_id!r}"
            )
        if implementation_id in implementation_ids:
            raise ReproductionContractError(
                "contract_invalid",
                f"contract repeats implementation history {implementation_id}",
            )
        run_ids.add(run_id)
        implementation_ids.add(implementation_id)

        environment = _mapping(run.get("environment"), f"{label}.environment")
        _exact_keys(
            environment,
            {"device", "machine", "python_version", "torch_version"},
            f"{label}.environment",
        )
        if _string(environment, "device", f"{label}.environment") != "cpu":
            raise ReproductionContractError(
                "contract_invalid", f"{label}.environment.device must be cpu"
            )
        for key in ("machine", "python_version", "torch_version"):
            _string(environment, key, f"{label}.environment")

        observations = _mapping(run.get("observations"), f"{label}.observations")
        _exact_keys(observations, _HISTORICAL_OBSERVATION_KEYS, f"{label}.observations")
        for key in ("initial_eval_loss", "final_eval_loss", "final_train_loss"):
            _number(observations, key, f"{label}.observations")
        _string(observations, "sample_text", f"{label}.observations")
        _require_sha256(observations, "bundle_manifest_sha256", f"{label}.observations")
        _require_sha256(observations, "tokenizer_sha256", f"{label}.observations")
        if _integer(observations, "bundle_format_version", f"{label}.observations") != 2:
            raise ReproductionContractError(
                "contract_invalid", f"{label}.observations.bundle_format_version must be 2"
            )
        for key in ("checkpoint_round_trip_match", "tokenizer_round_trip_match"):
            if not _boolean(observations, key, f"{label}.observations"):
                raise ReproductionContractError(
                    "contract_invalid", f"{label}.observations.{key} must be true"
                )
        reduction = _number(
            observations, "initial_eval_loss", f"{label}.observations"
        ) - _number(observations, "final_eval_loss", f"{label}.observations")
        if reduction <= minimum_reduction:
            raise ReproductionContractError(
                "contract_invalid",
                f"{label} does not satisfy its own minimum loss reduction",
            )
        validated.append(run)
    return validated


def _load_contract(contract_bytes: bytes | None = None) -> tuple[dict[str, Any], str]:
    if contract_bytes is None:
        try:
            contract_bytes = (
                resource_files("nano_deepseek_v4").joinpath(CONTRACT_RESOURCE).read_bytes()
            )
        except OSError as exc:
            raise ReproductionContractError(
                "contract_resource_missing", f"cannot read {CONTRACT_RESOURCE}"
            ) from exc
    try:
        payload = json.loads(contract_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReproductionContractError(
            "contract_invalid", f"{CONTRACT_RESOURCE} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ReproductionContractError("contract_invalid", "contract must be a JSON object")

    _exact_keys(
        payload,
        {
            "acceptance",
            "contract_id",
            "expected_result",
            "historical_runs",
            "kind",
            "recipe",
            "schema_version",
        },
        "contract",
    )
    _require_exact_value(payload, "schema_version", CONTRACT_SCHEMA_VERSION, "contract")
    _require_exact_value(payload, "kind", CONTRACT_KIND, "contract")
    _require_exact_value(payload, "contract_id", CONTRACT_ID, "contract")

    recipe = _mapping(payload.get("recipe"), "contract.recipe")
    _exact_keys(recipe, set(_FIXED_RECIPE), "contract.recipe")
    for key, expected in _FIXED_RECIPE.items():
        _require_exact_value(recipe, key, expected, "contract.recipe")

    acceptance = _mapping(payload.get("acceptance"), "contract.acceptance")
    _exact_keys(acceptance, {"minimum_validation_loss_reduction"}, "contract.acceptance")
    minimum_reduction = _number(
        acceptance, "minimum_validation_loss_reduction", "contract.acceptance"
    )
    if minimum_reduction != _FIXED_MINIMUM_VALIDATION_LOSS_REDUCTION:
        raise ReproductionContractError(
            "contract_invalid",
            "contract.acceptance.minimum_validation_loss_reduction must remain "
            f"{_FIXED_MINIMUM_VALIDATION_LOSS_REDUCTION}",
        )

    expected_result = _mapping(payload.get("expected_result"), "contract.expected_result")
    expected_keys = set(_FIXED_EXPECTED_RESULT) | {"bundle_verification"}
    _exact_keys(expected_result, expected_keys, "contract.expected_result")
    for key, expected in _FIXED_EXPECTED_RESULT.items():
        _require_exact_value(expected_result, key, expected, "contract.expected_result")
    bundle_verification = _mapping(
        expected_result.get("bundle_verification"),
        "contract.expected_result.bundle_verification",
    )
    _exact_keys(
        bundle_verification,
        set(_FIXED_BUNDLE_VERIFICATION),
        "contract.expected_result.bundle_verification",
    )
    for key, expected in _FIXED_BUNDLE_VERIFICATION.items():
        _require_exact_value(
            bundle_verification,
            key,
            expected,
            "contract.expected_result.bundle_verification",
        )

    _validate_historical_runs(
        payload.get("historical_runs"),
        minimum_reduction=minimum_reduction,
    )
    return payload, hashlib.sha256(contract_bytes).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _equality_check(
    check_id: str,
    reason_code: str,
    expected: object,
    observed: object,
) -> dict[str, object]:
    normalized_expected = _json_value(expected)
    normalized_observed = _json_value(observed)
    passed = normalized_observed == normalized_expected
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "reason_code": None if passed else reason_code,
        "expected": normalized_expected,
        "observed": normalized_observed,
    }


def _boolean_check(
    check_id: str,
    reason_code: str,
    passed: bool,
    *,
    observed: object,
) -> dict[str, object]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "reason_code": None if passed else reason_code,
        "expected": True,
        "observed": observed,
    }


def _matching_historical_run(
    contract: Mapping[str, Any], implementation_sha256: str
) -> Mapping[str, Any] | None:
    matches = [
        _mapping(raw_run, f"contract.historical_runs[{index}]")
        for index, raw_run in enumerate(
            _list(contract.get("historical_runs"), "contract.historical_runs")
        )
        if isinstance(raw_run, Mapping)
        and raw_run.get("implementation_sha256") == implementation_sha256
    ]
    if len(matches) > 1:
        raise ReproductionContractError(
            "contract_invalid",
            f"contract repeats implementation history {implementation_sha256}",
        )
    return matches[0] if matches else None


def _historical_comparison(
    result: TinyTextTrainingResult,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    historical_run = _matching_historical_run(contract, result.implementation_sha256)
    if historical_run is None:
        return {
            "informational_only": True,
            "status": "unverified_implementation",
            "reference_id": None,
            "reference_environment": None,
            "different_check_ids": [],
            "checks": [],
        }

    observations = _mapping(
        historical_run.get("observations"), "historical_run.observations"
    )
    fields = {
        "initial_eval_loss": result.initial_eval_loss,
        "final_eval_loss": result.final_eval_loss,
        "final_train_loss": result.final_train_loss,
        "sample_text": result.sample_text,
        "bundle_format_version": result.bundle_format_version,
        "bundle_manifest_sha256": result.bundle_manifest_sha256,
        "tokenizer_sha256": result.tokenizer_sha256,
        "checkpoint_round_trip_match": result.checkpoint_round_trip_match,
        "tokenizer_round_trip_match": result.tokenizer_round_trip_match,
    }
    checks = [
        _equality_check(
            field,
            f"historical_{field}_different",
            observations[field],
            observed,
        )
        for field, observed in fields.items()
    ]
    different_ids = [
        str(check["id"]) for check in checks if check["status"] == "fail"
    ]
    return {
        "informational_only": True,
        "status": "exact_match" if not different_ids else "different",
        "reference_id": historical_run["id"],
        "reference_environment": historical_run["environment"],
        "different_check_ids": different_ids,
        "checks": checks,
    }


def evaluate_reproduction(
    result: TinyTextTrainingResult,
    bundle_report: PretrainedBundleReport,
    contract: Mapping[str, Any],
    *,
    staging_bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate portable acceptance independently from historical exact bytes."""

    recipe = _mapping(contract.get("recipe"), "contract.recipe")
    expected = _mapping(contract.get("expected_result"), "contract.expected_result")
    expected_bundle = _mapping(
        expected.get("bundle_verification"),
        "contract.expected_result.bundle_verification",
    )
    acceptance = _mapping(contract.get("acceptance"), "contract.acceptance")

    checks: list[dict[str, object]] = [
        _equality_check(
            "result_schema_version",
            "result_schema_version_mismatch",
            expected["result_schema_version"],
            result.schema_version,
        ),
        _equality_check(
            "corpus_digest",
            "corpus_digest_mismatch",
            expected["corpus_sha256"],
            result.corpus_sha256,
        ),
        _equality_check(
            "config_digest",
            "config_digest_mismatch",
            expected["config_sha256"],
            result.config_sha256,
        ),
        _equality_check("device", "device_mismatch", recipe["device"], result.device),
        _equality_check("seed", "protocol_mismatch", recipe["seed"], result.seed),
        _equality_check("steps", "protocol_mismatch", recipe["steps"], result.steps),
        _equality_check(
            "context_length",
            "protocol_mismatch",
            recipe["context_length"],
            result.context_length,
        ),
        _equality_check(
            "batch_size", "protocol_mismatch", recipe["batch_size"], result.batch_size
        ),
        _equality_check(
            "eval_batches",
            "protocol_mismatch",
            recipe["eval_batches"],
            result.eval_batches,
        ),
        _equality_check(
            "learning_rate",
            "protocol_mismatch",
            recipe["learning_rate"],
            result.learning_rate,
        ),
        _equality_check(
            "max_new_tokens",
            "protocol_mismatch",
            recipe["max_new_tokens"],
            result.max_new_tokens,
        ),
        _equality_check(
            "generation_temperature",
            "protocol_mismatch",
            recipe["temperature"],
            result.generation_temperature,
        ),
        _equality_check(
            "generation_top_p",
            "protocol_mismatch",
            recipe["top_p"],
            result.generation_top_p,
        ),
        _equality_check(
            "prompt_text",
            "protocol_mismatch",
            expected["prompt_text"],
            result.prompt_text,
        ),
        _equality_check(
            "optimizer_names",
            "optimizer_mismatch",
            expected["optimizer_names"],
            result.optimizer_names,
        ),
        _equality_check(
            "parameter_count",
            "parameter_count_mismatch",
            expected["parameter_count"],
            result.parameter_count,
        ),
        _equality_check(
            "trainable_parameter_count",
            "trainable_parameter_count_mismatch",
            expected["trainable_parameter_count"],
            result.trainable_parameter_count,
        ),
        _equality_check(
            "trained_tokens",
            "trained_token_count_mismatch",
            expected["trained_tokens"],
            result.trained_tokens,
        ),
    ]

    finite_losses = all(
        math.isfinite(value)
        for value in (
            result.initial_eval_loss,
            result.final_eval_loss,
            result.final_train_loss,
        )
    )
    checks.append(
        _boolean_check(
            "finite_losses",
            "non_finite_loss",
            finite_losses,
            observed=finite_losses,
        )
    )
    minimum_reduction = _number(
        acceptance, "minimum_validation_loss_reduction", "contract.acceptance"
    )
    loss_reduction = result.initial_eval_loss - result.final_eval_loss
    reduction_passed = math.isfinite(loss_reduction) and loss_reduction > minimum_reduction
    checks.append(
        {
            "id": "validation_loss_reduction",
            "status": "pass" if reduction_passed else "fail",
            "reason_code": (
                None if reduction_passed else "insufficient_validation_loss_reduction"
            ),
            "expected": {"operator": ">", "value": minimum_reduction},
            "observed": _json_value(loss_reduction),
        }
    )

    resolved_staging = staging_bundle.resolve()
    try:
        reported_save_directory = (
            Path(result.save_directory).resolve()
            if result.save_directory is not None
            else None
        )
    except OSError:
        reported_save_directory = None
    checks.extend(
        (
            _boolean_check(
                "bundle_saved_to_staging",
                "bundle_path_mismatch",
                reported_save_directory == resolved_staging,
                observed=result.save_directory is not None,
            ),
            _equality_check(
                "bundle_format",
                "bundle_format_mismatch",
                expected["bundle_format"],
                bundle_report.format,
            ),
            _equality_check(
                "bundle_format_version",
                "bundle_format_mismatch",
                expected["bundle_format_version"],
                result.bundle_format_version,
            ),
            _equality_check(
                "verified_bundle_format_version",
                "bundle_format_mismatch",
                expected["bundle_format_version"],
                bundle_report.format_version,
            ),
            _equality_check(
                "tokenizer_format",
                "tokenizer_format_mismatch",
                expected["tokenizer_format"],
                bundle_report.tokenizer_format,
            ),
            _equality_check(
                "tokenizer_format_version",
                "tokenizer_format_mismatch",
                expected["tokenizer_format_version"],
                bundle_report.tokenizer_format_version,
            ),
            _equality_check(
                "tokenizer_type",
                "tokenizer_format_mismatch",
                expected["tokenizer_type"],
                bundle_report.tokenizer_type,
            ),
            _equality_check(
                "checkpoint_round_trip",
                "checkpoint_round_trip_mismatch",
                expected["checkpoint_round_trip_match"],
                result.checkpoint_round_trip_match,
            ),
            _equality_check(
                "tokenizer_round_trip",
                "tokenizer_round_trip_mismatch",
                expected["tokenizer_round_trip_match"],
                result.tokenizer_round_trip_match,
            ),
        )
    )
    for field in (
        "is_complete",
        "generation_ready",
        "checksums_verified",
        "tokenizer_verified",
        "shard_headers_verified",
    ):
        checks.append(
            _equality_check(
                f"bundle_{field}",
                "bundle_verification_failed",
                expected_bundle[field],
                getattr(bundle_report, field),
            )
        )

    manifest_consistent = (
        _is_sha256(result.bundle_manifest_sha256)
        and _is_sha256(bundle_report.manifest_sha256)
        and result.bundle_manifest_sha256 == bundle_report.manifest_sha256
    )
    checks.append(
        _boolean_check(
            "bundle_manifest_consistency",
            "bundle_manifest_mismatch",
            manifest_consistent,
            observed={
                "runner": result.bundle_manifest_sha256,
                "verifier": bundle_report.manifest_sha256,
            },
        )
    )
    tokenizer_consistent = (
        _is_sha256(result.tokenizer_sha256)
        and _is_sha256(bundle_report.tokenizer_sha256)
        and result.tokenizer_sha256 == bundle_report.tokenizer_sha256
    )
    checks.append(
        _boolean_check(
            "tokenizer_digest_consistency",
            "tokenizer_digest_mismatch",
            tokenizer_consistent,
            observed={
                "runner": result.tokenizer_sha256,
                "verifier": bundle_report.tokenizer_sha256,
            },
        )
    )
    checks.append(
        _equality_check(
            "bundle_config_digest_consistency",
            "bundle_config_digest_mismatch",
            result.config_sha256,
            bundle_report.config_sha256,
        )
    )

    reason_codes = [
        str(check["reason_code"])
        for check in checks
        if check["status"] == "fail" and check["reason_code"] is not None
    ]
    unique_reason_codes = list(dict.fromkeys(reason_codes))
    passed_count = sum(check["status"] == "pass" for check in checks)
    acceptance_report = {
        "passed": not unique_reason_codes,
        "reason_codes": unique_reason_codes,
        "summary": {
            "total": len(checks),
            "passed": passed_count,
            "failed": len(checks) - passed_count,
        },
        "checks": checks,
    }
    return acceptance_report, _historical_comparison(result, contract)


def _evaluator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _run_with_bundle(save_directory: Path) -> TinyTextTrainingResult:
    return run_tiny_text_training(
        steps=20,
        context_length=32,
        batch_size=4,
        eval_batches=4,
        learning_rate=2e-3,
        seed=0,
        device="cpu",
        max_new_tokens=32,
        prompt=None,
        temperature=None,
        top_p=1.0,
        save_directory=save_directory,
    )


def _sanitize_verifier_report(
    report: PretrainedBundleReport,
    *,
    public_bundle_path: Path | None,
) -> dict[str, Any]:
    payload = report.to_dict()
    private_path = str(payload["bundle_path"])
    payload["bundle_path"] = (
        str(public_bundle_path) if public_bundle_path is not None else None
    )
    payload["errors"] = [
        str(error).replace(private_path, "<staged-bundle>")
        for error in payload["errors"]
    ]
    return payload


def _build_report(
    result: TinyTextTrainingResult,
    bundle_report: PretrainedBundleReport,
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    acceptance: Mapping[str, Any],
    historical: Mapping[str, Any],
    requested_path: Path | None,
    published: bool,
) -> tuple[dict[str, Any], int]:
    passed = acceptance.get("passed") is True
    reason_codes = list(acceptance.get("reason_codes", []))
    normalized_training_result = _json_value(result.to_dict())
    if not isinstance(normalized_training_result, dict):
        raise TypeError("training result must serialize to an object")
    training_result = normalized_training_result
    public_bundle_path = requested_path if published else None
    training_result["save_directory"] = (
        str(public_bundle_path) if public_bundle_path is not None else None
    )
    if published:
        disposition = "preserved"
    elif requested_path is None:
        disposition = "temporary_deleted"
    else:
        disposition = "not_published"

    report: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": "tiny-text-reproduction",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "complete": True,
        "reason_codes": reason_codes,
        "package": {"name": PACKAGE_NAME, "version": __version__},
        "contract": {
            "id": contract["contract_id"],
            "schema_version": contract["schema_version"],
            "resource": CONTRACT_RESOURCE,
            "sha256": contract_sha256,
        },
        "environment": {
            "python_version": result.python_version,
            "torch_version": result.torch_version,
            "machine": result.machine,
            "device": result.device,
        },
        "protocol": dict(_FIXED_RECIPE),
        "provenance": {
            "corpus_sha256": result.corpus_sha256,
            "config_sha256": result.config_sha256,
            "runner_implementation_sha256": result.implementation_sha256,
            "evaluator_sha256": _evaluator_sha256(),
        },
        "acceptance": dict(acceptance),
        "historical_observations": dict(historical),
        "artifact": {
            "disposition": disposition,
            "requested_path": str(requested_path) if requested_path is not None else None,
            "published": published,
            "bundle_path": (
                str(public_bundle_path) if public_bundle_path is not None else None
            ),
            "bundle_format_version": result.bundle_format_version,
            "manifest_sha256": result.bundle_manifest_sha256,
            "verification_complete": bundle_report.is_complete,
        },
        "bundle_verification": _sanitize_verifier_report(
            bundle_report,
            public_bundle_path=public_bundle_path,
        ),
        "training_result": training_result,
        "claim_boundary": {
            "establishes": (
                "The installed package learns the fixed tiny byte corpus on CPU and "
                "persists a checksum-complete, generation-ready native v2 bundle."
            ),
            "does_not_establish": [
                "official DeepSeek-V4 weights or training parity",
                "language quality or benchmark performance",
                "GPU kernel correctness or accelerator performance",
                "full-scale convergence",
            ],
        },
    }
    return report, EXIT_PASS if passed else EXIT_CONTRACT_FAILED


def _validate_save_directory(path: Path) -> None:
    if path.is_symlink():
        raise ReproductionUsageError("--save-directory must not be a symbolic link")
    if path.exists():
        if not path.is_dir():
            raise ReproductionUsageError("--save-directory must be a directory path")
        try:
            has_entries = next(path.iterdir(), None) is not None
        except OSError as exc:
            raise ReproductionUsageError("--save-directory cannot be inspected") from exc
        if has_entries:
            raise ReproductionUsageError("--save-directory must be absent or empty")


def _validate_output_location(save_directory: Path | None, output: Path | None) -> None:
    if save_directory is None or output is None:
        return
    try:
        destination = save_directory.resolve(strict=False)
        receipt = output.resolve(strict=False)
    except OSError as exc:
        raise ReproductionUsageError("output paths cannot be resolved") from exc
    if (
        receipt == destination
        or destination in receipt.parents
        or receipt in destination.parents
    ):
        raise ReproductionUsageError(
            "--output and --save-directory must not overlap"
        )


def _publish_bundle(staging_bundle: Path, destination: Path) -> None:
    _validate_save_directory(destination)
    removed_empty_destination = False
    if destination.exists():
        destination.rmdir()
        removed_empty_destination = True
    try:
        os.replace(staging_bundle, destination)
    except OSError:
        if removed_empty_destination and not destination.exists():
            destination.mkdir()
        raise


def _run_staged(
    staging_bundle: Path,
    *,
    contract: Mapping[str, Any],
) -> tuple[TinyTextTrainingResult, PretrainedBundleReport, dict[str, Any], dict[str, Any]]:
    try:
        result = _run_with_bundle(staging_bundle)
        bundle_report = verify_deepseek_v4_pretrained_bundle(
            staging_bundle,
            verify_checksums=True,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReproductionOperationalError(
            "training_execution_error", type(exc).__name__
        ) from exc
    acceptance, historical = evaluate_reproduction(
        result,
        bundle_report,
        contract,
        staging_bundle=staging_bundle,
    )
    return result, bundle_report, acceptance, historical


def run_reproduction(
    *,
    save_directory: Path | None = None,
    contract_bytes: bytes | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the fixed CPU experiment and return its evidence-carrying receipt."""

    contract, contract_sha256 = _load_contract(contract_bytes)
    if save_directory is None:
        try:
            with tempfile.TemporaryDirectory(
                prefix="nano-deepseek-v4-reproduce-"
            ) as temporary:
                staging_bundle = Path(temporary) / "bundle"
                result, bundle_report, acceptance, historical = _run_staged(
                    staging_bundle,
                    contract=contract,
                )
                return _build_report(
                    result,
                    bundle_report,
                    contract=contract,
                    contract_sha256=contract_sha256,
                    acceptance=acceptance,
                    historical=historical,
                    requested_path=None,
                    published=False,
                )
        except ReproductionOperationalError:
            raise
        except OSError as exc:
            raise ReproductionOperationalError(
                "staging_creation_failed", type(exc).__name__
            ) from exc

    _validate_save_directory(save_directory)
    try:
        save_directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{save_directory.name}.reproduce-",
            dir=save_directory.parent,
        ) as temporary:
            staging_bundle = Path(temporary) / "bundle"
            result, bundle_report, acceptance, historical = _run_staged(
                staging_bundle,
                contract=contract,
            )
            published = False
            if acceptance["passed"] is True:
                try:
                    _publish_bundle(staging_bundle, save_directory)
                except OSError as exc:
                    raise ReproductionOperationalError(
                        "bundle_publish_failed", type(exc).__name__
                    ) from exc
                published = True
            return _build_report(
                result,
                bundle_report,
                contract=contract,
                contract_sha256=contract_sha256,
                acceptance=acceptance,
                historical=historical,
                requested_path=save_directory,
                published=published,
            )
    except (ReproductionOperationalError, ReproductionUsageError):
        raise
    except OSError as exc:
        raise ReproductionOperationalError(
            "staging_creation_failed", type(exc).__name__
        ) from exc


def _error_report(
    reason_code: str,
    error_type: str,
    *,
    detail: str | None = None,
    artifact: object | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": "tiny-text-reproduction",
        "status": "error",
        "passed": False,
        "complete": False,
        "reason_codes": [reason_code],
        "package": {"name": PACKAGE_NAME, "version": __version__},
        "environment": {
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "error": {"reason_code": reason_code, "type": error_type},
    }
    if detail is not None:
        error = report["error"]
        if isinstance(error, dict):
            error["detail"] = detail
    if artifact is not None:
        report["artifact"] = artifact
    return report


def _failed_artifact(save_directory: Path | None) -> dict[str, object]:
    return {
        "disposition": (
            "temporary_deleted" if save_directory is None else "not_published"
        ),
        "requested_path": (
            str(save_directory) if save_directory is not None else None
        ),
        "published": False,
        "bundle_path": None,
    }


def _serialize_report(report: Mapping[str, Any]) -> str:
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _print_human(report: Mapping[str, Any]) -> None:
    acceptance = report["acceptance"]
    historical = report["historical_observations"]
    training = report["training_result"]
    artifact = report["artifact"]
    if not all(
        isinstance(value, Mapping)
        for value in (acceptance, historical, training, artifact)
    ):
        raise TypeError("reproduction report sections must be objects")
    acceptance = dict(acceptance)
    historical = dict(historical)
    training = dict(training)
    artifact = dict(artifact)
    reduction = float(training["initial_eval_loss"]) - float(training["final_eval_loss"])
    threshold = _FIXED_MINIMUM_VALIDATION_LOSS_REDUCTION
    print(f"nano-deepseek-v4 reproduce ({CONTRACT_ID})")
    print(f"overall: {str(report['status']).upper()}")
    print(f"loss reduction: {reduction:.4f} > {threshold:.4f}")
    print(
        "bundle verification: "
        f"{'PASS' if artifact['verification_complete'] else 'FAIL'}"
    )
    print(f"historical observations: {historical['status']}")
    if artifact["published"]:
        print(f"bundle: {artifact['bundle_path']}")
    else:
        print(f"bundle: {artifact['disposition']}")
    if acceptance["reason_codes"]:
        print("failures: " + ", ".join(acceptance["reason_codes"]))
    print(
        "scope: fixed tiny CPU learning and native bundle persistence; "
        "not model quality or official-weight parity"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and adjudicate the wheel-bundled 20-step tiny-text CPU reproduction."
        ),
    )
    parser.add_argument(
        "--save-directory",
        type=Path,
        help=(
            "atomically retain a passing bundle here; otherwise verify and remove "
            "a temporary bundle"
        ),
    )
    parser.add_argument("--output", type=Path, help="atomically write the JSON receipt")
    parser.add_argument("--json", action="store_true", help="emit the complete JSON receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_output_location(args.save_directory, args.output)
        report, exit_code = run_reproduction(save_directory=args.save_directory)
    except ReproductionUsageError as exc:
        parser.error(str(exc))
    except ReproductionContractError as exc:
        report = _error_report(
            exc.reason_code,
            type(exc).__name__,
            detail=exc.detail,
        )
        exit_code = EXIT_EXECUTION_ERROR
    except ReproductionOperationalError as exc:
        report = _error_report(
            exc.reason_code,
            exc.cause_type,
            artifact=_failed_artifact(args.save_directory),
        )
        exit_code = EXIT_EXECUTION_ERROR

    try:
        payload = _serialize_report(report)
    except (TypeError, ValueError) as exc:
        report = _error_report(
            "report_serialization_failed",
            type(exc).__name__,
            artifact=report.get("artifact"),
        )
        payload = _serialize_report(report)
        exit_code = EXIT_EXECUTION_ERROR

    if args.output is not None:
        try:
            _write_json_receipt(args.output, payload)
        except OSError as exc:
            report = _error_report(
                "output_write_failed",
                type(exc).__name__,
                artifact=report.get("artifact"),
            )
            payload = _serialize_report(report)
            exit_code = EXIT_EXECUTION_ERROR

    if args.json:
        print(payload, end="")
    elif report["status"] == "error":
        error = report["error"]
        assert isinstance(error, Mapping)
        print(
            "nano-deepseek-v4 reproduce: ERROR "
            f"({error['reason_code']}, {error['type']})",
            file=sys.stderr,
        )
    else:
        _print_human(report)
        if args.output is not None:
            print(f"receipt: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
