from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import re
import runpy
import socket
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import nano_deepseek_v4.dspark as dspark
from nano_deepseek_v4.dspark import (
    build_dspark_vector_fixture,
    load_packaged_dspark_vector,
    run_dspark_conformance,
    run_native_dspark,
)
from nano_deepseek_v4.dspark_oracle import (
    DSparkOracleFixture,
    fixture_tensor_dict,
    result_tensor_dict,
    run_dspark_oracle,
)

_FLOAT_ATOL = 1e-6
_FLOAT_RTOL = 1e-6
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VECTOR_RESOURCE = "_receipts/dspark-semantic-v1.json"
_VECTOR_SHA256 = "2a6f485b4dd8017f5a55e314d0feeb970e3bf5460a2f7233712e354e98717e82"
_FIXTURE_SHA256 = "6aff9f6a25e38603a358aeddcec9e411c1f961f5b3c8736111b3508b69a2138e"
_EXPECTED_DRAFT_TOKEN_IDS = [[6, 11, 7, 11, 7], [3, 11, 2, 11, 7]]
_EXPECTED_SOURCE = {
    "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "model_source_sha256": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
    "readme_sha256": "252acafdc9204d0dba3fde1b0a93d71cd1664a4ceadfe222b60117ed0ccc56ff",
    "paper": "arXiv:2607.05147v1",
    "paper_markdown_sha256": "6a0b9338cf1b6eb062a2a73b2bd91831fd3eae542683e1b24801c067acb654e4",
}
_MUTATION_CASES: tuple[tuple[dspark.DSparkMutation, str], ...] = (
    ("causal_mask", "noncausal_attention"),
    ("omit_markov_bias", "markov_bias"),
    ("current_token_markov", "previous_token_markov"),
    ("reverse_target_layers", "target_layer_order"),
    ("confidence_current_token", "confidence_previous_token"),
    ("omit_confidence_sigmoid", "confidence_sigmoid"),
)
_ORACLE_COMPUTATION_NAMES = frozenset(
    {"rms_norm_fp32", "run_dense_noncausal_stage", "run_dspark_oracle"}
)
_NATIVE_FUNCTION_NAMES = frozenset({"_native_rms_norm", "_native_dense_stage", "run_native_dspark"})


def _named_tensors(value: object, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            child = getattr(value, field.name)
            child_prefix = f"{prefix}.{field.name}" if prefix else field.name
            yield from _named_tensors(child, child_prefix)
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _named_tensors(value[key], child_prefix)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _named_tensors(child, child_prefix)


def _tensor_snapshot(value: object) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in _named_tensors(value)}


def _assert_tensor_mappings_equal(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        actual_tensor = actual[name]
        expected_tensor = expected[name]
        assert actual_tensor.dtype == expected_tensor.dtype, name
        assert actual_tensor.shape == expected_tensor.shape, name
        if actual_tensor.is_floating_point():
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                atol=_FLOAT_ATOL,
                rtol=_FLOAT_RTOL,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        else:
            assert torch.equal(actual_tensor, expected_tensor), name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_mapping_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key]
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fixture_sha256(value: DSparkOracleFixture) -> str:
    digest = hashlib.sha256()
    digest.update(b"nano-deepseek-v4:dspark-fixture-v1\0")
    digest.update(
        json.dumps(
            asdict(value.config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(bytes.fromhex(_tensor_mapping_sha256(fixture_tensor_dict(value))))
    return digest.hexdigest()


def _imports_module(source: str, module_suffix: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(module_suffix):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.endswith(module_suffix) for alias in node.names
        ):
            return True
    return False


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _native_oracle_computation_calls(source: str) -> set[str]:
    """Resolve direct and module-alias calls into the independent oracle."""

    tree = ast.parse(source)
    direct_aliases: dict[str, str] = {}
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("dspark_oracle"):
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("dspark_oracle"):
                for alias in node.names:
                    if alias.name == "*":
                        direct_aliases.update((name, name) for name in _ORACLE_COMPUTATION_NAMES)
                    elif alias.name in _ORACLE_COMPUTATION_NAMES:
                        direct_aliases[alias.asname or alias.name] = alias.name
            elif any(alias.name == "dspark_oracle" for alias in node.names):
                for alias in node.names:
                    if alias.name == "dspark_oracle":
                        module_aliases.add(alias.asname or alias.name)

    target_functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _NATIVE_FUNCTION_NAMES
    }
    references: set[str] = set()
    for function_name, function in target_functions.items():
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            called = _dotted_name(node.func)
            if called in direct_aliases:
                references.add(f"{function_name}:{direct_aliases[called]}")
            for module_alias in module_aliases:
                for computation in _ORACLE_COMPUTATION_NAMES:
                    if called == f"{module_alias}.{computation}":
                        references.add(f"{function_name}:{computation}")
    return references


def _strict_json_loads(payload: str) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise AssertionError(f"non-finite JSON constant: {value}")

    return json.loads(payload, parse_constant=reject_nonfinite)


def _packaged_payload() -> dict[str, Any]:
    resource = resources.files("nano_deepseek_v4").joinpath(*_VECTOR_RESOURCE.split("/"))
    return json.loads(resource.read_text(encoding="utf-8"))


def _redirect_packaged_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: Mapping[str, Any],
) -> None:
    package_root = tmp_path / "nano_deepseek_v4"
    resource = package_root.joinpath(*_VECTOR_RESOURCE.split("/"))
    resource.parent.mkdir(parents=True)
    resource.write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(dspark, "resource_files", lambda package: package_root)


def test_native_and_oracle_equations_are_structurally_independent():
    oracle_path = Path(inspect.getsourcefile(run_dspark_oracle) or "")
    oracle_source = oracle_path.read_text(encoding="utf-8")
    native_source = Path(dspark.__file__ or "").read_text(encoding="utf-8")

    assert not _imports_module(oracle_source, "dspark")
    assert not _imports_module(oracle_source, "modeling")
    assert not _imports_module(oracle_source, "config")
    assert _native_oracle_computation_calls(native_source) == set()


@pytest.mark.parametrize(
    "source",
    [
        """
from nano_deepseek_v4.dspark_oracle import run_dspark_oracle as solve
def run_native_dspark(fixture):
    return solve(fixture.config, fixture.inputs, fixture.weights)
""",
        """
import nano_deepseek_v4.dspark_oracle as oracle
def _native_dense_stage(*args):
    return oracle.run_dense_noncausal_stage(*args)
""",
        """
from . import dspark_oracle as oracle
def _native_rms_norm(*args):
    return oracle.rms_norm_fp32(*args)
""",
    ],
)
def test_structural_independence_detector_rejects_oracle_call_aliases(source: str):
    assert _native_oracle_computation_calls(source)


def test_packaged_vector_is_portable_complete_and_hash_bound():
    resource = resources.files("nano_deepseek_v4").joinpath(*_VECTOR_RESOURCE.split("/"))
    assert resource.is_file()
    payload = json.loads(resource.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["kind"] == "dspark-semantic-vectors"
    assert set(payload) == {
        "schema_version",
        "kind",
        "vector_set_id",
        "fixture_sha256",
        "expected_sha256",
        "source",
        "config",
        "inputs",
        "weights",
        "expected",
        "claim_boundary",
    }
    assert payload["fixture_sha256"] == _FIXTURE_SHA256
    assert _SHA256.fullmatch(payload["expected_sha256"])
    assert payload["source"] == _EXPECTED_SOURCE
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == _VECTOR_SHA256

    fixture = load_packaged_dspark_vector()
    rebuilt = build_dspark_vector_fixture()
    assert rebuilt.config == fixture.config
    assert fixture.config.target_layer_count == 3
    assert fixture.config.stage_count == 3
    assert fixture.config.block_size == 5
    assert _fixture_sha256(fixture) == _FIXTURE_SHA256
    oracle = run_dspark_oracle(fixture.config, fixture.inputs, fixture.weights)
    assert payload["expected_sha256"] == _tensor_mapping_sha256(fixture.expected)
    _assert_tensor_mappings_equal(
        result_tensor_dict(oracle),
        fixture.expected,
    )
    _assert_tensor_mappings_equal(
        _tensor_snapshot(rebuilt),
        _tensor_snapshot(fixture),
    )

    report = run_dspark_conformance(fixture)
    expected_hashes = {
        "vector_sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        "native_source_sha256": _sha256(Path(dspark.__file__ or "")),
        "oracle_source_sha256": _sha256(Path(inspect.getsourcefile(run_dspark_oracle) or "")),
    }
    for key, expected in expected_hashes.items():
        assert report.hashes[key] == expected
    assert _SHA256.fullmatch(report.hashes["config_sha256"])
    assert report.hashes["fixture_sha256"] == _FIXTURE_SHA256
    assert set(report.hashes) == {
        "vector_sha256",
        "native_source_sha256",
        "oracle_source_sha256",
        "config_sha256",
        "fixture_sha256",
    }
    assert all(_SHA256.fullmatch(value) for value in report.hashes.values())


def test_fixture_identity_binds_the_semantic_config():
    fixture = build_dspark_vector_fixture()
    changed = replace(
        fixture,
        config=replace(
            fixture.config,
            rms_norm_eps=fixture.config.rms_norm_eps * 2,
        ),
    )

    assert _fixture_sha256(fixture) == fixture.fixture_sha256
    assert _fixture_sha256(changed) != fixture.fixture_sha256
    report = run_dspark_conformance(changed)
    assert report.passed is False
    assert report.checks["fixture_immutable"] is False


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("model_id", "deepseek-ai/not-the-pinned-model"),
        ("revision", "0" * 40),
        ("config_sha256", "0" * 64),
        ("model_source_sha256", "0" * 64),
        ("readme_sha256", "0" * 64),
        ("paper", "arXiv:0000.00000v0"),
        ("paper_markdown_sha256", "0" * 64),
    ],
)
def test_loader_rejects_any_wrong_pinned_source_identity(
    field: str,
    wrong_value: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    payload = _packaged_payload()
    payload["source"][field] = wrong_value
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)

    with pytest.raises(ValueError, match="source provenance"):
        load_packaged_dspark_vector()


@pytest.mark.parametrize("wrong_value", [1.5, True])
def test_loader_rejects_non_integer_json_token_ids(
    wrong_value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    payload = _packaged_payload()
    payload["inputs"]["previous_token_ids"][0] = wrong_value
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)

    with pytest.raises(ValueError, match="only JSON integers"):
        load_packaged_dspark_vector()


@pytest.mark.parametrize("wrong_value", [5.5, True])
def test_loader_rejects_non_integer_json_config_fields(
    wrong_value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    payload = _packaged_payload()
    payload["config"]["block_size"] = wrong_value
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)

    with pytest.raises(ValueError, match="positive integer"):
        load_packaged_dspark_vector()


def test_loader_rejects_boolean_json_float_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    payload = _packaged_payload()
    payload["inputs"]["target_hidden_states"][0][0][0][0] = True
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)

    with pytest.raises(ValueError, match="finite JSON numbers"):
        load_packaged_dspark_vector()


def test_native_equations_match_the_independent_oracle_without_mutating_fixture():
    fixture = build_dspark_vector_fixture()
    before = _tensor_snapshot(fixture)

    native = run_native_dspark(fixture)
    oracle = run_dspark_oracle(fixture.config, fixture.inputs, fixture.weights)

    _assert_tensor_mappings_equal(
        _tensor_snapshot(native),
        _tensor_snapshot(oracle),
    )
    _assert_tensor_mappings_equal(_tensor_snapshot(fixture), before)


def test_native_projection_normalization_and_shared_heads_follow_explicit_equations():
    fixture = build_dspark_vector_fixture()
    result = run_native_dspark(fixture)

    expected_concat = torch.cat(fixture.inputs.target_hidden_states, dim=-1)
    torch.testing.assert_close(
        result.target_hidden_concat,
        expected_concat,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    expected_projected = F.linear(expected_concat, fixture.weights.main_projection)
    torch.testing.assert_close(
        result.projected_main_hidden,
        expected_projected,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    expected_main_context = expected_projected * torch.rsqrt(
        expected_projected.square().mean(dim=-1, keepdim=True) + fixture.config.rms_norm_eps
    )
    expected_main_context = expected_main_context * fixture.weights.main_norm
    torch.testing.assert_close(
        result.main_context,
        expected_main_context,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    expected_draft_embeddings = F.embedding(
        result.draft_input_ids,
        fixture.weights.token_embedding,
    ) + fixture.weights.draft_position_embedding.unsqueeze(0)
    torch.testing.assert_close(
        result.draft_embeddings,
        expected_draft_embeddings,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    expected_base_logits = F.linear(
        result.normalized_final_hidden_states,
        fixture.weights.lm_head,
    )
    torch.testing.assert_close(
        result.base_logits,
        expected_base_logits,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )


def test_first_draft_slot_is_influenced_by_a_future_draft_slot():
    fixture = build_dspark_vector_fixture()
    changed_positions = fixture.weights.draft_position_embedding.clone()
    changed_positions[-1] += 0.5
    perturbed = replace(
        fixture,
        weights=replace(
            fixture.weights,
            draft_position_embedding=changed_positions,
        ),
    )

    baseline = run_native_dspark(fixture)
    changed = run_native_dspark(perturbed)

    torch.testing.assert_close(
        baseline.draft_embeddings[:, 0],
        changed.draft_embeddings[:, 0],
        atol=0.0,
        rtol=0.0,
    )
    first_slot_delta = (
        (
            baseline.stage_results[0].hidden_states[:, 0]
            - changed.stage_results[0].hidden_states[:, 0]
        )
        .abs()
        .max()
    )
    assert first_slot_delta.item() > 1e-5


def test_noncausal_check_rejects_a_finite_causal_mask():
    fixture = build_dspark_vector_fixture()
    masked = run_native_dspark(fixture, mutation="causal_mask")
    finite_stages = tuple(
        replace(
            stage,
            attention_logits=torch.where(
                torch.isneginf(stage.attention_logits),
                torch.full_like(stage.attention_logits, -1e9),
                stage.attention_logits,
            ),
        )
        for stage in masked.stage_results
    )
    finite_masked = replace(masked, stage_results=finite_stages)

    assert all(
        bool(torch.isfinite(stage.attention_logits).all()) for stage in finite_masked.stage_results
    )
    assert dspark._has_full_draft_attention(finite_masked, fixture) is False


def test_markov_chain_and_confidence_use_the_previous_token_sequentially():
    fixture = build_dspark_vector_fixture()
    result = run_native_dspark(fixture)

    for position in range(fixture.config.block_size):
        previous_token = result.token_chain[:, position]
        expected_embedding = F.embedding(previous_token, fixture.weights.markov_w1)
        expected_bias = F.linear(expected_embedding, fixture.weights.markov_w2)
        expected_logits = result.base_logits[:, position] + expected_bias
        torch.testing.assert_close(
            result.markov_embeddings[:, position],
            expected_embedding,
            atol=_FLOAT_ATOL,
            rtol=_FLOAT_RTOL,
        )
        torch.testing.assert_close(
            result.markov_bias_logits[:, position],
            expected_bias,
            atol=_FLOAT_ATOL,
            rtol=_FLOAT_RTOL,
        )
        torch.testing.assert_close(
            result.logits[:, position],
            expected_logits,
            atol=_FLOAT_ATOL,
            rtol=_FLOAT_RTOL,
        )
        assert torch.equal(
            result.token_chain[:, position + 1],
            expected_logits.argmax(dim=-1),
        )

    confidence_features = torch.cat(
        (result.final_hidden_states, result.markov_embeddings),
        dim=-1,
    )
    expected_confidence_logits = F.linear(
        confidence_features,
        fixture.weights.confidence,
    ).squeeze(-1)
    torch.testing.assert_close(
        result.confidence_logits,
        expected_confidence_logits,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    torch.testing.assert_close(
        result.confidence_probabilities,
        torch.sigmoid(expected_confidence_logits),
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )


def test_conformance_report_carries_exact_tokens_checks_and_claim_boundary():
    report = run_dspark_conformance()

    assert report.schema_version == 1
    assert report.kind == "dspark-semantic-conformance"
    assert report.status == "pass"
    assert report.passed is True
    assert report.mutation is None
    assert report.exact_draft_token_ids == report.expected_draft_token_ids
    assert report.exact_draft_token_ids == _EXPECTED_DRAFT_TOKEN_IDS
    assert report.exact_draft_token_ids == report.native_result.greedy_token_ids.tolist()
    assert report.checks
    assert all(report.checks.values())
    assert set(report.max_abs_errors) >= {
        "target_hidden_concat",
        "main_context",
        "attention_probabilities",
        "base_logits",
        "markov_bias_logits",
        "logits",
        "confidence_logits",
        "confidence_probabilities",
    }
    assert max(report.max_abs_errors.values(), default=0.0) <= _FLOAT_ATOL
    assert report.source["model_id"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert report.source["revision"]
    assert report.environment["device"] == "cpu"
    assert report.environment["dtype"] == "float32"
    assert report.environment["network_attempted"] is False
    assert "scheduler" in report.claim_boundary.lower()
    assert "accept" in report.claim_boundary.lower()

    compact = report.to_dict()
    assert compact["status"] == "pass"
    assert "native_result" not in compact
    assert "oracle_result" not in compact


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    _MUTATION_CASES,
)
def test_semantic_mutations_are_detected_by_their_own_check(
    mutation: dspark.DSparkMutation,
    failed_check: str,
):
    fixture = build_dspark_vector_fixture()
    before = _tensor_snapshot(fixture)
    report = run_dspark_conformance(fixture, mutation=mutation)

    assert report.status == "fail"
    assert report.passed is False
    assert report.mutation == mutation
    assert report.checks[failed_check] is False
    assert any(not passed for passed in report.checks.values())
    serialized = dspark._json_payload(report)
    compact = _strict_json_loads(serialized)
    assert compact["mutation"] == mutation
    assert compact["status"] == "fail"
    assert all(math.isfinite(value) for value in compact["max_abs_errors"].values())
    _assert_tensor_mappings_equal(_tensor_snapshot(fixture), before)


def test_conformance_is_offline_even_when_network_entry_points_are_traps(monkeypatch):
    attempts: list[tuple[object, ...]] = []

    def forbidden_connection(*args: object, **kwargs: object) -> None:
        attempts.append((*args, kwargs))
        raise AssertionError("DSpark conformance attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden_connection)
    monkeypatch.setattr(socket.socket, "connect", forbidden_connection)

    report = run_dspark_conformance()

    assert report.passed
    assert attempts == []
    assert report.environment["network_attempted"] is False


def test_cli_json_success_is_compact_and_machine_readable(capsys):
    assert dspark.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["passed"] is True
    assert "native_result" not in payload
    assert "oracle_result" not in payload


def test_cli_can_write_the_compact_receipt(tmp_path, capsys):
    output = tmp_path / "dspark.json"

    assert dspark.main(["--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["hashes"]["fixture_sha256"] == _FIXTURE_SHA256
    assert capsys.readouterr().err == ""


def test_cli_returns_one_for_a_semantic_mismatch(monkeypatch, capsys):
    failed = run_dspark_conformance(mutation="causal_mask")
    monkeypatch.setattr(
        dspark,
        "run_dspark_conformance",
        lambda *args, **kwargs: failed,
    )

    assert dspark.main(["--json"]) == 1
    captured = capsys.readouterr()
    payload = _strict_json_loads(captured.out)
    assert payload["status"] == "fail"
    assert payload["mutation"] == "causal_mask"
    assert captured.err == ""


def test_vector_generator_check_mode_is_rng_independent_and_non_mutating(
    tmp_path,
    capsys,
):
    generator_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_dspark_vectors.py"
    namespace = runpy.run_path(str(generator_path))
    generator_main = namespace["main"]
    checked_in = Path(__file__).resolve().parents[1] / "nano_deepseek_v4" / _VECTOR_RESOURCE
    checked_in_before = checked_in.read_bytes()

    assert generator_main(["--check", "--output", str(checked_in)]) == 0
    assert checked_in.read_bytes() == checked_in_before
    assert capsys.readouterr().err == ""

    drifted = json.loads(checked_in_before)
    drifted["expected"]["base_logits"][0][0][0] += 5e-7
    drifted_tensors = namespace["_expected_tensor_mapping"](drifted["expected"])
    assert drifted_tensors is not None
    drifted["expected_sha256"] = namespace["_tensor_stream_sha256"](
        drifted_tensors
    )
    drifted_bytes = (
        json.dumps(drifted, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    generator_main.__globals__["_payload_bytes"] = lambda: drifted_bytes

    assert generator_main(["--check", "--output", str(checked_in)]) == 0
    assert checked_in.read_bytes() == checked_in_before
    assert capsys.readouterr().err == ""

    canonical = b'{"fixture":"canonical"}\n'
    generator_main.__globals__["_payload_bytes"] = lambda: canonical
    output = tmp_path / "vectors.json"

    output.write_bytes(canonical)
    assert generator_main(["--check", "--output", str(output)]) == 0
    assert output.read_bytes() == canonical
    assert capsys.readouterr().err == ""

    stale = b'{"fixture":"stale"}\n'
    output.write_bytes(stale)
    assert generator_main(["--check", "--output", str(output)]) == 1
    assert output.read_bytes() == stale
    assert "do not match the generator" in capsys.readouterr().err


def test_cli_returns_four_for_harness_errors_without_a_traceback(monkeypatch, capsys):
    def broken_harness(*args: object, **kwargs: object) -> None:
        raise ValueError("broken fixture")

    monkeypatch.setattr(dspark, "run_dspark_conformance", broken_harness)

    assert dspark.main(["--json"]) == 4
    captured = capsys.readouterr()
    assert captured.err == "DSpark conformance failed: ValueError\n"
    assert "broken fixture" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_returns_four_when_output_cannot_be_written(tmp_path, capsys):
    output_directory = tmp_path / "report-directory"
    output_directory.mkdir()

    assert dspark.main(["--json", "--output", str(output_directory)]) == 4
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
