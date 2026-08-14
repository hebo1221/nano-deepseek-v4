from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import nano_deepseek_v4.attention_reach as attention_reach
from nano_deepseek_v4.attention_reach import (
    _build_gradient_cotangent,
    _build_probe_batch,
    build_attention_reach_configs,
    main,
    run_attention_reach,
)

_RECEIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "attention-reach-conformance.json"
)
_DISTANCES = (16, 64, 94, 95, 128, 224)
_SOURCE_POSITIONS = (240, 192, 162, 161, 128, 32)


def _assert_finite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)


@pytest.fixture(scope="module")
def reach_execution():
    rng_before = torch.random.get_rng_state().clone()
    report = run_attention_reach()
    rng_after = torch.random.get_rng_state()
    return report, rng_before, rng_after


def test_fixed_protocol_geometry_and_parameter_matched_configs():
    configs = build_attention_reach_configs()
    local = configs["local"].to_dict()
    full_window = configs["full_window"].to_dict()

    assert set(configs) == {"hybrid", "local", "full_window"}
    assert configs["hybrid"].layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert configs["local"].layer_types == ["sliding_attention"] * 3
    assert configs["full_window"].layer_types == ["sliding_attention"] * 3
    assert all(config.num_nextn_predict_layers == 0 for config in configs.values())
    assert all(config.mtp_layer_types == [] for config in configs.values())
    assert configs["hybrid"].moe_intermediate_size == 192
    assert configs["local"].moe_intermediate_size == 197
    assert configs["local"].sliding_window == 32
    assert configs["full_window"].sliding_window == 256
    assert all(config.pad_token_id == 0 for config in configs.values())
    assert all(config.bos_token_id == 1 for config in configs.values())
    assert all(config.eos_token_id == 2 for config in configs.values())
    assert all(config.partial_rotary_factor == 0.5 for config in configs.values())
    assert [key for key in local if local[key] != full_window[key]] == [
        "sliding_window"
    ]


def test_probe_batch_is_deterministic_and_mutates_exactly_one_source_token():
    probes, batch = _build_probe_batch(input_seed=20_260_811, probe_count=3)
    same_probes, same_batch = _build_probe_batch(input_seed=20_260_811, probe_count=3)
    other_probes, other_batch = _build_probe_batch(input_seed=20_260_812, probe_count=3)

    assert torch.equal(probes, same_probes)
    assert torch.equal(batch, same_batch)
    assert not torch.equal(probes, other_probes)
    assert not torch.equal(batch, other_batch)
    assert tuple(probes.shape) == (3, 256)
    assert tuple(batch.shape) == (3 * (1 + len(_DISTANCES)), 256)
    assert int(probes.min()) >= 3
    assert int(probes.max()) < 64

    grouped = batch.view(3, 1 + len(_DISTANCES), 256)
    assert torch.equal(grouped[:, 0], probes)
    for probe_idx in range(3):
        for offset, source_position in enumerate(_SOURCE_POSITIONS, start=1):
            changed = grouped[probe_idx, offset].ne(probes[probe_idx]).nonzero().flatten()
            assert changed.tolist() == [source_position]
            replacement = int(grouped[probe_idx, offset, source_position])
            original = int(probes[probe_idx, source_position])
            assert 3 <= replacement < 64
            assert replacement != original


def test_gradient_cotangent_is_deterministic_independent_and_rademacher():
    cotangent = _build_gradient_cotangent(gradient_seed=424_242, probe_count=3)
    same = _build_gradient_cotangent(gradient_seed=424_242, probe_count=3)
    other = _build_gradient_cotangent(gradient_seed=424_243, probe_count=3)

    assert torch.equal(cotangent, same)
    assert not torch.equal(cotangent, other)
    assert tuple(cotangent.shape) == (3, 64)
    assert set(cotangent.unique().tolist()) == {-1.0, 1.0}


def test_default_cpu_conformance_matches_theoretical_boundary(reach_execution):
    report, rng_before, rng_after = reach_execution

    assert report.passed
    assert torch.equal(rng_before, rng_after)
    assert report.distances == _DISTANCES
    assert report.query_position == 255
    assert report.local_max_source_lag == 93
    assert report.first_disconnected_distance == 95
    assert report.influence_atol == 1e-6
    assert report.gradient_seed == 424_242
    assert report.gradient_objective == "fixed Rademacher VJP of final-position logits"
    assert report.local_full_config_difference == ("sliding_window",)
    assert report.local_full_state_dict_identical
    assert report.hybrid_minus_local_parameter_count == 60
    assert report.variants["hybrid"].parameter_count == 930_405
    assert report.variants["local"].parameter_count == 930_345
    assert report.variants["full_window"].parameter_count == 930_345

    for name in ("hybrid", "full_window"):
        variant = report.variants[name]
        assert variant.passed
        assert variant.perturbation_passed
        assert variant.gradient_passed
        assert all(
            item.materially_changed_probe_count == report.probe_count
            for item in variant.perturbation_observations
        )
        assert all(
            item.nonzero_probe_count == report.probe_count
            for item in variant.gradient_observations
        )

    local = report.variants["local"]
    assert [item.source_position for item in local.perturbation_observations] == list(
        _SOURCE_POSITIONS
    )
    assert [item.source_lag for item in local.perturbation_observations] == [
        15,
        63,
        93,
        94,
        127,
        223,
    ]
    assert [
        item.theoretically_reachable for item in local.perturbation_observations
    ] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert [
        item.materially_changed_probe_count
        for item in local.perturbation_observations
    ] == [
        5,
        5,
        5,
        0,
        0,
        0,
    ]
    for item in local.perturbation_observations[3:]:
        assert item.max_linf_delta <= report.influence_atol
    assert [item.nonzero_probe_count for item in local.gradient_observations] == [
        5,
        5,
        5,
        0,
        0,
        0,
    ]
    assert local.gradient_observations[2].min_l2_gradient > 0
    for item in local.gradient_observations[3:]:
        assert item.max_l2_gradient == 0.0


def test_report_is_strict_finite_json_with_bound_source_hashes(reach_execution):
    report = reach_execution[0]
    payload = report.to_dict()
    root = Path(__file__).resolve().parents[1]

    _assert_finite(payload)
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(serialized)["passed"] is True
    assert len(report.protocol_sha256) == 64
    assert len(report.probe_input_sha256) == 64
    assert len(report.mutation_batch_sha256) == 64
    assert len(report.gradient_cotangent_sha256) == 64
    for relative_path, digest in report.source_sha256.items():
        assert len(digest) == 64
        assert digest == hashlib.sha256((root / relative_path).read_bytes()).hexdigest()


def test_checked_in_receipt_binds_protocol_topology_and_current_sources(
    reach_execution,
):
    receipt = json.loads(_RECEIPT_PATH.read_text())
    live = reach_execution[0].to_dict()
    root = _RECEIPT_PATH.parents[1]

    assert receipt["schema_version"] == 2
    assert receipt["name"] == "native-attention-path-reachability"
    assert receipt["passed"] is True
    assert receipt["distances"] == list(_DISTANCES)
    assert receipt["local_max_source_lag"] == 93
    assert receipt["first_disconnected_distance"] == 95
    assert receipt["influence_atol"] == 1e-6
    assert receipt["hybrid_minus_local_parameter_count"] == 60
    assert receipt["local_full_config_difference"] == ["sliding_window"]
    assert receipt["local_full_state_dict_identical"] is True
    for field in (
        "protocol_sha256",
        "model_seed",
        "input_seed",
        "gradient_seed",
        "probe_count",
        "probe_input_sha256",
        "mutation_batch_sha256",
        "gradient_cotangent_sha256",
    ):
        assert receipt[field] == live[field]
    for name, expected_count in (
        ("hybrid", 930_405),
        ("local", 930_345),
        ("full_window", 930_345),
    ):
        assert receipt["variants"][name]["parameter_count"] == expected_count
        assert receipt["variants"][name]["passed"] is True
        assert (
            receipt["variants"][name]["config_sha256"]
            == live["variants"][name]["config_sha256"]
        )
    for relative_path, digest in receipt["source_sha256"].items():
        assert digest == hashlib.sha256((root / relative_path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_seed": -1}, "model_seed"),
        ({"input_seed": True}, "input_seed"),
        ({"gradient_seed": -1}, "gradient_seed"),
        ({"probe_count": 0}, "probe_count"),
        ({"probe_count": 65}, "probe_count"),
        ({"device": "meta"}, "device"),
    ],
)
def test_invalid_run_arguments_fail_before_model_construction(monkeypatch, kwargs, message):
    def unexpected_construction():
        raise AssertionError("model configs must not be built")

    monkeypatch.setattr(
        attention_reach,
        "build_attention_reach_configs",
        unexpected_construction,
    )
    with pytest.raises(ValueError, match=message):
        run_attention_reach(**kwargs)


def test_json_cli_writes_byte_identical_report(monkeypatch, reach_execution, tmp_path, capsys):
    report = reach_execution[0]
    monkeypatch.setattr(attention_reach, "run_attention_reach", lambda **_: report)
    output = tmp_path / "nested" / "report.json"

    return_code = main(["--json", "--output", str(output)])
    stdout = capsys.readouterr().out

    assert return_code == 0
    assert stdout == output.read_text()
    assert json.loads(stdout)["passed"] is True


def test_human_cli_states_boundary_and_returns_one_on_contract_failure(
    monkeypatch,
    reach_execution,
    capsys,
):
    report = reach_execution[0]
    monkeypatch.setattr(attention_reach, "run_attention_reach", lambda **_: report)
    assert main([]) == 0
    stdout = capsys.readouterr().out
    assert "distance 94 reachable; distance 95 disconnected" in stdout
    assert "embedding VJP" in stdout
    assert "structural influence only" in stdout

    monkeypatch.setattr(
        attention_reach,
        "run_attention_reach",
        lambda **_: replace(report, passed=False),
    )
    assert main([]) == 1
    assert "overall: FAIL" in capsys.readouterr().out
