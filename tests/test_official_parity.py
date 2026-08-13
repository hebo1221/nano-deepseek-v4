from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nano_deepseek_v4.official_parity import run_transformers_parity

_RECEIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "transformers-v5.15.0-parity.json"
)


def test_checked_in_transformers_parity_receipt_is_complete():
    receipt = json.loads(_RECEIPT_PATH.read_text())

    assert receipt["schema_version"] == 3
    assert receipt["reference_package"] == "transformers"
    assert receipt["reference_version"] == "5.15.0"
    assert receipt["reference_git_tag"] == "v5.15.0"
    assert len(receipt["reference_git_commit"]) == 40
    assert len(receipt["reference_modeling_sha256"]) == 64
    assert len(receipt["reference_config_sha256"]) == 64
    assert len(receipt["reference_rope_utils_sha256"]) == 64
    assert receipt["reference_git_tag"] in receipt["reference_config_source_url"]
    assert receipt["reference_git_tag"] in receipt["reference_rope_utils_source_url"]
    assert len(receipt["mtp_equation_source_revision"]) == 40
    assert len(receipt["mtp_equation_source_sha256"]) == 64
    assert receipt["mtp_equation_source_revision"] in receipt["mtp_equation_source_url"]
    assert len(receipt["native_modeling_sha256"]) == 64
    modeling_path = _RECEIPT_PATH.parents[1] / "nano_deepseek_v4" / "modeling.py"
    assert receipt["native_modeling_sha256"] == hashlib.sha256(
        modeling_path.read_bytes()
    ).hexdigest()
    assert receipt["reference_tensor_count"] == 89
    assert receipt["mapped_tensor_count"] == 104
    assert receipt["gradient_tensor_count"] == 101
    assert receipt["rope_position_ids"] == [0, 1, 127, 65535, 65536, 1048575]
    assert receipt["rope_dimension"] == 64
    assert receipt["main_rope_theta"] == 10_000.0
    assert receipt["compressed_rope_theta"] == 160_000.0
    assert receipt["rope_scaling"] == {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65_536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    assert receipt["main_rope_inv_freq"]["element_count"] == 32
    assert receipt["compressed_rope_inv_freq"]["element_count"] == 32
    assert receipt["main_rope_inv_freq"]["max_abs_error"] <= 1e-6
    assert receipt["compressed_rope_inv_freq"]["max_abs_error"] <= 1e-6
    assert receipt["main_rope"]["max_abs_error"] <= 1e-6
    assert receipt["compressed_rope"]["max_abs_error"] <= 1e-6
    assert receipt["full_forward"]["max_abs_error"] == 0.0
    assert receipt["passed"] is True
    for name in (
        "full_forward",
        "main_rope_inv_freq",
        "compressed_rope_inv_freq",
        "main_rope",
        "compressed_rope",
        "cached_decode",
        "reference_cache_equivalence",
        "native_cache_equivalence",
        "backward_gradients",
        "mtp_residual_streams",
        "mtp_hidden_states",
    ):
        assert receipt[name]["allclose"] is True


def test_pinned_full_cache_backward_and_mtp_parity():
    pytest.importorskip("transformers")

    report = run_transformers_parity()

    assert report.reference_version == "5.15.0"
    assert report.reference_tensor_count == 89
    assert report.mapped_tensor_count == 104
    assert report.gradient_tensor_count == 101
    assert report.rope_position_ids == (0, 1, 127, 65_535, 65_536, 1_048_575)
    assert report.rope_dimension == 64
    assert report.rope_scaling["factor"] == 16
    assert report.main_rope_inv_freq.max_abs_error <= 1e-6
    assert report.compressed_rope_inv_freq.max_abs_error <= 1e-6
    assert report.main_rope.max_abs_error <= 1e-6
    assert report.compressed_rope.max_abs_error <= 1e-6
    assert report.full_forward.max_abs_error == 0.0
    assert report.cached_decode.max_abs_error <= 1e-6
    assert report.reference_cache_equivalence.max_abs_error <= 1e-6
    assert report.native_cache_equivalence.max_abs_error <= 1e-6
    assert report.backward_gradients.max_abs_error <= 1e-6
    assert report.mtp_residual_streams.max_abs_error <= 1e-6
    assert report.mtp_hidden_states.max_abs_error <= 1e-6
    assert report.passed
