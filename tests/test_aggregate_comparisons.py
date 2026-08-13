from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

from nano_deepseek_v4.aggregate_comparisons import (
    _aggregator_sha256,
    _summarize,
    aggregate_bundle_comparison_reports,
    main,
)
from nano_deepseek_v4.compare_bundles import _evaluator_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _loss(mean: float) -> dict[str, Any]:
    return {
        "mean": mean,
        "sample_std": 0.1,
        "standard_error": 0.05,
    }


def _paired(mean: float) -> dict[str, Any]:
    return {
        **_loss(mean),
        "descriptive_normal_approx_95pct_interval": [mean - 0.1, mean + 0.1],
    }


def _comparison(run: str, total: float, *, has_mtp: bool = True) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator_sha256": _evaluator_sha256(),
        "source": "corpus.txt",
        "corpus_sha256": _digest("corpus"),
        "python_version": "3.11.15",
        "torch_version": "2.13.0+cpu",
        "machine": "aarch64",
        "device": "cpu",
        "validation_seed": 1338,
        "batches": 4,
        "batch_size": 2,
        "context_length": 16,
        "left": {
            "label": "hybrid",
            "bundle_path": f"{run}-left",
            "manifest_sha256": _digest(f"{run}-left-manifest"),
            "config_sha256": _digest("left-config"),
            "parameter_count": 100,
            "total_loss": _loss(2.0),
            "causal_lm_loss": _loss(1.5),
            "mtp_loss": _loss(1.0) if has_mtp else None,
        },
        "right": {
            "label": "all_sliding",
            "bundle_path": f"{run}-right",
            "manifest_sha256": _digest(f"{run}-right-manifest"),
            "config_sha256": _digest("right-config"),
            "parameter_count": 90,
            "total_loss": _loss(2.0 - total),
            "causal_lm_loss": _loss(1.5 - total),
            "mtp_loss": _loss(1.0 - total) if has_mtp else None,
        },
        "paired_left_minus_right_total": _paired(total),
        "paired_left_minus_right_causal_lm": _paired(total / 2),
        "paired_left_minus_right_mtp": _paired(total / 3) if has_mtp else None,
        "interval_kind": "descriptive_normal_approximation_over_overlapping_windows",
        "independent_window_assumption": False,
        "training_seed_variance_measured": False,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_aggregate_summarizes_distinct_compatible_runs(tmp_path: Path) -> None:
    first = _write_report(tmp_path / "first.json", _comparison("first", -0.03))
    second = _write_report(tmp_path / "second.json", _comparison("second", 0.01))

    result = aggregate_bundle_comparison_reports(
        [("seed-2", second), ("seed-1", first)],
    )

    assert result.run_ids == ("seed-1", "seed-2")
    assert result.run_count == 2
    assert result.paired_left_minus_right_total.mean == pytest.approx(-0.01)
    assert result.paired_left_minus_right_total.sample_std == pytest.approx(
        0.0282842712474619,
    )
    assert result.paired_left_minus_right_total.negative_count == 1
    assert result.paired_left_minus_right_total.positive_count == 1
    assert result.paired_left_minus_right_causal_lm.mean == pytest.approx(-0.005)
    assert result.paired_left_minus_right_mtp is not None
    assert result.paired_left_minus_right_mtp.mean == pytest.approx(-0.0033333333)
    assert result.comparison_evaluator_matches_installed is True
    assert result.python_version == "3.11.15"
    assert result.run_independence_verified is False
    assert result.training_seed_variance_measured is False
    assert result.runs[0].report_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()


def test_aggregate_uses_cross_python_stable_sample_statistics(tmp_path: Path) -> None:
    effects = [
        0.00308168339543046,
        -0.0015039648395031684,
        0.002467820653691886,
    ]
    reports = [
        (
            f"seed-{index}",
            _write_report(
                tmp_path / f"seed-{index}.json",
                _comparison(f"seed-{index}", effect),
            ),
        )
        for index, effect in enumerate(effects)
    ]

    summary = aggregate_bundle_comparison_reports(reports).paired_left_minus_right_total

    assert summary.mean.hex() == "0x1.61812e2222237p-10"
    assert summary.sample_std.hex() == "0x1.46477c6799f31p-9"
    assert summary.standard_error.hex() == "0x1.78c13ba610053p-10"


def test_aggregate_preserves_fmean_for_cancellation_sensitive_values() -> None:
    values = [-4989873172751189.0, 8194925119364804.0, 9655709520753060.0]

    summary = _summarize(values)

    assert summary.mean == statistics.fmean(values)
    assert summary.mean == 4286920489122225.5


def test_aggregate_preserves_composed_standard_error() -> None:
    values = [-4821664994140733.0, 225494427372170.0, -1901317250991714.0]

    summary = _summarize(values)

    assert summary.standard_error == summary.sample_std / math.sqrt(len(values))
    assert summary.standard_error == 1462979780415965.5


def test_aggregate_falls_back_to_exact_mean_when_fmean_overflows() -> None:
    summary = _summarize([1e308, 1e308])

    assert summary.mean == 1e308
    assert summary.sample_std == 0.0
    assert summary.standard_error == 0.0


def test_aggregate_supports_reports_without_mtp(tmp_path: Path) -> None:
    first = _write_report(
        tmp_path / "first.json",
        _comparison("first", -0.03, has_mtp=False),
    )
    second = _write_report(
        tmp_path / "second.json",
        _comparison("second", 0.01, has_mtp=False),
    )

    result = aggregate_bundle_comparison_reports([("a", first), ("b", second)])

    assert result.paired_left_minus_right_mtp is None


def test_aggregate_rejects_contract_drift(tmp_path: Path) -> None:
    first = _write_report(tmp_path / "first.json", _comparison("first", -0.03))
    changed = _comparison("second", 0.01)
    changed["context_length"] = 32
    second = _write_report(tmp_path / "second.json", changed)

    with pytest.raises(ValueError, match="common contract: context_length"):
        aggregate_bundle_comparison_reports([("a", first), ("b", second)])


def test_aggregate_rejects_python_runtime_drift(tmp_path: Path) -> None:
    first = _write_report(tmp_path / "first.json", _comparison("first", -0.03))
    changed = _comparison("second", 0.01)
    changed["python_version"] = "9.99.0"
    second = _write_report(tmp_path / "second.json", changed)

    with pytest.raises(ValueError, match="common contract: python_version"):
        aggregate_bundle_comparison_reports([("a", first), ("b", second)])


def test_aggregate_rejects_reused_bundle_manifest(tmp_path: Path) -> None:
    first_payload = _comparison("first", -0.03)
    second_payload = _comparison("second", 0.01)
    second_payload["left"]["manifest_sha256"] = first_payload["left"]["manifest_sha256"]
    first = _write_report(tmp_path / "first.json", first_payload)
    second = _write_report(tmp_path / "second.json", second_payload)

    with pytest.raises(ValueError, match="distinct left and right bundle manifests"):
        aggregate_bundle_comparison_reports([("a", first), ("b", second)])


def test_aggregate_rejects_bundle_reused_on_the_opposite_side(tmp_path: Path) -> None:
    first_payload = _comparison("first", -0.03)
    second_payload = _comparison("second", 0.01)
    second_payload["right"]["manifest_sha256"] = first_payload["left"][
        "manifest_sha256"
    ]
    first = _write_report(tmp_path / "first.json", first_payload)
    second = _write_report(tmp_path / "second.json", second_payload)

    with pytest.raises(ValueError, match="distinct left and right bundle manifests"):
        aggregate_bundle_comparison_reports([("a", first), ("b", second)])


def test_aggregate_rejects_too_few_or_duplicate_run_ids(tmp_path: Path) -> None:
    first = _write_report(tmp_path / "first.json", _comparison("first", -0.03))
    second = _write_report(tmp_path / "second.json", _comparison("second", 0.01))

    with pytest.raises(ValueError, match="at least two"):
        aggregate_bundle_comparison_reports([("a", first)])
    with pytest.raises(ValueError, match="duplicate run_id: a"):
        aggregate_bundle_comparison_reports([("a", first), ("a", second)])


def test_aggregate_rejects_non_finite_effect(tmp_path: Path) -> None:
    first_payload = _comparison("first", -0.03)
    first_payload["paired_left_minus_right_total"]["mean"] = float("nan")
    first = _write_report(tmp_path / "first.json", first_payload)
    second = _write_report(tmp_path / "second.json", _comparison("second", 0.01))

    with pytest.raises(ValueError, match="must be a finite number"):
        aggregate_bundle_comparison_reports([("a", first), ("b", second)])


def test_aggregate_cli_writes_same_json_as_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _write_report(tmp_path / "first.json", _comparison("first", -0.03))
    second = _write_report(tmp_path / "second.json", _comparison("second", 0.01))
    output = tmp_path / "nested" / "aggregate.json"

    assert main(
        [
            "--report",
            f"seed-1={first}",
            "--report",
            f"seed-2={second}",
            "--output",
            str(output),
            "--json",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == stdout
    assert json.loads(stdout)["run_ids"] == ["seed-1", "seed-2"]


def test_aggregator_sha_matches_source() -> None:
    source = Path(__file__).parents[1] / "nano_deepseek_v4" / "aggregate_comparisons.py"
    assert _aggregator_sha256() == hashlib.sha256(source.read_bytes()).hexdigest()
