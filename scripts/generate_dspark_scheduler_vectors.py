#!/usr/bin/env python3
"""Generate the fixed CPU vectors for the DSpark prefix scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from nano_deepseek_v4.dspark_scheduler_oracle import (
    oracle_causal_greedy,
    oracle_lagged_topk,
)

_DEFAULT_OUTPUT = Path("nano_deepseek_v4/_receipts/dspark-scheduler-v1.json")
_VECTOR_SET_ID = "dspark-scheduler-v1"
_TIE_BREAK = "survival descending, one-based position ascending, request index ascending"
_CLAIM_BOUNDARY = (
    "CPU arithmetic and causal-trace conformance for DSpark Algorithm 1 and Section 5.2 "
    "on supplied calibrated probability and SPS tables; not input temporal provenance "
    "or request-slot alignment, STS calibration, target rejection sampling, end-to-end "
    "distribution preservation, hardware profiling, optimized kernels, serving "
    "throughput, or model-quality evidence."
)
_SOURCE = {
    "paper": "arXiv:2607.05147v1",
    "paper_markdown_sha256": ("6a0b9338cf1b6eb062a2a73b2bd91831fd3eae542683e1b24801c067acb654e4"),
    "scheduler_sections": "Algorithm 1, Section 5.2, Appendix A",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sps(*rates: float, start: int) -> dict[str, float]:
    return {str(start + index): rate for index, rate in enumerate(rates)}


def _causal_case(
    case_id: str,
    confidence_probabilities: list[list[float]],
    steps_per_second: dict[str, float],
) -> dict[str, Any]:
    numeric_sps = {int(key): value for key, value in steps_per_second.items()}
    expected = oracle_causal_greedy(
        confidence_probabilities,
        numeric_sps,
    ).summary()
    return {
        "case_id": case_id,
        "mode": "causal_greedy",
        "confidence_probabilities": confidence_probabilities,
        "historical_confidence_probabilities": None,
        "steps_per_second": steps_per_second,
        "expected": expected,
    }


def _lagged_case(
    case_id: str,
    current_confidence_probabilities: list[list[float]],
    historical_confidence_probabilities: list[list[float]],
    steps_per_second: dict[str, float],
) -> dict[str, Any]:
    numeric_sps = {int(key): value for key, value in steps_per_second.items()}
    expected = oracle_lagged_topk(
        current_confidence_probabilities,
        historical_confidence_probabilities,
        numeric_sps,
    ).summary()
    return {
        "case_id": case_id,
        "mode": "lagged_topk",
        "confidence_probabilities": current_confidence_probabilities,
        "historical_confidence_probabilities": historical_confidence_probabilities,
        "steps_per_second": steps_per_second,
        "expected": expected,
    }


def build_payload() -> dict[str, Any]:
    cases = [
        _causal_case(
            "smooth-causal",
            [[0.9, 0.5, 0.5], [0.8, 0.75, 0.25]],
            _sps(1.0, 0.9, 0.8, 0.65, 0.55, 0.5, 0.45, start=2),
        ),
        _causal_case(
            "appendix-a-high",
            [[0.8, 0.9]],
            _sps(1.0, 0.5, 0.45, start=1),
        ),
        _causal_case(
            "appendix-a-low",
            [[0.8, 0.0]],
            _sps(1.0, 0.5, 0.45, start=1),
        ),
        _lagged_case(
            "jagged-lagged",
            [[0.55, 0.4, 0.4], [0.9, 0.85, 0.2]],
            [[0.95, 0.8, 0.8], [0.9, 0.5, 0.5]],
            _sps(1.0, 0.9, 0.6, 0.59, 0.45, 0.4, 0.35, start=2),
        ),
    ]
    fixture_payloads = [
        {key: case[key] for key in sorted(set(case) - {"expected"})} for case in cases
    ]
    expected_payloads = [case["expected"] for case in cases]
    return {
        "schema_version": 1,
        "kind": "dspark-scheduler-vectors",
        "vector_set_id": _VECTOR_SET_ID,
        "fixture_sha256": _sha256_json(fixture_payloads),
        "expected_sha256": _sha256_json(expected_payloads),
        "source": dict(_SOURCE),
        "tie_break": _TIE_BREAK,
        "claim_boundary": _CLAIM_BOUNDARY,
        "cases": cases,
    }


def _payload_bytes() -> bytes:
    return (json.dumps(build_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing output differs instead of rewriting it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = _payload_bytes()
    if args.check:
        try:
            observed = args.output.read_bytes()
        except OSError as exc:
            print(f"could not read DSpark scheduler vectors: {type(exc).__name__}", file=sys.stderr)
            return 1
        if observed != payload:
            print("packaged DSpark scheduler vectors do not match the generator", file=sys.stderr)
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
