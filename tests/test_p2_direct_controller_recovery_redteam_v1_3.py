from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_direct_controller_matrix_v1_3 as matrix  # noqa: E402


def _disk_payload(registry: list[dict[str, int]]) -> dict[str, Any]:
    payload: dict[str, Any] = {field: None for field in matrix._MATRIX_FIELDS}
    payload.update(
        {
            "status": "in_progress",
            "worker_ledger_registry": registry,
            "persistent_session_ledger": {},
        }
    )
    return payload


def _install_predecessor_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_ledgers: dict[int, list[str]],
    current_gpu_bindings: dict[int, dict[str, int]],
    raw_registry: list[dict[str, int]],
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        matrix,
        "_load_worker_ledgers",
        lambda **_kwargs: (current_ledgers, current_gpu_bindings),
    )
    monkeypatch.setattr(matrix, "_validate_gpu_lease_binding", lambda value: dict(value))

    def historical_registry(ledgers: dict[int, list[str]], **_kwargs: Any) -> list[dict[str, int]]:
        observed["historical_ledgers"] = ledgers
        return raw_registry

    monkeypatch.setattr(matrix, "_historical_worker_ledger_registry", historical_registry)

    def merged_records(
        ledgers: dict[int, list[str]], **_kwargs: Any
    ) -> tuple[list[str], dict[int, str]]:
        merged = {
            worker_index + sequence_index * 2: record
            for worker_index in sorted(ledgers)
            for sequence_index, record in enumerate(ledgers[worker_index])
        }
        return [merged[index] for index in sorted(merged)], merged

    monkeypatch.setattr(matrix, "_merge_worker_records", merged_records)
    monkeypatch.setattr(matrix, "coordinates", lambda: (object(), object(), object()))
    monkeypatch.setattr(matrix, "_worker_ledger_root_binding", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        matrix,
        "_matrix_payload",
        lambda *_args, **_kwargs: {
            "payload_sha256": "a" * 64,
            "attestation": {},
            "persistent_session_ledger": {},
        },
    )
    monkeypatch.setattr(
        matrix,
        "_attested_payload",
        lambda *_args, **_kwargs: _disk_payload(raw_registry),
    )
    monkeypatch.setattr(
        matrix, "_matrix_body_without_session_projection", lambda _value: {"same": True}
    )
    monkeypatch.setattr(
        matrix,
        "_validate_persistent_session_ledger_snapshot",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        matrix,
        "_crosscheck_matrix_records_with_session_ledger",
        lambda records, *_args, **_kwargs: observed.setdefault("crosschecked_records", records),
    )
    return observed


@pytest.mark.parametrize(
    ("current_ledgers", "raw_registry", "expected_historical"),
    [
        (
            {0: ["old-0", "new-0"], 1: ["old-1"]},
            [
                {"worker_index": 0, "completed_shards": 1},
                {"worker_index": 1, "completed_shards": 1},
            ],
            {0: ["old-0"], 1: ["old-1"]},
        ),
        (
            {0: ["old-0"], 1: []},
            [{"worker_index": 0, "completed_shards": 1}],
            {0: ["old-0"]},
        ),
    ],
    ids=("one-worker-plus-one", "new-authenticated-empty-worker"),
)
def test_stale_distributed_predecessor_accepts_only_the_two_exact_crash_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_ledgers: dict[int, list[str]],
    raw_registry: list[dict[str, int]],
    expected_historical: dict[int, list[str]],
) -> None:
    gpu_bindings = {index: {"gpu": index} for index in current_ledgers}
    observed = _install_predecessor_stubs(
        monkeypatch,
        current_ledgers=current_ledgers,
        current_gpu_bindings=gpu_bindings,
        raw_registry=raw_registry,
    )

    matrix._validate_stale_distributed_matrix_predecessor(
        _disk_payload(raw_registry),
        layout=SimpleNamespace(output_root=tmp_path.resolve()),
        worker_count=2,
        prerequisites=SimpleNamespace(trust_root=object()),
        evaluator_script=tmp_path / "evaluator.py",
        evaluator_binding={},
        matrix_lock_binding={},
        expected_gpu_worker_leases=gpu_bindings,
    )

    assert observed["historical_ledgers"] == expected_historical
    assert observed["crosschecked_records"] == [
        record for index in sorted(expected_historical) for record in expected_historical[index]
    ]


@pytest.mark.parametrize(
    ("current_ledgers", "raw_registry"),
    [
        (
            {0: ["old", "new-1", "new-2"], 1: []},
            [
                {"worker_index": 0, "completed_shards": 1},
                {"worker_index": 1, "completed_shards": 0},
            ],
        ),
        (
            {0: ["old-0", "new-0"], 1: ["old-1", "new-1"]},
            [
                {"worker_index": 0, "completed_shards": 1},
                {"worker_index": 1, "completed_shards": 1},
            ],
        ),
        (
            {0: ["old-0"], 1: ["new-nonempty"]},
            [{"worker_index": 0, "completed_shards": 1}],
        ),
    ],
    ids=("plus-two", "two-workers-advance", "new-worker-nonempty"),
)
def test_stale_distributed_predecessor_rejects_ambiguous_advancement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_ledgers: dict[int, list[str]],
    raw_registry: list[dict[str, int]],
) -> None:
    gpu_bindings = {index: {"gpu": index} for index in current_ledgers}
    _install_predecessor_stubs(
        monkeypatch,
        current_ledgers=current_ledgers,
        current_gpu_bindings=gpu_bindings,
        raw_registry=raw_registry,
    )

    with pytest.raises(ValueError, match="exact one-step worker-ledger predecessor"):
        matrix._validate_stale_distributed_matrix_predecessor(
            _disk_payload(raw_registry),
            layout=SimpleNamespace(output_root=tmp_path.resolve()),
            worker_count=2,
            prerequisites=SimpleNamespace(trust_root=object()),
            evaluator_script=tmp_path / "evaluator.py",
            evaluator_binding={},
            matrix_lock_binding={},
            expected_gpu_worker_leases=gpu_bindings,
        )


def _blocked_disk_payload(
    raw_registry: list[dict[str, int]],
    *,
    declared_claims: list[int],
) -> dict[str, Any]:
    reason: dict[str, Any] = {field: None for field in matrix._INFRASTRUCTURE_DRAIN_FIELDS}
    reason.update(
        {
            "pause_worker_index": 0,
            "active_claim_coordinate_indices": declared_claims,
            "filesystem_available_bytes": 123,
            "signal_type": "infrastructure-storage-headroom-pause-v1",
        }
    )
    payload: dict[str, Any] = {field: None for field in matrix._DRAINING_MATRIX_FIELDS}
    payload.update(
        {
            "status": "draining_infrastructure",
            "worker_ledger_registry": raw_registry,
            "globally_committed_shards": 1,
            "persistent_session_ledger": {},
            "infrastructure_drain": reason,
        }
    )
    return payload


def _install_blocked_predecessor_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_ledgers: dict[int, list[str]],
    raw_registry: list[dict[str, int]],
    canonical_disk: dict[str, Any],
) -> tuple[dict[int, dict[str, int]], dict[str, Any], dict[str, Any]]:
    gpu_bindings = {index: {"gpu": index} for index in current_ledgers}
    current_resume = {
        "status": "in_progress",
        "globally_committed_shards": 2,
    }
    rebound = {"status": "draining_infrastructure", "rebound": True}
    observed: dict[str, Any] = {}
    coordinate_items = [object(), object(), object(), object()]
    historical_resume = {"status": "in_progress", "historical": True}
    historical_reconstructed = {
        "status": "in_progress",
        "persistent_session_ledger": {"stale": True},
        "payload_sha256": "a" * 64,
        "attestation": {},
    }

    monkeypatch.setattr(
        matrix,
        "_load_worker_ledgers",
        lambda **_kwargs: (current_ledgers, gpu_bindings),
    )
    monkeypatch.setattr(matrix, "_validate_gpu_lease_binding", lambda value: dict(value))
    monkeypatch.setattr(
        matrix,
        "_historical_worker_ledger_registry",
        lambda *_args, **_kwargs: raw_registry,
    )

    def merge(ledgers: dict[int, list[str]], **_kwargs: Any) -> tuple[list[str], dict[int, str]]:
        merged = {0: ledgers[0][0]}
        if ledgers[1]:
            merged[1] = ledgers[1][0]
        return [merged[index] for index in sorted(merged)], merged

    monkeypatch.setattr(matrix, "_merge_worker_records", merge)
    monkeypatch.setattr(matrix, "coordinates", lambda: tuple(coordinate_items))
    monkeypatch.setattr(matrix, "_worker_ledger_root_binding", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        matrix, "_matrix_payload", lambda *_args, **_kwargs: historical_reconstructed
    )
    monkeypatch.setattr(
        matrix,
        "_validate_persistent_session_ledger_snapshot",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        matrix,
        "_crosscheck_matrix_records_with_session_ledger",
        lambda records, *_args, **_kwargs: observed.setdefault("historical_records", list(records)),
    )
    monkeypatch.setattr(matrix, "_attested_payload", lambda *_args, **_kwargs: historical_resume)
    monkeypatch.setattr(
        matrix,
        "_assigned_coordinates",
        lambda *, worker_index, worker_count: (
            (coordinate_items[0], coordinate_items[2])
            if (worker_index, worker_count) == (0, 2)
            else (coordinate_items[1], coordinate_items[3])
        ),
    )
    monkeypatch.setattr(
        matrix,
        "_headroom_signal_payload",
        lambda **kwargs: {
            "coordinate": kwargs["coordinate"],
            "filesystem_available_bytes": kwargs["filesystem_available_bytes"],
        },
    )

    def historical_draining(
        resume: dict[str, Any],
        *,
        signal: dict[str, Any],
        worker_index: int,
        active_claim_coordinate_indices: list[int],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        observed["historical_draining"] = {
            "resume": resume,
            "signal": signal,
            "worker_index": worker_index,
            "declared_claims": active_claim_coordinate_indices,
        }
        return canonical_disk

    monkeypatch.setattr(matrix, "_draining_matrix_payload", historical_draining)
    monkeypatch.setattr(
        matrix,
        "_preflight_distributed_output_tree",
        lambda **_kwargs: {2: {"claim": "still-live"}},
    )
    monkeypatch.setattr(
        matrix,
        "_distributed_summary_from_ledgers",
        lambda *_args, **_kwargs: (
            current_resume,
            {0: current_ledgers[0][0], 1: current_ledgers[1][0]},
        ),
    )

    def rebind(**kwargs: Any) -> dict[str, Any]:
        observed["rebind"] = kwargs
        return rebound

    monkeypatch.setattr(matrix, "_rebind_infrastructure_after_commit", rebind)
    return gpu_bindings, rebound, observed


def test_draining_predecessor_accepts_one_exact_predeclared_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_registry = [
        {"worker_index": 0, "completed_shards": 1},
        {"worker_index": 1, "completed_shards": 0},
    ]
    disk = _blocked_disk_payload(raw_registry, declared_claims=[1, 2])
    gpu_bindings, rebound, observed = _install_blocked_predecessor_stubs(
        monkeypatch,
        current_ledgers={0: ["old-0"], 1: ["new-1"]},
        raw_registry=raw_registry,
        canonical_disk=disk,
    )

    result = matrix._validate_stale_distributed_matrix_predecessor(
        disk,
        layout=SimpleNamespace(
            output_root=tmp_path.resolve(), matrix_summary=tmp_path / "matrix.json"
        ),
        worker_count=2,
        prerequisites=SimpleNamespace(trust_root=object()),
        evaluator_script=tmp_path / "evaluator.py",
        evaluator_binding={},
        matrix_lock_binding={},
        expected_gpu_worker_leases=gpu_bindings,
    )

    assert result is rebound
    assert observed["historical_records"] == ["old-0"]
    assert observed["historical_draining"]["declared_claims"] == [1, 2]
    assert observed["rebind"]["active_claim_coordinate_indices"] == (2,)


def test_paused_predecessor_rejects_worker_ledger_advancement(
    tmp_path: Path,
) -> None:
    payload: dict[str, Any] = {field: None for field in matrix._PAUSED_MATRIX_FIELDS}
    payload["status"] = "paused_infrastructure"

    with pytest.raises(ValueError, match="Only an active or draining matrix"):
        matrix._validate_stale_distributed_matrix_predecessor(
            payload,
            layout=SimpleNamespace(output_root=tmp_path.resolve()),
            worker_count=2,
            prerequisites=SimpleNamespace(trust_root=object()),
            evaluator_script=tmp_path / "evaluator.py",
            evaluator_binding={},
            matrix_lock_binding={},
            expected_gpu_worker_leases={},
        )


@pytest.mark.parametrize(
    ("current_ledgers", "declared_claims", "canonical_reason_drift", "message"),
    [
        (
            {0: ["old-0"], 1: ["new-1", "new-3"]},
            [1, 2],
            False,
            "exact one-step worker-ledger predecessor",
        ),
        (
            {0: ["old-0"], 1: ["new-1"]},
            [2],
            False,
            "predeclared live claim",
        ),
        (
            {0: ["old-0"], 1: ["new-1"]},
            [1, 2],
            True,
            "exact authenticated blocked state",
        ),
    ],
    ids=("plus-two", "undeclared-commit", "reason-drift"),
)
def test_draining_predecessor_rejects_unreachable_or_tampered_advancement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_ledgers: dict[int, list[str]],
    declared_claims: list[int],
    canonical_reason_drift: bool,
    message: str,
) -> None:
    raw_registry = [
        {"worker_index": 0, "completed_shards": 1},
        {"worker_index": 1, "completed_shards": 0},
    ]
    disk = _blocked_disk_payload(raw_registry, declared_claims=declared_claims)
    canonical_disk = disk
    if canonical_reason_drift:
        canonical_disk = _blocked_disk_payload(raw_registry, declared_claims=declared_claims)
        canonical_disk["infrastructure_drain"] = {
            **canonical_disk["infrastructure_drain"],
            "signal_type": "infrastructure-storage-headroom-pause-v1",
        }
        disk = {
            **canonical_disk,
            "infrastructure_drain": {
                **canonical_disk["infrastructure_drain"],
                "signal_type": "tampered-signal",
            },
        }
    gpu_bindings, _rebound, _observed = _install_blocked_predecessor_stubs(
        monkeypatch,
        current_ledgers=current_ledgers,
        raw_registry=raw_registry,
        canonical_disk=canonical_disk,
    )

    with pytest.raises(ValueError, match=message):
        matrix._validate_stale_distributed_matrix_predecessor(
            disk,
            layout=SimpleNamespace(
                output_root=tmp_path.resolve(), matrix_summary=tmp_path / "matrix.json"
            ),
            worker_count=2,
            prerequisites=SimpleNamespace(trust_root=object()),
            evaluator_script=tmp_path / "evaluator.py",
            evaluator_binding={},
            matrix_lock_binding={},
            expected_gpu_worker_leases=gpu_bindings,
        )


def test_distributed_recovery_publishes_rebound_blocked_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []
    summary = tmp_path / "matrix.json"
    summary.write_text("{}\n", encoding="utf-8")
    disk_payload = {
        "worker_ledger_registry": [{"old": True}],
        "globally_committed_shards": 1,
    }
    expected_payload = {
        "worker_ledger_registry": [{"new": True}],
        "globally_committed_shards": 2,
        "records": [],
    }
    rebound_payload = {
        **expected_payload,
        "status": "draining_infrastructure",
        "infrastructure_drain": {"preserved": True},
    }
    gpu_bindings = {0: {"gpu": 0}}

    monkeypatch.setattr(matrix, "_load_json_nofollow", lambda *_args, **_kwargs: disk_payload)
    monkeypatch.setattr(matrix, "_verify_attested_payload", lambda *_args, **_kwargs: None)

    def recover(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append(("validate-predecessor", disk_payload))
        return rebound_payload

    monkeypatch.setattr(matrix, "_validate_stale_distributed_matrix_predecessor", recover)
    monkeypatch.setattr(
        matrix,
        "_atomic_write_json",
        lambda _path, payload: events.append(("write-matrix", payload)),
    )
    monkeypatch.setattr(
        matrix,
        "validate_matrix_summary",
        lambda payload, *_args, **_kwargs: events.append(("validate-written", payload)),
    )

    result = matrix._validate_distributed_disk_summary(
        layout=SimpleNamespace(matrix_summary=summary, output_root=tmp_path.resolve()),
        expected_payload=expected_payload,
        worker_count=1,
        prerequisites=SimpleNamespace(trust_root=object()),
        evaluator_script=tmp_path / "evaluator.py",
        evaluator_binding={},
        matrix_lock_binding={},
        expected_gpu_worker_leases=gpu_bindings,
    )

    assert result == rebound_payload
    assert events == [
        ("validate-predecessor", disk_payload),
        ("write-matrix", rebound_payload),
        ("validate-written", rebound_payload),
    ]


def test_full_stale_recovery_replays_every_bundle_before_matrix_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    summary = tmp_path / "matrix.json"
    summary.write_text("{}\n", encoding="utf-8")
    disk_payload = {
        "worker_ledger_registry": [{"old": True}],
        "globally_committed_shards": matrix.EXPECTED_SHARDS - 1,
    }
    expected_payload = {
        "worker_ledger_registry": [{"new": True}],
        "globally_committed_shards": matrix.EXPECTED_SHARDS,
        "records": [],
    }
    replayed_ledgers = {0: [object()]}
    replayed_gpu_bindings = {0: {"gpu": 0}}
    merged = {index: object() for index in range(matrix.EXPECTED_SHARDS)}

    monkeypatch.setattr(matrix, "_load_json_nofollow", lambda *_args, **_kwargs: disk_payload)
    monkeypatch.setattr(matrix, "_verify_attested_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        matrix,
        "_validate_stale_distributed_matrix_predecessor",
        lambda *_args, **_kwargs: events.append("validate-predecessor"),
    )

    def load_ledgers(**kwargs: Any) -> tuple[dict[int, list[object]], dict[int, dict[str, int]]]:
        assert kwargs["verify_bundles"] is True
        events.append("replay-all-bundles")
        return replayed_ledgers, replayed_gpu_bindings

    monkeypatch.setattr(matrix, "_load_worker_ledgers", load_ledgers)

    def rebuild(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[int, object]]:
        events.append("rebuild-after-replay")
        return expected_payload, merged

    monkeypatch.setattr(matrix, "_distributed_summary_from_ledgers", rebuild)
    monkeypatch.setattr(
        matrix,
        "_atomic_write_json",
        lambda *_args, **_kwargs: events.append("write-matrix"),
    )
    monkeypatch.setattr(
        matrix,
        "validate_matrix_summary",
        lambda *_args, **_kwargs: events.append("validate-written-matrix"),
    )

    result = matrix._validate_distributed_disk_summary(
        layout=SimpleNamespace(matrix_summary=summary, output_root=tmp_path.resolve()),
        expected_payload=expected_payload,
        worker_count=1,
        prerequisites=SimpleNamespace(trust_root=object()),
        evaluator_script=tmp_path / "evaluator.py",
        evaluator_binding={},
        matrix_lock_binding={},
        expected_gpu_worker_leases=replayed_gpu_bindings,
    )

    assert result == expected_payload
    assert events == [
        "validate-predecessor",
        "replay-all-bundles",
        "rebuild-after-replay",
        "write-matrix",
        "validate-written-matrix",
    ]
