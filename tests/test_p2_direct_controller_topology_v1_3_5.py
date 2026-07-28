from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import p2_direct_controller_contract_v1_3_5 as contract  # noqa: E402
import p2_direct_controller_topology_v1_3_5 as topology  # noqa: E402
import probe_p2_direct_controller_topology_v1_3_5 as probe  # noqa: E402
import run_p2_direct_controller_matrix_v1_3 as matrix  # noqa: E402


def _phase_a(
    capacities: tuple[float, float, float, float],
    *,
    admissible: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> list[dict[str, Any]]:
    return [
        {
            "worker_count": worker_count,
            "primary_capacity_work_units_per_second": capacity,
            "hard_admissible": allowed,
            "quality_values_accessed": False,
        }
        for worker_count, capacity, allowed in zip(
            probe.CANDIDATE_WORKER_COUNTS,
            capacities,
            admissible,
            strict=True,
        )
    ]


def test_system_probe_selector_uses_smallest_near_best_parallel_count() -> None:
    assert probe._select_worker_count(_phase_a((1.0, 2.0, 2.9, 3.0))) == 3


def test_system_probe_selector_falls_back_when_acceleration_is_not_useful() -> None:
    assert probe._select_worker_count(_phase_a((1.0, 1.4, 1.45, 1.3))) == 1


def test_system_probe_selector_rejects_quality_access_or_missing_reference() -> None:
    quality_observing = _phase_a((1.0, 2.0, 3.0, 4.0))
    quality_observing[2]["quality_values_accessed"] = True
    with pytest.raises(ValueError, match="quality-observing"):
        probe._select_worker_count(quality_observing)

    with pytest.raises(ValueError, match="one-worker reference"):
        probe._select_worker_count(
            _phase_a(
                (1.0, 2.0, 3.0, 4.0),
                admissible=(False, True, True, True),
            )
        )


def test_activation_root_is_exact_two_and_binds_persistent_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "activation"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    lock = root / contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.name
    receipt = root / contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name
    for path in (lock, receipt):
        path.write_bytes(b"")
        path.chmod(0o600)
    canonical = tmp_path / "canonical-activation"

    root_binding, lock_binding, activation_path = topology._activation_root_members(
        root,
        canonical_root=canonical,
    )

    assert root_binding["path"] == str(canonical)
    assert lock_binding["path"] == str(canonical / lock.name)
    assert lock_binding["persistent_inode"] is True
    assert lock_binding["unlink_on_release"] is False
    assert activation_path == receipt

    extra = root / "unexpected"
    extra.write_bytes(b"")
    extra.chmod(0o600)
    with pytest.raises(ValueError, match="exact-two"):
        topology._activation_root_members(root, canonical_root=canonical)


def test_v1_3_5_borrowed_gpu_view_cannot_release_supervisor_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gpu.lock"
    owner = matrix.acquire_gpu_lock("supervisor", path=path)
    borrowed = topology.borrow_gpu_lock_lease(owner)
    try:
        assert borrowed is not owner
        assert borrowed.fileno() == owner.fileno()
        assert topology.active_gpu_lock_borrows(owner) == 1
        borrowed.close()
        assert topology.active_gpu_lock_borrows(owner) == 0
        owner.assert_held()
        with pytest.raises(RuntimeError, match="already locked"):
            matrix.acquire_gpu_lock("contender", path=path)
    finally:
        if not borrowed.closed:
            borrowed.close()
        topology.close_gpu_lock_owner(owner)

    reacquired = matrix.acquire_gpu_lock("reacquired", path=path)
    reacquired.close()


def test_generated_matrix_lock_serializes_distinct_worker_threads(
    tmp_path: Path,
) -> None:
    lock = (tmp_path / ".matrix.lock").resolve()
    summary = (tmp_path / "matrix.json").resolve()
    lock.write_bytes(b"")
    lock.chmod(0o600)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    failures: list[BaseException] = []

    def first() -> None:
        try:
            with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
                first_entered.set()
                assert release_first.wait(timeout=5)
        except BaseException as error:
            failures.append(error)

    def second() -> None:
        try:
            with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
                second_entered.set()
        except BaseException as error:
            failures.append(error)

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread.start()
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    assert second_entered.wait(timeout=5)
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not failures


def test_same_gpu_worker_group_drains_failure_and_releases_every_view(
    tmp_path: Path,
) -> None:
    scheduler = matrix.acquire_gpu_lock("supervisor", path=tmp_path / "gpu.lock")
    guard = matrix.acquire_gpu_lock("supervisor-guard", path=tmp_path / "guard.lock")
    ready = threading.Barrier(3)
    peer_drained = [threading.Event(), threading.Event()]

    def worker(
        worker_index: int,
        worker_scheduler: matrix.GPULockLease,
        worker_guard: matrix.GPULockLease,
        drain: threading.Event,
    ) -> dict[str, Any]:
        worker_scheduler.assert_held()
        worker_guard.assert_held()
        ready.wait(timeout=5)
        if worker_index == 0:
            raise ValueError("synthetic worker failure")
        assert drain.wait(timeout=5)
        peer_drained[worker_index - 1].set()
        return {"worker_index": worker_index}

    try:
        with pytest.raises(RuntimeError, match="worker 0.*synthetic worker failure"):
            matrix._run_same_gpu_worker_group(
                worker_count=3,
                scheduler_owner=scheduler,
                device_guard_owner=guard,
                run_worker=worker,
            )
        assert all(event.is_set() for event in peer_drained)
        assert topology.active_gpu_lock_borrows(scheduler) == 0
        assert topology.active_gpu_lock_borrows(guard) == 0
        scheduler.assert_held()
        guard.assert_held()
    finally:
        topology.close_gpu_lock_owner(guard)
        topology.close_gpu_lock_owner(scheduler)
