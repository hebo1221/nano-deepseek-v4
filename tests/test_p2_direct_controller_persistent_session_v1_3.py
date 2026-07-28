from __future__ import annotations

import hashlib
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_attestation as attestation  # noqa: E402
import p2_direct_controller_contract_v1_3_5 as contract  # noqa: E402
import p2_direct_controller_persistent_session_v1_3 as session  # noqa: E402


@pytest.fixture
def trust_root() -> attestation.TrustRoot:
    key = bytes(range(64))
    return attestation.TrustRoot(key=key, key_id=attestation.derive_key_id(key))


def test_persistent_session_uses_only_v1_3_5_authorities_and_paths() -> None:
    assert (
        session.READY_ONLY_PREFLIGHT_SESSION_ROLE
        == (contract.V1_3_5_READY_ONLY_PREFLIGHT_SESSION_ROLE)
        == "ready_only_preflight"
    )
    assert session.QUALITY_SESSION_ROLE == contract.V1_3_5_QUALITY_SESSION_ROLE == "quality"
    assert session.SESSION_ROLES == frozenset({"ready_only_preflight", "quality"})
    assert session.PLAN_ATTESTATION_PURPOSE == (
        contract.V1_3_5_PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE
    )
    assert session.WORK_ATTESTATION_PURPOSE == (
        contract.V1_3_5_PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE
    )
    assert session.RESULT_ATTESTATION_PURPOSE == (
        contract.V1_3_5_PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE
    )
    assert session.RECEIPT_ATTESTATION_PURPOSE == (
        contract.V1_3_5_PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE
    )
    assert session.LAUNCH_LEDGER_ATTESTATION_PURPOSE == (
        contract.V1_3_5_PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE
    )
    assert session.TERMINAL_LEDGER_ATTESTATION_PURPOSE == (
        contract.V1_3_5_PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE
    )
    assert session._CANONICAL_OUTPUT_ROOT == contract.V1_3_5_OUTPUT_ROOT
    assert session._CANONICAL_SESSION_LEDGER_ROOT == (
        contract.V1_3_5_PERSISTENT_SESSION_LEDGER_ROOT
    )
    assert session._CANONICAL_SESSION_LEDGER_LOCK_PATH == (
        contract.V1_3_5_PERSISTENT_SESSION_LEDGER_LOCK_PATH
    )
    assert "v1-3-5" in session.SESSION_LEDGER_ROOT_SUFFIX
    assert "v1-3-1" not in session.SESSION_LEDGER_ROOT_SUFFIX
    assert "v1-3-3" not in session.SESSION_LEDGER_ROOT_SUFFIX
    purposes = (
        session.PLAN_ATTESTATION_PURPOSE,
        session.WORK_ATTESTATION_PURPOSE,
        session.RESULT_ATTESTATION_PURPOSE,
        session.RECEIPT_ATTESTATION_PURPOSE,
        session.LAUNCH_LEDGER_ATTESTATION_PURPOSE,
        session.TERMINAL_LEDGER_ATTESTATION_PURPOSE,
    )
    assert all("v1-3-5" in purpose for purpose in purposes)
    assert all("v1-3-2" not in purpose for purpose in purposes)
    assert all("v1-3-3" not in purpose for purpose in purposes)


def _plan(
    tmp_path: Path,
    trust_root: attestation.TrustRoot,
    *,
    coordinates: list[dict[str, int | str]] | None = None,
    session_digit: str = "1",
    authority_digit: str = "a",
    controlled_stop: bool = True,
    session_role: str = session.QUALITY_SESSION_ROLE,
) -> dict[str, object]:
    frozen = list(session.assigned_coordinates(worker_index=0, worker_count=1))
    selected = coordinates or frozen[:3]
    start = frozen.index(selected[0])
    return session.build_session_plan(
        selected,
        session_nonce=session_digit * 64,
        launch_authority_nonce=authority_digit * 64,
        worker_index=0,
        worker_count=1,
        assignment_completed_prefix=frozen[:start],
        max_new_cells_stop_limit=len(selected) if controlled_stop else None,
        input_binding_digest="b" * 64,
        canonical_evaluator_digest="c" * 64,
        gpu_lease_binding_digest="d" * 64,
        prerequisites_binding_digest="e" * 64,
        output_root=tmp_path.resolve(),
        trust_root=trust_root,
        session_role=session_role,
    )


def _result_sequence(
    tmp_path: Path,
    trust_root: attestation.TrustRoot,
    plan: dict[str, object],
    *,
    count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    work: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for index in range(count):
        envelope = tmp_path / f"shard-{index}.json"
        order = session.build_work_order(
            plan,
            sequence_index=index,
            launch_nonce=hashlib.sha256(f"launch-{index}".encode()).hexdigest(),
            envelope_path=envelope,
            projected_remaining_shards=9_000 - index,
            projected_remaining_token_rows=(
                contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES - index
            ),
            trust_root=trust_root,
        )
        result = session.build_work_result(
            plan,
            order,
            terminal_decision="INTEGRITY-PASS",
            envelope_binding={
                "path": str(envelope),
                "bytes": 123,
                "sha256": "4" * 64,
                "payload_sha256": "5" * 64,
                "attestation_mac": "6" * 64,
            },
            completed_in_session=index + 1,
            model_state_reset_count=index + 1,
            trust_root=trust_root,
        )
        work.append(order)
        results.append(result)
    return work, results


def _session_argv(plan: dict[str, object]) -> list[str]:
    return [
        "/sealed/python",
        "-I",
        "--scale",
        str(plan["scale"]),
        "--training-seed",
        str(plan["training_seed"]),
        "--launch-nonce",
        str(plan["session_nonce"]),
        "--persistent-session",
    ]


def _recovery_record(
    plan: dict[str, object],
    work_order: dict[str, object],
    work_result: dict[str, object] | None,
) -> dict[str, object]:
    plan_attestation = plan["attestation"]
    assert isinstance(plan_attestation, dict)
    return {
        "persistent_session_execution": {
            "protocol": "sealed-hmac-jsonl-model-resident-v1",
            "plan": {
                "session_nonce": plan["session_nonce"],
                "launch_authority_nonce": plan["launch_authority_nonce"],
                "session_role": plan["session_role"],
                "payload_sha256": plan["payload_sha256"],
                "attestation_mac": plan_attestation["mac"],
                "worker_index": plan["worker_index"],
                "worker_count": plan["worker_count"],
                "scale": plan["scale"],
                "training_seed": plan["training_seed"],
                "coordinate_count": plan["coordinate_count"],
                "coordinate_digest": plan["coordinate_digest"],
                "model_load_limit": plan["model_load_limit"],
                "child_full_historical_evidence_replay_count": plan[
                    "child_full_historical_evidence_replay_count"
                ],
            },
            "session_command": _session_argv(plan),
            "work_order": work_order,
            "work_result": work_result,
            "published_bundle_reingested_after_child_eof": work_result is None,
        }
    }


def test_plan_is_one_exact_ordered_worker_scale_seed_cohort(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    plan = _plan(tmp_path, trust_root)
    assert session.validate_session_plan(plan, trust_root=trust_root) == plan
    assert plan["coordinate_count"] == 3
    assert plan["model_load_limit"] == 1
    assert plan["child_full_historical_evidence_replay_count"] == 0

    crossed = [dict(contract.quality_coordinates()[0])]
    crossed.append(
        next(
            dict(item)
            for item in contract.quality_coordinates()
            if item["training_seed"] != crossed[0]["training_seed"]
        )
    )
    with pytest.raises(ValueError, match="maximal remaining cohort prefix"):
        _plan(tmp_path, trust_root, coordinates=crossed)

    reversed_coordinates = [dict(item) for item in reversed(contract.quality_coordinates()[:2])]
    with pytest.raises(ValueError, match="maximal remaining cohort prefix"):
        _plan(tmp_path, trust_root, coordinates=reversed_coordinates)


def test_work_result_and_receipt_are_hmac_bound_and_monotone(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    plan = _plan(tmp_path, trust_root)
    work, results = _result_sequence(tmp_path, trust_root, plan, count=2)
    receipt = session.build_session_receipt(plan, work, results, trust_root=trust_root)

    assert (
        session.validate_work_order(
            work[1], plan=plan, expected_sequence_index=1, trust_root=trust_root
        )
        == work[1]
    )
    assert (
        session.validate_work_result(
            results[1], plan=plan, work_order=work[1], trust_root=trust_root
        )
        == results[1]
    )
    assert (
        session.validate_session_receipt(
            receipt,
            plan=plan,
            work_orders=work,
            results=results,
            trust_root=trust_root,
        )
        == receipt
    )
    assert receipt["status"] == "stopped"
    assert receipt["model_load_count"] == 1
    assert receipt["model_state_reset_count"] == 2

    tampered = dict(work[0])
    tampered["launch_nonce"] = "f" * 64
    with pytest.raises(ValueError, match="digest drifted"):
        session.validate_work_order(
            tampered,
            plan=plan,
            expected_sequence_index=0,
            trust_root=trust_root,
        )


def test_published_target_remains_validatable_but_cannot_be_reissued(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    plan = _plan(tmp_path, trust_root, coordinates=[dict(contract.quality_coordinates()[0])])
    envelope = tmp_path / "published.json"
    work = session.build_work_order(
        plan,
        sequence_index=0,
        launch_nonce="7" * 64,
        envelope_path=envelope,
        projected_remaining_shards=9_000,
        projected_remaining_token_rows=contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
        trust_root=trust_root,
    )
    envelope.write_text("published\n", encoding="utf-8")
    assert (
        session.validate_work_order(
            work, plan=plan, expected_sequence_index=0, trust_root=trust_root
        )
        == work
    )
    result = session.build_work_result(
        plan,
        work,
        terminal_decision="INTEGRITY-PASS",
        envelope_binding={
            "path": str(envelope),
            "bytes": envelope.stat().st_size,
            "sha256": "8" * 64,
            "payload_sha256": "9" * 64,
            "attestation_mac": "a" * 64,
        },
        completed_in_session=1,
        model_state_reset_count=1,
        trust_root=trust_root,
    )
    receipt = session.build_session_receipt(plan, [work], [result], trust_root=trust_root)
    assert (
        session.validate_work_result(result, plan=plan, work_order=work, trust_root=trust_root)
        == result
    )
    assert (
        session.validate_session_receipt(
            receipt,
            plan=plan,
            work_orders=[work],
            results=[result],
            trust_root=trust_root,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="unexpectedly pre-existing"):
        session.build_work_order(
            plan,
            sequence_index=0,
            launch_nonce="b" * 64,
            envelope_path=envelope,
            projected_remaining_shards=9_000,
            projected_remaining_token_rows=(contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES),
            trust_root=trust_root,
        )


def test_sealed_plan_transport_rejects_mutable_descriptor(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    plan = _plan(tmp_path, trust_root)
    descriptor = session.create_sealed_plan_fd(plan)
    try:
        assert session.read_sealed_plan_fd(descriptor, trust_root=trust_root) == plan
    finally:
        os.close(descriptor)

    path = tmp_path / "mutable-plan.json"
    path.write_bytes(attestation.canonical_json(plan))
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="not an exact sealed"):
            session.read_sealed_plan_fd(descriptor, trust_root=trust_root)
    finally:
        os.close(descriptor)


def test_normal_model_load_bound_is_unique_worker_scale_seed_assignments() -> None:
    coordinates = list(session.assigned_coordinates(worker_index=0, worker_count=1))
    assert (
        session.normal_model_load_upper_bound(coordinates, worker_index=0, worker_count=1)
        == len(contract.SCALES) * len(contract.TRAINING_SEEDS)
        == 10
    )

    assigned = session.assigned_coordinates(worker_index=2, worker_count=7)
    assert session.normal_model_load_upper_bound(assigned, worker_index=2, worker_count=7) == 10


def test_efficiency_counter_counts_sessions_not_shards(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    assigned = list(session.assigned_coordinates(worker_index=0, worker_count=1))
    first_coordinates = assigned[:3]
    second_start = next(
        index
        for index, item in enumerate(assigned)
        if item["training_seed"] != first_coordinates[0]["training_seed"]
    )
    second_coordinates = assigned[second_start : second_start + 3]
    first = _plan(tmp_path, trust_root, coordinates=first_coordinates, session_digit="1")
    second = _plan(tmp_path, trust_root, coordinates=second_coordinates, session_digit="2")
    first_work, first_results = _result_sequence(tmp_path, trust_root, first, count=3)
    second_work, second_results = _result_sequence(tmp_path, trust_root, second, count=3)
    first_receipt = session.build_session_receipt(
        first, first_work, first_results, trust_root=trust_root
    )
    second_receipt = session.build_session_receipt(
        second, second_work, second_results, trust_root=trust_root
    )

    counters = session.efficiency_counters(
        [first, second], [first_receipt, second_receipt], trust_root=trust_root
    )
    assert counters == {
        "persistent_session_count": 2,
        "unique_worker_scale_seed_assignments": 2,
        "model_load_count": 2,
        "normal_model_load_upper_bound": 2,
        "normal_model_load_bound_satisfied": True,
        "child_full_historical_evidence_replay_count": 0,
    }


def test_durable_ledger_launch_terminal_and_exact_root_binding(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = tmp_path / "quality"
    plan = _plan(output_root, trust_root, session_digit="7")
    work, results = _result_sequence(output_root, trust_root, plan, count=3)
    ready = session.build_session_receipt(plan, (), (), trust_root=trust_root)
    final = session.build_session_receipt(plan, work, results, trust_root=trust_root)

    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )
    launch_only = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert launch_only["launch_attempt_count"] == 1
    assert launch_only["launch_only_interrupted_attempt_count"] == 1
    assert launch_only["sessions"][0]["actual_session_argv"] == _session_argv(plan)

    session.publish_session_terminal(
        output_root,
        plan,
        status="complete",
        ready_receipt=ready,
        final_receipt=final,
        work_orders=work,
        results=results,
        published_bundle_reingestion_count=0,
        actual_session_argv=_session_argv(plan),
        child_process_returncode=0,
        trust_root=trust_root,
    )
    terminal = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert terminal["terminal_count"] == 1
    assert terminal["observed_successful_model_loads"] == 1
    assert terminal["durably_evidenced_parent_full_evidence_replays"] == 1
    assert terminal["sessions"][0]["actual_session_argv"] == _session_argv(plan)

    other_root = tmp_path / "other-quality"
    with pytest.raises(ValueError, match="plan/output-root cross-binding"):
        session.publish_session_launch(
            other_root,
            plan,
            actual_session_argv=_session_argv(plan),
            trust_root=trust_root,
        )

    other_ledger = session.session_ledger_root(other_root)
    other_ledger.mkdir(mode=0o700)
    other_lock = session.session_ledger_lock_path(other_root)
    other_lock.touch(mode=0o600)
    other_lock.chmod(0o600)
    for source in session.session_ledger_root(output_root).iterdir():
        target = other_ledger / source.name
        shutil.copyfile(source, target)
        target.chmod(0o600)
    with pytest.raises(ValueError, match="plan/output-root cross-binding"):
        session.load_session_ledger_projection(other_root, trust_root=trust_root)


def test_ledger_loader_is_nonblocking_bounded_and_recovers_exact_hardlink_tmp(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    fifo_root = tmp_path / "fifo-quality"
    fifo_ledger = session.session_ledger_root(fifo_root)
    fifo_ledger.mkdir(mode=0o700)
    fifo_lock = session.session_ledger_lock_path(fifo_root)
    fifo_lock.touch(mode=0o600)
    fifo_lock.chmod(0o600)
    fifo = fifo_ledger / f"{'1' * 64}.launch.json"
    os.mkfifo(fifo, mode=0o600)
    with pytest.raises(ValueError, match="metadata is unsafe"):
        session.load_session_ledger_projection(fifo_root, trust_root=trust_root)

    huge_root = tmp_path / "huge-quality"
    huge_ledger = session.session_ledger_root(huge_root)
    huge_ledger.mkdir(mode=0o700)
    huge_lock = session.session_ledger_lock_path(huge_root)
    huge_lock.touch(mode=0o600)
    huge_lock.chmod(0o600)
    huge = huge_ledger / f"{'2' * 64}.launch.json"
    with huge.open("wb") as stream:
        stream.truncate(session.MAXIMUM_LEDGER_BYTES + 1)
    huge.chmod(0o600)
    with pytest.raises(ValueError, match="byte size is unsafe"):
        session.load_session_ledger_projection(huge_root, trust_root=trust_root)

    recovered_root = tmp_path / "recovered-quality"
    plan = _plan(recovered_root, trust_root, session_digit="8")
    binding = session.publish_session_launch(
        recovered_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )
    final = Path(str(binding["path"]))
    interrupted_tmp = final.parent / f".{final.name}.abcdefgh.tmp"
    os.link(final, interrupted_tmp)
    assert final.stat().st_nlink == 2
    projection = session.load_session_ledger_projection(recovered_root, trust_root=trust_root)
    assert projection["launch_attempt_count"] == 1
    assert not interrupted_tmp.exists()
    assert final.stat().st_nlink == 1


def test_session_ledger_absence_lock_only_recovery_and_missing_lock_fail_closed(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = tmp_path / "quality"
    ledger_root = session.session_ledger_root(output_root)
    lock_path = session.session_ledger_lock_path(output_root)

    empty = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert empty["launch_attempt_count"] == 0
    assert not ledger_root.exists()
    assert not lock_path.exists()

    lock_path.touch(mode=0o600)
    lock_path.chmod(0o600)
    lock_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    lock_only = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert lock_only["launch_attempt_count"] == 0
    assert not ledger_root.exists()
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == lock_identity

    plan = _plan(output_root, trust_root, session_digit="9")
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )
    assert ledger_root.is_dir()
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == lock_identity

    broken_output = tmp_path / "broken-quality"
    broken_root = session.session_ledger_root(broken_output)
    broken_root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="lock is missing"):
        session.load_session_ledger_projection(broken_output, trust_root=trust_root)
    broken_plan = _plan(broken_output, trust_root, session_digit="a")
    with pytest.raises(ValueError, match="lock is missing"):
        session.publish_session_launch(
            broken_output,
            broken_plan,
            actual_session_argv=_session_argv(broken_plan),
            trust_root=trust_root,
        )


def test_parent_crash_recovery_closes_authenticated_committed_prefix(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    plan = _plan(output_root, trust_root, session_digit="9")
    work, results = _result_sequence(output_root, trust_root, plan, count=2)
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )

    projection = session.reconcile_committed_launch_only_sessions(
        output_root,
        [
            _recovery_record(plan, work_order, work_result)
            for work_order, work_result in zip(work, results, strict=True)
        ],
        worker_index=0,
        gpu_lease_binding_digest=str(plan["gpu_lease_binding_digest"]),
        terminal_authority_check=lambda _plan, _argv: None,
        trust_root=trust_root,
    )

    row = projection["sessions"][0]
    assert row["status"] == "parent_crash_recovered"
    assert row["completed_work_payload_sha256"] == [item["payload_sha256"] for item in work]
    assert row["completed_result_payload_sha256"] == [item["payload_sha256"] for item in results]
    assert row["actual_session_argv"] == _session_argv(plan)
    assert row["child_process_returncode"] is None
    assert projection["launch_only_interrupted_attempt_count"] == 0
    assert projection["terminal_count"] == 1
    assert projection["normal_path_model_load_bound_applicable"] is False
    assert projection["normal_path_model_load_bound_observed_satisfied"] is None


def test_parent_crash_recovery_authority_failure_leaves_launch_only(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    plan = _plan(output_root, trust_root, session_digit="a")
    work, results = _result_sequence(output_root, trust_root, plan, count=1)
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )

    def reject_terminal_authority(_plan: Mapping[str, object], _argv: Sequence[str]) -> None:
        raise RuntimeError("terminal-authority-drifted")

    with pytest.raises(RuntimeError, match="terminal-authority-drifted"):
        session.reconcile_committed_launch_only_sessions(
            output_root,
            [_recovery_record(plan, work[0], results[0])],
            worker_index=0,
            gpu_lease_binding_digest=str(plan["gpu_lease_binding_digest"]),
            terminal_authority_check=reject_terminal_authority,
            trust_root=trust_root,
        )

    projection = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert projection["terminal_count"] == 0
    assert projection["sessions"][0]["status"] == "launch_only"


def test_parent_crash_recovery_rejects_wrong_reacquired_gpu_binding(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    plan = _plan(output_root, trust_root, session_digit="a")
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )

    with pytest.raises(ValueError, match="reacquired worker GPU lease"):
        session.reconcile_committed_launch_only_sessions(
            output_root,
            [],
            worker_index=0,
            gpu_lease_binding_digest="f" * 64,
            terminal_authority_check=lambda _plan, _argv: None,
            trust_root=trust_root,
        )
    projection = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert projection["sessions"][0]["status"] == "launch_only"


@pytest.mark.parametrize("failure", ["gap", "duplicate", "forged-result"])
def test_parent_crash_recovery_rejects_noncanonical_or_forged_records(
    tmp_path: Path,
    trust_root: attestation.TrustRoot,
    failure: str,
) -> None:
    output_root = (tmp_path / failure).resolve()
    digit = {"gap": "b", "duplicate": "c", "forged-result": "d"}[failure]
    plan = _plan(output_root, trust_root, session_digit=digit)
    work, results = _result_sequence(output_root, trust_root, plan, count=2)
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )
    if failure == "gap":
        records = [_recovery_record(plan, work[1], results[1])]
    elif failure == "duplicate":
        first = _recovery_record(plan, work[0], results[0])
        records = [first, first]
    else:
        forged = dict(results[0])
        forged["terminal_decision"] = "INTEGRITY-FAIL"
        records = [_recovery_record(plan, work[0], forged)]

    with pytest.raises(ValueError):
        session.reconcile_committed_launch_only_sessions(
            output_root,
            records,
            worker_index=0,
            gpu_lease_binding_digest=str(plan["gpu_lease_binding_digest"]),
            terminal_authority_check=lambda _plan, _argv: None,
            trust_root=trust_root,
        )
    projection = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    assert projection["sessions"][0]["status"] == "launch_only"


def test_parent_crash_recovery_conservatively_closes_zero_record_launch(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    plan = _plan(output_root, trust_root, session_digit="e")
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )

    projection = session.reconcile_committed_launch_only_sessions(
        output_root,
        [],
        worker_index=0,
        gpu_lease_binding_digest=str(plan["gpu_lease_binding_digest"]),
        terminal_authority_check=lambda _plan, _argv: None,
        trust_root=trust_root,
    )

    row = projection["sessions"][0]
    assert row["status"] == "parent_crash_recovered"
    assert row["completed_work_payload_sha256"] == []
    assert row["completed_result_payload_sha256"] == []
    assert row["published_bundle_reingestion_count"] == 0
    assert projection["ready_model_load_count"] == 0
    assert projection["normal_path_model_load_bound_applicable"] is False


def test_parent_crash_recovery_closes_final_eof_reingestion_after_prefix(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    plan = _plan(output_root, trust_root, session_digit="f")
    work, results = _result_sequence(output_root, trust_root, plan, count=2)
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )

    records = [
        _recovery_record(plan, work[0], results[0]),
        _recovery_record(plan, work[1], None),
    ]
    projection = session.reconcile_committed_launch_only_sessions(
        output_root,
        records,
        worker_index=0,
        gpu_lease_binding_digest=str(plan["gpu_lease_binding_digest"]),
        terminal_authority_check=lambda _plan, _argv: None,
        trust_root=trust_root,
    )

    row = projection["sessions"][0]
    assert row["status"] == "parent_crash_recovered"
    assert row["published_bundle_reingestion_count"] == 1
    assert row["completed_work_payload_sha256"] == [work[0]["payload_sha256"]]
    assert row["completed_result_payload_sha256"] == [results[0]["payload_sha256"]]


@pytest.mark.parametrize("layout", ["nonfinal", "two-eof"])
def test_parent_crash_recovery_rejects_nonfinal_or_repeated_eof(
    tmp_path: Path,
    trust_root: attestation.TrustRoot,
    layout: str,
) -> None:
    output_root = (tmp_path / layout).resolve()
    plan = _plan(output_root, trust_root, session_digit="0")
    work, results = _result_sequence(output_root, trust_root, plan, count=2)
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )
    records = (
        [
            _recovery_record(plan, work[0], None),
            _recovery_record(plan, work[1], results[1]),
        ]
        if layout == "nonfinal"
        else [
            _recovery_record(plan, work[0], None),
            _recovery_record(plan, work[1], None),
        ]
    )

    with pytest.raises(ValueError, match="EOF record"):
        session.reconcile_committed_launch_only_sessions(
            output_root,
            records,
            worker_index=0,
            gpu_lease_binding_digest=str(plan["gpu_lease_binding_digest"]),
            terminal_authority_check=lambda _plan, _argv: None,
            trust_root=trust_root,
        )


def _publish_ready_only_success(
    output_root: Path,
    trust_root: attestation.TrustRoot,
    *,
    session_digit: str,
    authority_digit: str = "a",
) -> dict[str, object]:
    plan = _plan(
        output_root,
        trust_root,
        coordinates=[dict(contract.quality_coordinates()[0])],
        session_digit=session_digit,
        authority_digit=authority_digit,
        session_role=session.READY_ONLY_PREFLIGHT_SESSION_ROLE,
    )
    ready = session.build_session_receipt(plan, (), (), trust_root=trust_root)
    session.publish_session_launch(
        output_root,
        plan,
        actual_session_argv=_session_argv(plan),
        trust_root=trust_root,
    )
    session.publish_session_terminal(
        output_root,
        plan,
        status="stopped",
        ready_receipt=ready,
        final_receipt=ready,
        work_orders=(),
        results=(),
        published_bundle_reingestion_count=0,
        actual_session_argv=_session_argv(plan),
        child_process_returncode=0,
        trust_root=trust_root,
    )
    return plan


def test_ready_only_role_is_plan_authenticated_and_cannot_substitute_quality(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    plan = _plan(
        tmp_path.resolve(),
        trust_root,
        coordinates=[dict(contract.quality_coordinates()[0])],
        session_role=session.READY_ONLY_PREFLIGHT_SESSION_ROLE,
    )
    assert plan["session_role"] == session.READY_ONLY_PREFLIGHT_SESSION_ROLE
    assert session.validate_session_plan(plan, trust_root=trust_root) == plan

    substituted = dict(plan)
    substituted["session_role"] = session.QUALITY_SESSION_ROLE
    with pytest.raises(ValueError, match="digest drifted"):
        session.validate_session_plan(substituted, trust_root=trust_root)


def test_ready_only_binding_is_exact_zero_work_and_metrics_exclude_quality(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    preflight = _publish_ready_only_success(output_root, trust_root, session_digit="1")
    assigned = session.assigned_coordinates(worker_index=0, worker_count=1)
    second_cohort_start = next(
        index
        for index, item in enumerate(assigned)
        if (item["scale"], item["training_seed"])
        != (preflight["scale"], preflight["training_seed"])
    )
    quality = _plan(
        output_root,
        trust_root,
        coordinates=[dict(assigned[second_cohort_start])],
        session_digit="2",
        authority_digit="b",
    )
    session.publish_session_launch(
        output_root,
        quality,
        actual_session_argv=_session_argv(quality),
        trust_root=trust_root,
    )

    projection = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    binding = session.ready_only_preflight_binding(projection)
    assert binding is not None
    assert binding["status"] == "stopped"
    assert binding["ready_receipt_binding"] == binding["final_receipt_binding"]
    assert binding["quality_work_order_count"] == 0
    assert binding["quality_result_count"] == 0
    assert binding["model_state_reset_count"] == 0
    assert binding["launch_ledger_binding"]["sha256"] in {
        row["sha256"] for row in projection["registry"]
    }
    assert binding["terminal_ledger_binding"]["sha256"] in {
        row["sha256"] for row in projection["registry"]
    }
    assert projection["ready_only_preflight_launch_attempt_count"] == 1
    assert projection["ready_only_preflight_ready_model_load_count"] == 1
    assert projection["quality_launch_attempt_count"] == 1
    assert projection["quality_ready_model_load_count"] == 0
    assert projection["normal_path_model_load_bound"] == 1
    assert projection["single_worker_normal_path_ready_only_preflight_model_load_upper_bound"] == 1
    assert projection["single_worker_normal_path_quality_model_load_upper_bound"] == 10
    assert projection["single_worker_normal_path_total_model_load_upper_bound"] == 11


def test_ready_only_launch_only_is_recovered_then_one_success_is_adoptable(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    interrupted = _plan(
        output_root,
        trust_root,
        coordinates=[dict(contract.quality_coordinates()[0])],
        session_digit="3",
        session_role=session.READY_ONLY_PREFLIGHT_SESSION_ROLE,
    )
    session.publish_session_launch(
        output_root,
        interrupted,
        actual_session_argv=_session_argv(interrupted),
        trust_root=trust_root,
    )
    recovered = session.reconcile_committed_launch_only_sessions(
        output_root,
        (),
        worker_index=0,
        gpu_lease_binding_digest=str(interrupted["gpu_lease_binding_digest"]),
        terminal_authority_check=lambda _plan, _argv: None,
        trust_root=trust_root,
    )
    assert session.ready_only_preflight_binding(recovered) is None
    assert recovered["sessions"][0]["status"] == "parent_crash_recovered"

    _publish_ready_only_success(output_root, trust_root, session_digit="4")
    adopted = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    binding = session.ready_only_preflight_binding(adopted)
    assert binding is not None
    assert binding["launch_attempt_count"] == 2
    assert binding["failed_attempt_count"] == 1
    assert adopted["ready_only_preflight_success_count"] == 1


def test_ready_only_failed_load_then_success_exceeds_only_the_normal_path_count(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    output_root = (tmp_path / "quality").resolve()
    failed = _plan(
        output_root,
        trust_root,
        coordinates=[dict(contract.quality_coordinates()[0])],
        session_digit="8",
        session_role=session.READY_ONLY_PREFLIGHT_SESSION_ROLE,
    )
    failed_ready = session.build_session_receipt(failed, (), (), trust_root=trust_root)
    session.publish_session_launch(
        output_root,
        failed,
        actual_session_argv=_session_argv(failed),
        trust_root=trust_root,
    )
    session.publish_session_terminal(
        output_root,
        failed,
        status="child_eof",
        ready_receipt=failed_ready,
        final_receipt=None,
        work_orders=(),
        results=(),
        published_bundle_reingestion_count=0,
        actual_session_argv=_session_argv(failed),
        child_process_returncode=7,
        trust_root=trust_root,
    )
    _publish_ready_only_success(output_root, trust_root, session_digit="9")

    projection = session.load_session_ledger_projection(output_root, trust_root=trust_root)
    binding = session.ready_only_preflight_binding(projection)
    assert binding is not None
    assert binding["launch_attempt_count"] == 2
    assert binding["failed_attempt_count"] == 1
    assert projection["checkpoint_model_load_attempt_upper_bound"] == 2
    assert projection["ready_only_preflight_ready_model_load_count"] == 2
    assert projection["single_worker_normal_path_ready_only_preflight_model_load_upper_bound"] == 1
    assert projection["single_worker_normal_path_total_model_load_upper_bound"] == 11


def test_ready_only_second_success_or_tampered_terminal_fails_closed(
    tmp_path: Path, trust_root: attestation.TrustRoot
) -> None:
    duplicate_root = (tmp_path / "duplicate").resolve()
    _publish_ready_only_success(duplicate_root, trust_root, session_digit="5")
    _publish_ready_only_success(duplicate_root, trust_root, session_digit="6")
    duplicate = session.load_session_ledger_projection(duplicate_root, trust_root=trust_root)
    with pytest.raises(ValueError, match="more than one successful"):
        session.ready_only_preflight_binding(duplicate)

    tampered_root = (tmp_path / "tampered").resolve()
    plan = _publish_ready_only_success(tampered_root, trust_root, session_digit="7")
    terminal_path = (
        session.session_ledger_root(tampered_root) / f"{plan['session_nonce']}.terminal.json"
    )
    raw = bytearray(terminal_path.read_bytes())
    raw[raw.index(b'"status": "stopped"')] = ord("X")
    terminal_path.write_bytes(raw)
    terminal_path.chmod(0o600)
    with pytest.raises(ValueError):
        session.load_session_ledger_projection(tampered_root, trust_root=trust_root)


def test_clean_cohort_boundary_restart_disables_single_parent_normal_claim() -> None:
    additional, applicable = session._normal_path_claim_semantics(
        launch_count=2,
        terminal_count=2,
        graceful_terminal_count=2,
        controlled_stop_count=0,
        cohort_count=2,
        launch_authority_count=2,
        worker_counts=[1, 1],
    )
    assert additional == 1
    assert applicable is False

    _additional, stopped_applicable = session._normal_path_claim_semantics(
        launch_count=1,
        terminal_count=1,
        graceful_terminal_count=0,
        controlled_stop_count=0,
        cohort_count=1,
        launch_authority_count=1,
        worker_counts=[1],
    )
    assert stopped_applicable is False


def test_parallel_normal_path_allows_exactly_one_authority_per_worker() -> None:
    additional, applicable = session._normal_path_claim_semantics(
        launch_count=30,
        terminal_count=30,
        graceful_terminal_count=30,
        controlled_stop_count=0,
        cohort_count=30,
        launch_authority_count=3,
        worker_counts=[3] * 30,
    )
    assert additional == 0
    assert applicable is True

    extra, retry_applicable = session._normal_path_claim_semantics(
        launch_count=30,
        terminal_count=30,
        graceful_terminal_count=30,
        controlled_stop_count=0,
        cohort_count=30,
        launch_authority_count=4,
        worker_counts=[3] * 30,
    )
    assert extra == 1
    assert retry_applicable is False
