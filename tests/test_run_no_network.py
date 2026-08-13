from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.run_no_network as runner
from scripts.run_no_network import NetworkIsolationError, validate_isolation

ISOLATED_STATUS = """\
Name:\tpython
CapInh:\t0000000000000000
CapPrm:\t0000000000000000
CapEff:\t0000000000000000
CapBnd:\t0000000000000000
CapAmb:\t0000000000000000
NoNewPrivs:\t1
"""


def test_isolation_validation_accepts_only_loopback_without_capabilities():
    validate_isolation(
        parent_netns="net:[1]",
        current_netns="net:[2]",
        interfaces={"lo"},
        loopback_flags=0x9,
        status_text=ISOLATED_STATUS,
        expected_uid=1001,
        expected_gid=1002,
        current_uid=1001,
        current_gid=1002,
        supplementary_groups=set(),
    )


@pytest.mark.parametrize(
    ("current_netns", "interfaces", "loopback_flags", "status_text", "message"),
    [
        ("net:[1]", {"lo"}, 0x9, ISOLATED_STATUS, "did not change"),
        ("net:[2]", {"eth0", "lo"}, 0x9, ISOLATED_STATUS, "interface drift"),
        ("net:[2]", {"lo"}, 0x8, ISOLATED_STATUS, "loopback is not up"),
        (
            "net:[2]",
            {"lo"},
            0x9,
            ISOLATED_STATUS.replace(
                "CapEff:\t0000000000000000",
                "CapEff:\t0000000000000001",
            ),
            "CapEff capabilities were not dropped",
        ),
        (
            "net:[2]",
            {"lo"},
            0x9,
            ISOLATED_STATUS.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0"),
            "no_new_privs is not active",
        ),
    ],
)
def test_isolation_validation_fails_closed(
    current_netns: str,
    interfaces: set[str],
    loopback_flags: int,
    status_text: str,
    message: str,
):
    with pytest.raises(NetworkIsolationError, match=message):
        validate_isolation(
            parent_netns="net:[1]",
            current_netns=current_netns,
            interfaces=interfaces,
            loopback_flags=loopback_flags,
            status_text=status_text,
            expected_uid=1001,
            expected_gid=1002,
            current_uid=1001,
            current_gid=1002,
            supplementary_groups=set(),
        )


@pytest.mark.parametrize(
    ("expected_uid", "expected_gid", "current_uid", "current_gid", "groups", "message"),
    [
        (0, 1002, 0, 1002, set(), "non-root identity"),
        (1001, 1002, 1003, 1002, set(), "identity drift"),
        (1001, 1002, 1001, 1002, {27}, "groups were not cleared"),
    ],
)
def test_isolation_validation_rejects_privilege_drift(
    expected_uid: int,
    expected_gid: int,
    current_uid: int,
    current_gid: int,
    groups: set[int],
    message: str,
):
    with pytest.raises(NetworkIsolationError, match=message):
        validate_isolation(
            parent_netns="net:[1]",
            current_netns="net:[2]",
            interfaces={"lo"},
            loopback_flags=0x9,
            status_text=ISOLATED_STATUS,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            current_uid=current_uid,
            current_gid=current_gid,
            supplementary_groups=groups,
        )


def test_outer_command_uses_root_only_to_create_namespace_and_then_drops_privileges(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runner, "_required_binary", lambda name: f"/usr/bin/{name}")

    command = runner._outer_argv(
        ["/venv/bin/nano-deepseek-v4", "conformance", "--profile", "offline"],
        environ={"HOME": "/home/runner", "PATH": "/tools/bin:/usr/bin"},
        uid=1001,
        gid=1002,
        parent_netns="net:[1]",
        executable="/tools/bin/python",
        script=Path("/workspace/scripts/run_no_network.py"),
    )

    assert command[:6] == [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/unshare",
        "--net",
        "--",
        "/usr/bin/sh",
    ]
    assert command[6:11] == [
        "-c",
        '"$1" link set dev lo up && shift && exec "$@"',
        "run-no-network",
        "/usr/bin/ip",
        "/usr/bin/setpriv",
    ]
    assert "--reuid=1001" in command
    assert "--regid=1002" in command
    assert "--clear-groups" in command
    assert "--bounding-set=-all" in command
    assert "--no-new-privs" in command
    assert "/usr/bin/env" in command
    assert "-i" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert f"{runner.PARENT_NETNS_ENV}=net:[1]" in command
    assert f"{runner.EXPECTED_UID_ENV}=1001" in command
    assert f"{runner.EXPECTED_GID_ENV}=1002" in command
    assert command[-5:] == [
        "--inside",
        "/venv/bin/nano-deepseek-v4",
        "conformance",
        "--profile",
        "offline",
    ]


def test_privileged_command_chain_ignores_caller_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    for name in runner._TRUSTED_BINARY_PATHS:
        fake = tmp_path / name
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    selected = {
        name: Path(runner._required_binary(name))
        for name in runner._TRUSTED_BINARY_PATHS
    }

    assert all(path.is_absolute() for path in selected.values())
    assert all(tmp_path not in path.parents for path in selected.values())
    assert selected["sudo"] != tmp_path / "sudo"


def test_required_binary_rejects_untrusted_absolute_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fake = tmp_path / "sudo"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setitem(runner._TRUSTED_BINARY_PATHS, "sudo", (fake,))

    with pytest.raises(NetworkIsolationError, match="root-owned"):
        runner._required_binary("sudo")


def test_inside_revalidates_before_marking_and_execing(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[object] = []

    def validate(parent: str, uid: int, gid: int) -> None:
        events.append(("validate", parent, uid, gid))

    def execute(file: str, argv: list[str], environ: dict[str, str]) -> None:
        events.append(("exec", file, argv, environ[runner.ISOLATION_ENV]))
        raise OSError("stop after observation")

    monkeypatch.setenv(runner.PARENT_NETNS_ENV, "net:[1]")
    monkeypatch.setenv(runner.EXPECTED_UID_ENV, "1001")
    monkeypatch.setenv(runner.EXPECTED_GID_ENV, "1002")
    monkeypatch.delenv(runner.ISOLATION_ENV, raising=False)
    monkeypatch.setattr(runner, "_validate_current_isolation", validate)
    monkeypatch.setattr(os, "execvpe", execute)

    with pytest.raises(OSError, match="stop after observation"):
        runner._run_inside(["tool", "arg"])

    assert events == [
        ("validate", "net:[1]", 1001, 1002),
        ("exec", "tool", ["tool", "arg"], runner.ISOLATION_VALUE),
    ]
