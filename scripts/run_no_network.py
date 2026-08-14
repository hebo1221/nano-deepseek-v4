"""Run a command in a verified, unprivileged Linux network namespace."""

from __future__ import annotations

import fcntl
import os
import socket
import stat
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PARENT_NETNS_ENV = "NANO_DEEPSEEK_V4_PARENT_NETNS"
ISOLATION_ENV = "NANO_DEEPSEEK_V4_OS_NETWORK_ISOLATION"
ISOLATION_VALUE = "linux_network_namespace"
EXPECTED_UID_ENV = "NANO_DEEPSEEK_V4_EXPECTED_UID"
EXPECTED_GID_ENV = "NANO_DEEPSEEK_V4_EXPECTED_GID"

_TRUSTED_BINARY_PATHS: dict[str, tuple[Path, ...]] = {
    "sudo": (Path("/usr/bin/sudo"),),
    "unshare": (Path("/usr/bin/unshare"),),
    "sh": (Path("/bin/sh"), Path("/usr/bin/sh")),
    "ip": (Path("/usr/sbin/ip"), Path("/usr/bin/ip")),
    "setpriv": (Path("/usr/bin/setpriv"),),
    "env": (Path("/usr/bin/env"),),
}


class NetworkIsolationError(RuntimeError):
    """Raised when the requested no-network execution boundary is not active."""


def _current_network_state() -> tuple[set[str], int]:
    """Read interfaces and loopback flags from the current network namespace."""

    try:
        interfaces = {name for _, name in socket.if_nameindex()}
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            request = struct.pack("256s", b"lo")
            response = fcntl.ioctl(probe.fileno(), 0x8913, request)
        loopback_flags = struct.unpack("H", response[16:18])[0]
    except (OSError, struct.error) as exc:
        raise NetworkIsolationError(
            "cannot inspect isolated network interfaces"
        ) from exc
    return interfaces, loopback_flags


def _status_integer(status_text: str, name: str, *, base: int) -> int:
    for line in status_text.splitlines():
        if line.startswith(f"{name}:"):
            try:
                return int(line.split(":", 1)[1].strip(), base)
            except ValueError as exc:
                raise NetworkIsolationError(f"invalid {name} value") from exc
    raise NetworkIsolationError(f"{name} is missing from /proc/self/status")


def validate_isolation(
    *,
    parent_netns: str,
    current_netns: str,
    interfaces: set[str],
    loopback_flags: int,
    status_text: str,
    expected_uid: int,
    expected_gid: int,
    current_uid: int,
    current_gid: int,
    supplementary_groups: set[int],
) -> None:
    """Validate the namespace, interface inventory, and dropped privileges."""

    if not parent_netns or current_netns == parent_netns:
        raise NetworkIsolationError("network namespace did not change")
    if interfaces != {"lo"}:
        raise NetworkIsolationError(
            f"isolated namespace interface drift: {sorted(interfaces)}"
        )
    if loopback_flags & 0x1 == 0:
        raise NetworkIsolationError("isolated namespace loopback is not up")
    if expected_uid == 0 or expected_gid == 0:
        raise NetworkIsolationError("isolated command must use a non-root identity")
    if current_uid != expected_uid or current_gid != expected_gid:
        raise NetworkIsolationError("isolated command identity drift")
    if supplementary_groups:
        raise NetworkIsolationError("supplementary groups were not cleared")
    for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if _status_integer(status_text, name, base=16) != 0:
            raise NetworkIsolationError(f"{name} capabilities were not dropped")
    if _status_integer(status_text, "NoNewPrivs", base=10) != 1:
        raise NetworkIsolationError("no_new_privs is not active")


def _required_binary(name: str) -> str:
    candidates = _TRUSTED_BINARY_PATHS.get(name)
    if candidates is None:
        raise NetworkIsolationError(f"no trusted path is configured for executable: {name}")
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
            raise NetworkIsolationError(
                f"trusted executable is not an executable file: {candidate}"
            )
        if metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise NetworkIsolationError(
                f"trusted executable must be root-owned and not group/other-writable: {candidate}"
            )
        return str(resolved)
    raise NetworkIsolationError(f"required trusted executable is unavailable: {name}")


def _outer_argv(
    command: Sequence[str],
    *,
    environ: Mapping[str, str],
    uid: int,
    gid: int,
    parent_netns: str,
    executable: str,
    script: Path,
) -> list[str]:
    if not command:
        raise NetworkIsolationError("a command is required")
    home = environ.get("HOME")
    path = environ.get("PATH")
    if not home or not path:
        raise NetworkIsolationError("HOME and PATH are required")
    return [
        _required_binary("sudo"),
        "-n",
        _required_binary("unshare"),
        "--net",
        "--",
        _required_binary("sh"),
        "-c",
        '"$1" link set dev lo up && shift && exec "$@"',
        "run-no-network",
        _required_binary("ip"),
        _required_binary("setpriv"),
        f"--reuid={uid}",
        f"--regid={gid}",
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        _required_binary("env"),
        "-i",
        f"HOME={home}",
        f"PATH={path}",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "PYTHONNOUSERSITE=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_DISABLE_TELEMETRY=1",
        f"{PARENT_NETNS_ENV}={parent_netns}",
        f"{EXPECTED_UID_ENV}={uid}",
        f"{EXPECTED_GID_ENV}={gid}",
        executable,
        str(script),
        "--inside",
        *command,
    ]


def _run_inside(command: Sequence[str]) -> None:
    if not command:
        raise NetworkIsolationError("a command is required")
    parent_netns = os.environ.get(PARENT_NETNS_ENV, "")
    try:
        expected_uid = int(os.environ.get(EXPECTED_UID_ENV, ""))
        expected_gid = int(os.environ.get(EXPECTED_GID_ENV, ""))
    except ValueError as exc:
        raise NetworkIsolationError("expected uid/gid markers are invalid") from exc
    _validate_current_isolation(parent_netns, expected_uid, expected_gid)
    os.environ[ISOLATION_ENV] = ISOLATION_VALUE
    os.execvpe(command[0], list(command), os.environ)


def _validate_current_isolation(
    parent_netns: str,
    expected_uid: int,
    expected_gid: int,
) -> None:
    interfaces, loopback_flags = _current_network_state()
    validate_isolation(
        parent_netns=parent_netns,
        current_netns=os.readlink("/proc/self/ns/net"),
        interfaces=interfaces,
        loopback_flags=loopback_flags,
        status_text=Path("/proc/self/status").read_text(encoding="utf-8"),
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        current_uid=os.geteuid(),
        current_gid=os.getegid(),
        supplementary_groups=set(os.getgroups()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments[:1] == ["--inside"]:
            _run_inside(arguments[1:])
        else:
            command = _outer_argv(
                arguments,
                environ=os.environ,
                uid=os.getuid(),
                gid=os.getgid(),
                parent_netns=os.readlink("/proc/self/ns/net"),
                executable=sys.executable,
                script=Path(__file__).resolve(),
            )
            os.execv(command[0], command)
    except (NetworkIsolationError, OSError) as exc:
        print(f"network-isolated execution failed: {exc}", file=sys.stderr)
        return 4
    raise AssertionError("exec unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
