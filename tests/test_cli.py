from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence

import pytest

from nano_deepseek_v4 import __version__, cli


def test_root_cli_without_arguments_prints_categorized_help(capsys):
    assert cli.main([]) == 0

    output = capsys.readouterr().out
    assert "usage: nano-deepseek-v4" in output
    assert "First run:" in output
    assert "Build and use bundles:" in output
    assert "Inspect and verify:" in output
    assert "Research receipts:" in output
    for command in cli.COMMANDS:
        assert command.name in output


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_root_cli_help_is_successful(flag: str, capsys):
    assert cli.main([flag]) == 0
    assert "nano-deepseek-v4 <command> --help" in capsys.readouterr().out


def test_root_cli_reports_the_package_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"nano-deepseek-v4 {__version__}\n"


@pytest.mark.parametrize(
    "arguments",
    [
        ["missing"],
        ["--unknown"],
        ["--version", "demo"],
        ["--help", "demo"],
    ],
)
def test_root_cli_rejects_invalid_root_arguments_without_traceback(
    arguments: list[str], capsys
):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments)

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("command", cli.COMMANDS, ids=lambda command: command.name)
def test_root_cli_lazily_forwards_arguments_and_return_code(command, monkeypatch):
    calls: list[tuple[str, list[str], str]] = []
    original_program = sys.argv[0]

    def fake_load(module_name: str):
        def fake_main(arguments: Sequence[str] | None = None) -> int:
            calls.append((module_name, list(arguments or []), sys.argv[0]))
            return 17

        return fake_main

    monkeypatch.setattr(cli, "_load_command_main", fake_load)

    assert cli.main([command.name, "--example", "value"]) == 17
    assert calls == [
        (command.module, ["--example", "value"], f"nano-deepseek-v4 {command.name}")
    ]
    assert sys.argv[0] == original_program


@pytest.mark.parametrize("command", cli.COMMANDS, ids=lambda command: command.name)
def test_root_cli_subcommand_help_uses_copyable_program_name(command, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([command.name, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert f"usage: nano-deepseek-v4 {command.name}" in output


def test_root_cli_demo_preserves_the_existing_smoke_test(capsys):
    assert cli.main(["demo"]) == 0
    output = capsys.readouterr().out
    assert "cached/full match: True" in output


def test_python_m_package_entry_point(monkeypatch):
    calls: list[Sequence[str] | None] = []

    def fake_main(arguments: Sequence[str] | None = None) -> int:
        calls.append(arguments)
        return 0

    monkeypatch.setattr(cli, "main", fake_main)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("nano_deepseek_v4.__main__", run_name="__main__")

    assert exc_info.value.code == 0
    assert calls == [None]
