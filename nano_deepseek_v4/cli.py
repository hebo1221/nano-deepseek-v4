from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from . import __version__

CommandMain = Callable[[Sequence[str] | None], int]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    module: str
    description: str
    category: str


COMMANDS = (
    CommandSpec(
        "demo",
        "nano_deepseek_v4.demo",
        "run the no-download tiny CPU and cache smoke test",
        "First run",
    ),
    CommandSpec(
        "reproduce",
        "nano_deepseek_v4.reproduce",
        "run the fixed tiny CPU learning and bundle contract",
        "First run",
    ),
    CommandSpec(
        "train",
        "nano_deepseek_v4.train_text",
        "train a tiny byte-text model and optionally save a native bundle",
        "Build and use bundles",
    ),
    CommandSpec(
        "generate",
        "nano_deepseek_v4.generate_text",
        "generate text from a verified tokenizer-bound bundle",
        "Build and use bundles",
    ),
    CommandSpec(
        "inspect",
        "nano_deepseek_v4.architecture",
        "inspect architectures, checkpoints, or bundles without model allocation",
        "Inspect and verify",
    ),
    CommandSpec(
        "conformance",
        "nano_deepseek_v4.conformance",
        "run a fixed evidence-carrying conformance profile",
        "Inspect and verify",
    ),
    CommandSpec(
        "dspark",
        "nano_deepseek_v4.dspark",
        "check the fixed offline DSpark semantic vectors",
        "Inspect and verify",
    ),
    CommandSpec(
        "dspark-scheduler",
        "nano_deepseek_v4.dspark_scheduler",
        "check or run the DSpark hardware-aware prefix scheduler",
        "Inspect and verify",
    ),
    CommandSpec(
        "attention-reach",
        "nano_deepseek_v4.attention_reach",
        "run the deterministic attention-path reachability check",
        "Inspect and verify",
    ),
    CommandSpec(
        "parity",
        "nano_deepseek_v4.official_parity",
        "compare equations with the pinned independent references",
        "Inspect and verify",
    ),
    CommandSpec(
        "verify-flash-0731",
        "nano_deepseek_v4.verify_flash_0731_receipt",
        "replay the pinned Flash-0731 metadata receipt",
        "Inspect and verify",
    ),
    CommandSpec(
        "compare",
        "nano_deepseek_v4.compare_bundles",
        "evaluate two bundles on identical held-out byte windows",
        "Research receipts",
    ),
    CommandSpec(
        "aggregate",
        "nano_deepseek_v4.aggregate_comparisons",
        "aggregate contract-compatible comparison reports",
        "Research receipts",
    ),
)

_COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}


def _format_command_help() -> str:
    categories: list[str] = []
    for command in COMMANDS:
        if command.category not in categories:
            categories.append(command.category)

    lines = ["commands:"]
    for category in categories:
        lines.append(f"  {category}:")
        for command in COMMANDS:
            if command.category == category:
                lines.append(f"    {command.name:<20} {command.description}")
    lines.extend(
        (
            "",
            "Run 'nano-deepseek-v4 <command> --help' for command-specific options.",
            "Existing legacy nano-deepseek-v4-* entry points remain supported.",
        )
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nano-deepseek-v4",
        usage="%(prog)s [--version] <command> [<args>]",
        description="Run the nano-deepseek-v4 reference implementation and verification tools.",
        epilog=_format_command_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="show the package version and exit")
    return parser


def _load_command_main(module_name: str) -> CommandMain:
    module = importlib.import_module(module_name)
    command_main = getattr(module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"command module {module_name!r} has no callable main")
    return cast(CommandMain, command_main)


def _run_command(command: CommandSpec, arguments: Sequence[str]) -> int:
    command_main = _load_command_main(command.module)
    previous_program = sys.argv[0]
    try:
        # The command modules also back the legacy console scripts and therefore
        # build their own ArgumentParser. Give them the canonical unified spelling
        # so their help and error output remains directly copyable.
        sys.argv[0] = f"nano-deepseek-v4 {command.name}"
        return command_main(arguments)
    finally:
        sys.argv[0] = previous_program


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not arguments or arguments == ["-h"] or arguments == ["--help"]:
        parser.print_help()
        return 0
    if arguments[0] in {"-h", "--help"}:
        parser.error("help does not accept additional arguments")
    if arguments[0] == "--version":
        if len(arguments) != 1:
            parser.error("--version does not accept additional arguments")
        print(f"nano-deepseek-v4 {__version__}")
        return 0
    if arguments[0].startswith("-"):
        parser.error(f"unrecognized option: {arguments[0]}")

    command = _COMMANDS_BY_NAME.get(arguments[0])
    if command is None:
        parser.error(f"unknown command: {arguments[0]!r}")
    return _run_command(command, arguments[1:])


if __name__ == "__main__":
    raise SystemExit(main())
