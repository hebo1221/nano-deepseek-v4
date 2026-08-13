#!/usr/bin/env python3
"""Verify a GitHub release draft before making it public."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class GitHubReleaseVerificationError(RuntimeError):
    """The draft release does not match the release job's local evidence."""


def collect_asset_inventory(
    *,
    assets: Sequence[Path],
    asset_directories: Sequence[Path],
) -> dict[str, int]:
    """Collect the basename and byte size of every expected release asset."""

    paths = list(assets)
    for directory in asset_directories:
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            raise GitHubReleaseVerificationError(
                f"cannot read expected asset directory {directory}: {exc}"
            ) from exc
        if not entries:
            raise GitHubReleaseVerificationError(
                f"expected asset directory is empty: {directory}"
            )
        paths.extend(entries)

    inventory: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            raise GitHubReleaseVerificationError(
                f"expected release asset is not a file: {path}"
            )
        name = path.name
        if name in inventory:
            raise GitHubReleaseVerificationError(
                f"expected release asset basename is duplicated: {name}"
            )
        inventory[name] = path.stat().st_size
    if not inventory:
        raise GitHubReleaseVerificationError("expected release asset inventory is empty")
    return inventory


def load_release_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GitHubReleaseVerificationError(
            f"cannot read GitHub release metadata {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GitHubReleaseVerificationError("GitHub release metadata must be an object")
    return payload


def _normalized_notes(value: str) -> str:
    return value.replace("\r\n", "\n").rstrip("\n")


def verify_github_release(
    release: Mapping[str, Any],
    *,
    tag: str,
    expected_notes: str,
    expected_assets: Mapping[str, int],
    require_complete_assets: bool,
    expected_draft: bool = True,
) -> dict[str, int]:
    """Validate draft identity, notes, and its partial or complete assets."""

    expected_fields = {
        "tagName": tag,
        "name": tag,
        "isDraft": expected_draft,
        "isPrerelease": False,
    }
    mismatched_fields = {
        field: {"expected": expected, "actual": release.get(field)}
        for field, expected in expected_fields.items()
        if release.get(field) != expected
    }
    if mismatched_fields:
        raise GitHubReleaseVerificationError(
            f"GitHub release identity drift: {mismatched_fields}"
        )

    body = release.get("body")
    if not isinstance(body, str) or _normalized_notes(body) != _normalized_notes(
        expected_notes
    ):
        raise GitHubReleaseVerificationError("GitHub release notes differ from generated notes")

    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        raise GitHubReleaseVerificationError("GitHub release assets must be a list")
    actual_assets: dict[str, int] = {}
    for asset in raw_assets:
        if not isinstance(asset, Mapping):
            raise GitHubReleaseVerificationError("GitHub release asset must be an object")
        name = asset.get("name")
        size = asset.get("size")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise GitHubReleaseVerificationError(
                f"GitHub release asset has invalid name or size: {asset!r}"
            )
        if name in actual_assets:
            raise GitHubReleaseVerificationError(
                f"GitHub release repeats asset name: {name}"
            )
        actual_assets[name] = size

    unexpected = sorted(actual_assets.keys() - expected_assets.keys())
    if unexpected:
        raise GitHubReleaseVerificationError(
            f"GitHub release has unexpected assets: {unexpected}"
        )
    if require_complete_assets:
        missing = sorted(expected_assets.keys() - actual_assets.keys())
        size_mismatches = sorted(
            name
            for name in actual_assets.keys() & expected_assets.keys()
            if actual_assets[name] != expected_assets[name]
        )
        if missing or size_mismatches:
            raise GitHubReleaseVerificationError(
                "GitHub release asset inventory mismatch: "
                f"missing={missing}, size_mismatch={size_mismatches}"
            )
    return actual_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-json", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--asset", action="append", type=Path, default=[])
    parser.add_argument(
        "--asset-directory",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--allow-partial-assets", action="store_true")
    parser.add_argument(
        "--published",
        action="store_true",
        help="require a public release instead of a draft",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        release = load_release_metadata(args.release_json)
        expected_notes = args.notes_file.read_text(encoding="utf-8")
        expected_assets = collect_asset_inventory(
            assets=args.asset,
            asset_directories=args.asset_directory,
        )
        actual_assets = verify_github_release(
            release,
            tag=args.tag,
            expected_notes=expected_notes,
            expected_assets=expected_assets,
            require_complete_assets=not args.allow_partial_assets,
            expected_draft=not args.published,
        )
    except (GitHubReleaseVerificationError, OSError, UnicodeError) as exc:
        print(f"GitHub release verification failed: {exc}", file=sys.stderr)
        return 1
    mode = "partial" if args.allow_partial_assets else "complete"
    state = "public" if args.published else "draft"
    print(f"GitHub release {state}: PASS ({mode}, {len(actual_assets)} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
