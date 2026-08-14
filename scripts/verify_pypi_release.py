#!/usr/bin/env python3
"""Verify that a PyPI release exposes the exact artifacts in a release manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReleaseVerificationError(RuntimeError):
    """The public PyPI metadata does not match the release manifest."""


def _normalize_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def load_release_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"cannot read release manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseVerificationError("release manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ReleaseVerificationError("release manifest schema_version must be 1")
    version = payload.get("version")
    artifacts = payload.get("artifacts")
    if not isinstance(version, str) or not version:
        raise ReleaseVerificationError("release manifest version must be a non-empty string")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ReleaseVerificationError("release manifest artifacts must be a non-empty object")
    for filename, digest in artifacts.items():
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ReleaseVerificationError(f"invalid release artifact filename: {filename!r}")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ReleaseVerificationError(f"invalid SHA-256 for release artifact {filename!r}")
    return payload


def verify_release_payload(
    project: str,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Compare one PyPI version JSON response with a local release manifest."""

    version = manifest.get("version")
    artifacts = manifest.get("artifacts")
    if not isinstance(version, str) or not isinstance(artifacts, Mapping):
        raise ReleaseVerificationError("release manifest is missing version or artifacts")

    info = payload.get("info")
    urls = payload.get("urls")
    if not isinstance(info, Mapping) or not isinstance(urls, list):
        raise ReleaseVerificationError("PyPI response is missing info or urls")
    published_name = info.get("name")
    published_version = info.get("version")
    if not isinstance(published_name, str) or (
        _normalize_project_name(published_name) != _normalize_project_name(project)
    ):
        raise ReleaseVerificationError(
            f"PyPI project name {published_name!r} does not match {project!r}"
        )
    if published_version != version:
        raise ReleaseVerificationError(
            f"PyPI version {published_version!r} does not match manifest version {version!r}"
        )

    published: dict[str, str] = {}
    for entry in urls:
        if not isinstance(entry, Mapping):
            raise ReleaseVerificationError("PyPI urls entry must be an object")
        filename = entry.get("filename")
        digests = entry.get("digests")
        if not isinstance(filename, str) or not isinstance(digests, Mapping):
            raise ReleaseVerificationError("PyPI urls entry is missing filename or digests")
        sha256 = digests.get("sha256")
        if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ReleaseVerificationError(f"PyPI file {filename!r} has no valid SHA-256")
        if filename in published:
            raise ReleaseVerificationError(f"PyPI response repeats filename {filename!r}")
        published[filename] = sha256

    expected = dict(artifacts)
    missing = sorted(expected.keys() - published.keys())
    unexpected = sorted(published.keys() - expected.keys())
    mismatched = sorted(
        filename
        for filename in expected.keys() & published.keys()
        if expected[filename] != published[filename]
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatched:
            details.append(f"sha256_mismatch={mismatched}")
        raise ReleaseVerificationError("PyPI artifact inventory mismatch: " + ", ".join(details))
    return published


def fetch_release_payload(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "nano-deepseek-v4-release-verifier/1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, OSError) as exc:
        raise ReleaseVerificationError(f"cannot fetch {url}: {exc}") from exc
    if len(body) > _MAX_RESPONSE_BYTES:
        raise ReleaseVerificationError("PyPI JSON response exceeds 2 MiB")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ReleaseVerificationError(f"PyPI returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseVerificationError("PyPI response must be a JSON object")
    return payload


def wait_for_release(
    project: str,
    manifest: Mapping[str, Any],
    *,
    base_url: str = "https://pypi.org",
    attempts: int = 12,
    delay_seconds: float = 5.0,
    timeout_seconds: float = 15.0,
    fetcher: Callable[[str, float], Mapping[str, Any]] = fetch_release_payload,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        raise ReleaseVerificationError("release manifest version must be a non-empty string")
    url = (
        f"{base_url.rstrip('/')}/pypi/{quote(project, safe='')}/"
        f"{quote(version, safe='')}/json"
    )

    last_error: ReleaseVerificationError | None = None
    for attempt in range(1, attempts + 1):
        try:
            payload = fetcher(url, timeout_seconds)
            return verify_release_payload(project, manifest, payload)
        except ReleaseVerificationError as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"PyPI verification attempt {attempt}/{attempts} failed: {exc}; retrying",
                file=sys.stderr,
            )
            sleeper(delay_seconds)
    assert last_error is not None
    raise ReleaseVerificationError(
        f"PyPI publication did not verify after {attempts} attempts: {last_error}"
    ) from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="PyPI project name")
    parser.add_argument("--manifest", type=Path, required=True, help="release-manifest.json")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--base-url", default="https://pypi.org", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_release_manifest(args.manifest)
        published = wait_for_release(
            args.project,
            manifest,
            base_url=args.base_url,
            attempts=args.attempts,
            delay_seconds=args.delay_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except (ReleaseVerificationError, ValueError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"verified {args.project} {manifest['version']} on PyPI: "
        f"{len(published)} artifacts match release-manifest.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
