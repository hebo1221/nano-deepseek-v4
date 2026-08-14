#!/usr/bin/env python3
"""Verify that a source tree is internally consistent for one release tag."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from datetime import date
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


class ReleaseSourceError(RuntimeError):
    """The source tree is not ready to publish under the requested tag."""


PROJECT_NAME = "nano-deepseek-v4"

TRANSITIONAL_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "README.md": (
        "The Unreleased unified CLI",
        "For every Unreleased command",
        "The current source also provides one discoverable command surface",
        "The current source can turn that smoke test into a guided trace",
        "The current source combines the local execution checks",
        "requires the **current source checkout** until the next release",
        "A wheel built from the current Unreleased source",
    ),
    "docs/verification/conformance.md": (
        "Install the current source with the",
        'python -m pip install -e ".[parity]"',
        'python -m pip install -e ".[parity,official]"',
    ),
    "docs/verification/dspark.md": (
        "Run the standalone check after installing the current source:",
    ),
    "docs/guides/official-checkpoints.md": (
        "A wheel built from the current Unreleased source",
    ),
    "docs/guides/train-and-generate.md": (
        "Install the current source from the repository root",
    ),
    "docs/guides/installation.md": (
        "| `parity` | Current source |",
        "has the official extra, but not parity",
        "Unreleased editable checkout",
    ),
}

RELEASE_DOCUMENTS = tuple(TRANSITIONAL_FRAGMENTS)

_SEMVER_RE = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_CANONICAL_NAME_RE = re.compile(r"[-_.]+")
_LEVEL_TWO_HEADING_RE = re.compile(r"^## [^\r\n]+$", re.MULTILINE)
_VERSION_HEADING_RE = re.compile(
    r"^## \[([^]\r\n]+)](?: - ([^\r\n]+))?$", re.MULTILINE
)
_LINK_DEFINITION_RE = re.compile(r"^\[([^]]+)]:\s*(\S+)\s*$", re.MULTILINE)
_EXACT_REQUIREMENT_RE = re.compile(
    r"(?<![A-Za-z0-9._-])"
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:\[[^]\r\n]+])?=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9!+._-]*)"
)
_EXTRA_ROW_RE = re.compile(
    r"^\|\s*`([^`\r\n]+)`\s*\|\s*([^|\r\n]+?)\s*\|.*\|\s*$",
    re.MULTILINE,
)


def _read_text(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReleaseSourceError(f"cannot read {relative}: {exc}") from exc


def _canonicalize_project_name(value: str) -> str:
    return _CANONICAL_NAME_RE.sub("-", value).lower()


def _project_metadata(root: Path) -> tuple[str, tuple[str, ...]]:
    try:
        payload = tomllib.loads(_read_text(root, "pyproject.toml"))
    except tomllib.TOMLDecodeError as exc:
        raise ReleaseSourceError(f"pyproject.toml is invalid: {exc}") from exc
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ReleaseSourceError("pyproject.toml must define a project table")
    name = project.get("name")
    if not isinstance(name, str) or _canonicalize_project_name(name) != PROJECT_NAME:
        raise ReleaseSourceError(
            f"pyproject.toml project.name must identify {PROJECT_NAME!r}"
        )
    version = project.get("version")
    if not isinstance(version, str) or _SEMVER_RE.fullmatch(version) is None:
        raise ReleaseSourceError(
            "pyproject.toml project.version must be a final X.Y.Z release"
        )
    optional = project.get("optional-dependencies")
    if not isinstance(optional, dict) or not optional:
        raise ReleaseSourceError(
            "pyproject.toml must define the published optional dependencies"
        )
    extras = tuple(sorted(optional))
    return version, extras


def _version_binding_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names: list[str] = []
        for alias in node.names:
            if alias.name == "*":
                names.append("__version__")
            elif alias.asname is not None:
                names.append(alias.asname)
            elif isinstance(node, ast.Import):
                names.append(alias.name.split(".", 1)[0])
            else:
                names.append(alias.name)
        return tuple(names)
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        return (node.name,)
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
        return (node.name,)
    if isinstance(node, ast.MatchMapping) and node.rest is not None:
        return (node.rest,)
    return ()


def _init_version(root: Path) -> str:
    source = _read_text(root, "nano_deepseek_v4/__init__.py")
    try:
        tree = ast.parse(source, filename="nano_deepseek_v4/__init__.py")
    except SyntaxError as exc:
        raise ReleaseSourceError(f"nano_deepseek_v4/__init__.py is invalid: {exc}") from exc
    assignments: list[tuple[ast.expr, ast.Name]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            if not (
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "__version__"
            ):
                raise ReleaseSourceError(
                    "nano_deepseek_v4/__init__.py must define one top-level string __version__"
                )
            assignments.append((statement.value, statement.targets[0]))
        elif isinstance(statement, ast.AnnAssign) and (
            isinstance(statement.target, ast.Name)
            and statement.target.id == "__version__"
        ):
            if statement.value is None:
                raise ReleaseSourceError(
                    "nano_deepseek_v4/__init__.py __version__ must be a literal string"
                )
            assignments.append((statement.value, statement.target))
    if len(assignments) != 1:
        raise ReleaseSourceError(
            "nano_deepseek_v4/__init__.py must define one top-level string __version__"
        )
    value_node, approved_target = assignments[0]
    try:
        value = ast.literal_eval(value_node)
    except (TypeError, ValueError) as exc:
        raise ReleaseSourceError(
            "nano_deepseek_v4/__init__.py __version__ must be a literal string"
        ) from exc
    if not isinstance(value, str):
        raise ReleaseSourceError(
            "nano_deepseek_v4/__init__.py __version__ must be a literal string"
        )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == "__version__"
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node is not approved_target
        ) or "__version__" in _version_binding_names(node):
            raise ReleaseSourceError(
                "nano_deepseek_v4/__init__.py must not write __version__ outside "
                "its single top-level literal assignment"
            )
    return value


def _citation_version(root: Path) -> str:
    citation = _read_text(root, "CITATION.cff")
    try:
        import yaml
        from yaml.nodes import MappingNode, ScalarNode
    except ModuleNotFoundError as exc:  # pragma: no cover - release bootstrap contract
        raise ReleaseSourceError(
            "PyYAML is required to validate CITATION.cff"
        ) from exc
    try:
        root_node = yaml.compose(citation, Loader=yaml.SafeLoader)
        payload = yaml.safe_load(citation)
    except yaml.YAMLError as exc:
        raise ReleaseSourceError(f"CITATION.cff is invalid: {exc}") from exc
    if not isinstance(root_node, MappingNode) or not isinstance(payload, dict):
        raise ReleaseSourceError("CITATION.cff must contain a top-level mapping")
    version_nodes = [
        value_node
        for key_node, value_node in root_node.value
        if isinstance(key_node, ScalarNode) and key_node.value == "version"
    ]
    if len(version_nodes) != 1:
        raise ReleaseSourceError("CITATION.cff must define one top-level version")
    version = payload.get("version")
    if not isinstance(version, str):
        raise ReleaseSourceError("CITATION.cff top-level version must be a string")
    return version


def _single_capture(text: str, pattern: str, label: str) -> str:
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ReleaseSourceError(f"{label} must appear exactly once")
    return matches[0]


def _verify_readme(root: Path, version: str) -> None:
    readme = _read_text(root, "README.md")
    surfaces = {
        "README published-version table row": _single_capture(
            readme,
            r"^\| Published PyPI `([^`]+)` \|",
            "README published-version table row",
        ),
        "README pinned install command": _single_capture(
            readme,
            r'python -m pip install "nano-deepseek-v4==([^"\s]+)"',
            "README pinned install command",
        ),
        "README expected demo version": _single_capture(
            readme,
            r"^nano-deepseek-v4 ([0-9]+\.[0-9]+\.[0-9]+)$",
            "README expected demo version",
        ),
    }
    mismatched = {label: value for label, value in surfaces.items() if value != version}
    if mismatched:
        raise ReleaseSourceError(f"README published-version fields are stale: {mismatched}")


def _verify_document_versions(
    root: Path,
    version: str,
    published_extras: tuple[str, ...],
) -> None:
    documents = {
        relative: _read_text(root, relative) for relative in RELEASE_DOCUMENTS
    }
    pins = {
        match.group("version")
        for text in documents.values()
        for match in _EXACT_REQUIREMENT_RE.finditer(text)
        if _canonicalize_project_name(match.group("name")) == PROJECT_NAME
    }
    if pins != {version}:
        raise ReleaseSourceError(
            f"documented exact package versions must all equal {version}: {sorted(pins)}"
        )

    installation = documents["docs/guides/installation.md"]
    rows: dict[str, str] = {}
    for extra, availability in _EXTRA_ROW_RE.findall(installation):
        if extra in rows:
            raise ReleaseSourceError(
                f"installation repeats optional-extra row {extra!r}"
            )
        rows[extra] = availability.strip()
    expected_extras = set(published_extras)
    actual_extras = set(rows)
    if actual_extras != expected_extras:
        raise ReleaseSourceError(
            "installation optional-extra rows do not match pyproject.toml: "
            f"missing={sorted(expected_extras - actual_extras)}, "
            f"unexpected={sorted(actual_extras - expected_extras)}"
        )
    expected_availability = f"PyPI {version} and current source"
    stale_rows = {
        extra: rows[extra]
        for extra in published_extras
        if rows[extra] != expected_availability
    }
    if stale_rows:
        raise ReleaseSourceError(
            f"installation optional-extra availability is stale: {stale_rows}"
        )
    expected_combined_pin = f"nano-deepseek-v4[official,parity]=={version}"
    if installation.count(expected_combined_pin) != 1:
        raise ReleaseSourceError(
            "installation must contain one published official+parity version pin"
        )


def _verify_transition_claims(root: Path) -> None:
    found: dict[str, list[str]] = {}
    for relative, fragments in TRANSITIONAL_FRAGMENTS.items():
        text = _read_text(root, relative)
        matches = [fragment for fragment in fragments if fragment in text]
        if matches:
            found[relative] = matches
    if found:
        raise ReleaseSourceError(f"release docs still contain transition claims: {found}")


def _heading_body(text: str, headings: list[re.Match[str]], index: int) -> str:
    start = headings[index].end()
    end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
    return text[start:end]


def _has_release_notes(body: str) -> bool:
    without_comments = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    for line in without_comments.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}(?:\s|$)", stripped):
            continue
        if re.fullmatch(r"(?:[-*_]\s*){3,}", stripped):
            continue
        if _LINK_DEFINITION_RE.fullmatch(stripped):
            continue
        return True
    return False


def _semantic_tuple(version: str) -> tuple[int, int, int]:
    if _SEMVER_RE.fullmatch(version) is None:
        raise ReleaseSourceError(f"CHANGELOG release version is not final SemVer: {version!r}")
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def _verify_changelog(root: Path, version: str) -> None:
    changelog = _read_text(root, "CHANGELOG.md")
    level_two_headings = list(_LEVEL_TWO_HEADING_RE.finditer(changelog))
    headings = list(_VERSION_HEADING_RE.finditer(changelog))
    if not headings:
        raise ReleaseSourceError("CHANGELOG has no release headings")
    if [(heading.start(), heading.group(0)) for heading in headings] != [
        (heading.start(), heading.group(0)) for heading in level_two_headings
    ]:
        valid_starts = {heading.start() for heading in headings}
        malformed = [
            heading.group(0)
            for heading in level_two_headings
            if heading.start() not in valid_starts
        ]
        raise ReleaseSourceError(
            f"CHANGELOG has malformed level-two release headings: {malformed}"
        )

    unreleased_indexes = [
        index for index, heading in enumerate(headings) if heading.group(1) == "Unreleased"
    ]
    if len(unreleased_indexes) != 1:
        raise ReleaseSourceError("CHANGELOG must contain exactly one Unreleased section")
    unreleased_index = unreleased_indexes[0]
    if unreleased_index != 0:
        raise ReleaseSourceError("CHANGELOG Unreleased section must be first")
    if headings[unreleased_index].group(2) is not None:
        raise ReleaseSourceError("CHANGELOG Unreleased section must not have a date")
    if _heading_body(changelog, headings, unreleased_index).strip():
        raise ReleaseSourceError("CHANGELOG Unreleased section must be empty at a release tag")

    release_headings = [heading for heading in headings if heading.group(1) != "Unreleased"]
    release_versions = [heading.group(1) for heading in release_headings]
    if len(release_versions) < 2:
        raise ReleaseSourceError("CHANGELOG has no previous release for comparison")
    if len(release_versions) != len(set(release_versions)):
        raise ReleaseSourceError("CHANGELOG contains duplicate release sections")
    if release_versions[0] != version:
        raise ReleaseSourceError("CHANGELOG release sections are out of order")
    current_index = headings.index(release_headings[0])
    if not _has_release_notes(_heading_body(changelog, headings, current_index)):
        raise ReleaseSourceError(
            "CHANGELOG current release section must contain release notes"
        )

    semantic_versions = [_semantic_tuple(value) for value in release_versions]
    if any(current <= following for current, following in zip(semantic_versions, semantic_versions[1:], strict=False)):
        raise ReleaseSourceError("CHANGELOG release versions must be strictly descending")

    release_dates: list[date] = []
    for heading in release_headings:
        raw_date = heading.group(2)
        if raw_date is None:
            raise ReleaseSourceError(
                f"CHANGELOG release {heading.group(1)} is missing its date"
            )
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise ReleaseSourceError(
                f"CHANGELOG release {heading.group(1)} date is invalid: {raw_date}"
            ) from exc
        if parsed_date.isoformat() != raw_date:
            raise ReleaseSourceError(
                f"CHANGELOG release {heading.group(1)} date is not canonical: {raw_date}"
            )
        release_dates.append(parsed_date)
    if any(
        current < following
        for current, following in zip(release_dates, release_dates[1:], strict=False)
    ):
        raise ReleaseSourceError(
            "CHANGELOG release dates must be in reverse chronological order"
        )

    links: dict[str, tuple[str, str]] = {}
    for label, url in _LINK_DEFINITION_RE.findall(changelog):
        normalized_label = " ".join(label.split()).casefold()
        if normalized_label in links:
            raise ReleaseSourceError(f"CHANGELOG repeats link definition [{label}]")
        links[normalized_label] = (label, url)
    repository = "https://github.com/hebo1221/nano-deepseek-v4"
    expected_links: dict[str, str] = {
        "Unreleased": f"{repository}/compare/v{version}...HEAD",
    }
    for index, release_version in enumerate(release_versions):
        if index + 1 < len(release_versions):
            previous_version = release_versions[index + 1]
            expected_links[release_version] = (
                f"{repository}/compare/v{previous_version}...v{release_version}"
            )
        else:
            expected_links[release_version] = f"{repository}/tree/v{release_version}"
    mismatched_links = {
        label: {
            "expected": expected,
            "actual": (
                links.get(" ".join(label.split()).casefold(), ("", None))[1]
            ),
        }
        for label, expected in expected_links.items()
        if links.get(" ".join(label.split()).casefold(), ("", None))[1]
        != expected
    }
    if mismatched_links:
        raise ReleaseSourceError(
            f"CHANGELOG comparison links are missing or stale: {mismatched_links}"
        )


def verify_release_source(root: Path, release_tag: str) -> str:
    """Verify release metadata, documentation, and changelog invariants."""

    resolved_root = root.resolve()
    version, published_extras = _project_metadata(resolved_root)
    expected_tag = f"v{version}"
    if release_tag != expected_tag:
        raise ReleaseSourceError(
            f"release tag {release_tag!r} must equal {expected_tag!r}"
        )
    init_version = _init_version(resolved_root)
    if init_version != version:
        raise ReleaseSourceError("__version__ must match pyproject.toml")
    citation_version = _citation_version(resolved_root)
    if citation_version != version:
        raise ReleaseSourceError("CITATION.cff version must match pyproject.toml")
    _verify_readme(resolved_root, version)
    _verify_document_versions(resolved_root, version, published_extras)
    _verify_transition_claims(resolved_root)
    _verify_changelog(resolved_root, version)
    return version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source tree to verify (default: repository containing this script)",
    )
    parser.add_argument("--tag", required=True, help="exact release tag, for example v0.3.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        version = verify_release_source(args.root, args.tag)
    except (ReleaseSourceError, ValueError) as exc:
        print(f"release source verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"release source: PASS (v{version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
