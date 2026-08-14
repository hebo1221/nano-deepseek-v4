"""Verify that wheel and source distributions preserve the release payload."""

from __future__ import annotations

import argparse
import base64
import binascii
import configparser
import csv
import hashlib
import io
import re
import stat
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from email import policy
from email.headerregistry import Address
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib


class DistributionVerificationError(RuntimeError):
    """Raised when a built distribution does not match the source checkout."""


def _source_files(root: Path, pattern: str) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.glob(pattern))
        if path.is_file()
    }


def _expected_package_files(root: Path) -> dict[str, bytes]:
    package = root / "nano_deepseek_v4"
    allowed = {".json", ".py"}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(package.rglob("*"))
        if path.is_file() and (path.suffix in allowed or path.name == "py.typed")
    }


def _member_path(name: str, *, label: str, is_directory: bool = False) -> PurePosixPath:
    if "\\" in name:
        raise DistributionVerificationError(f"{label} has non-canonical member {name!r}")
    stripped = name[:-1] if is_directory and name.endswith("/") else name
    member_path = PurePosixPath(stripped)
    if (
        not stripped
        or member_path == PurePosixPath(".")
        or member_path.is_absolute()
        or ".." in member_path.parts
    ):
        raise DistributionVerificationError(f"{label} has unsafe member {name!r}")
    if stripped != member_path.as_posix() or any(part in {"", "."} for part in stripped.split("/")):
        raise DistributionVerificationError(f"{label} has non-canonical member {name!r}")
    return member_path


def _read_wheel(path: Path) -> tuple[dict[str, bytes], set[str]]:
    with ZipFile(path) as archive:
        files: dict[str, bytes] = {}
        directories: set[str] = set()
        seen: set[str] = set()
        for member in archive.infolist():
            member_path = _member_path(
                member.filename,
                label="wheel",
                is_directory=member.is_dir(),
            )
            normalized = member_path.as_posix()
            if normalized in seen:
                raise DistributionVerificationError(f"wheel has duplicate member {normalized}")
            seen.add(normalized)
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise DistributionVerificationError(f"wheel has unsafe member {member.filename!r}")
            if member.is_dir():
                directories.add(normalized)
            else:
                files[normalized] = archive.read(member)
        return files, directories


def _read_sdist(path: Path, expected_root: str) -> tuple[dict[str, bytes], set[str]]:
    with tarfile.open(path, mode="r:gz") as archive:
        all_members = archive.getmembers()
        unsafe: list[str] = []
        seen: set[str] = set()
        for member in all_members:
            try:
                member_path = _member_path(
                    member.name,
                    label="sdist",
                    is_directory=member.isdir(),
                )
            except DistributionVerificationError as error:
                raise DistributionVerificationError(str(error)) from error
            normalized = member_path.as_posix()
            if normalized in seen:
                raise DistributionVerificationError(f"sdist has duplicate member {normalized}")
            seen.add(normalized)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                unsafe.append(member.name)
        if unsafe:
            raise DistributionVerificationError(f"sdist has unsafe members: {unsafe}")
        roots = {
            _member_path(
                member.name,
                label="sdist",
                is_directory=member.isdir(),
            ).parts[0]
            for member in all_members
        }
        if roots != {expected_root}:
            raise DistributionVerificationError(
                f"sdist top-level directory drift: expected={expected_root!r}, "
                f"actual={sorted(roots)}"
            )
        files: dict[str, bytes] = {}
        directories: set[str] = set()
        for member in all_members:
            member_path = _member_path(
                member.name,
                label="sdist",
                is_directory=member.isdir(),
            )
            if member_path.as_posix() == expected_root:
                continue
            relative = member_path.relative_to(expected_root).as_posix()
            if member.isdir():
                directories.add(relative)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise DistributionVerificationError(f"could not read sdist member {member.name}")
            files[relative] = extracted.read()
        return files, directories


def _allowed_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for path in paths:
        parent = PurePosixPath(path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _require_exact_archive(
    label: str,
    actual: Mapping[str, bytes],
    expected: Mapping[str, bytes],
    generated: set[str],
    directories: set[str],
) -> None:
    expected_paths = set(expected) | generated
    missing = sorted(expected_paths - set(actual))
    unexpected = sorted(set(actual) - expected_paths)
    changed = sorted(path for path, content in expected.items() if actual.get(path) != content)
    unexpected_directories = sorted(directories - _allowed_directories(expected_paths))
    if missing or unexpected or changed or unexpected_directories:
        raise DistributionVerificationError(
            f"{label} payload drift: missing={missing}, unexpected={unexpected}, "
            f"changed={changed}, unexpected_directories={unexpected_directories}"
        )


def _project_version(project: Mapping[str, object]) -> Version:
    try:
        return Version(str(project["version"]))
    except InvalidVersion as error:
        raise DistributionVerificationError(
            f"pyproject has invalid PEP 440 version {project.get('version')!r}"
        ) from error


def _distribution_component(name: str) -> str:
    return canonicalize_name(name).replace("-", "_")


def _version_component(version: str) -> str:
    try:
        normalized = str(Version(version))
    except InvalidVersion as error:
        raise DistributionVerificationError(f"invalid PEP 440 version {version!r}") from error
    return normalized.replace("-", "_")


def _entry_point_group(
    project: Mapping[str, object],
    project_key: str,
) -> dict[str, str]:
    entries = project.get(project_key, {})
    if not isinstance(entries, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in entries.items()
    ):
        raise DistributionVerificationError(
            f"pyproject project.{project_key} must be a string table"
        )
    return dict(entries) if entries else {}


def _expected_entry_points(project: Mapping[str, object]) -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    console_scripts = _entry_point_group(project, "scripts")
    if console_scripts:
        expected["console_scripts"] = console_scripts
    gui_scripts = _entry_point_group(project, "gui-scripts")
    if gui_scripts:
        expected["gui_scripts"] = gui_scripts

    custom_groups = project.get("entry-points", {})
    if not isinstance(custom_groups, dict):
        raise DistributionVerificationError("pyproject project.entry-points must be a table")
    for group, entries in custom_groups.items():
        if not isinstance(group, str) or not isinstance(entries, dict) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in entries.items()
        ):
            raise DistributionVerificationError(
                "pyproject project.entry-points groups must be string tables"
            )
        if group in {"console_scripts", "gui_scripts"}:
            raise DistributionVerificationError(
                f"pyproject project.entry-points.{group} conflicts with its dedicated table"
            )
        if entries:
            expected[group] = dict(entries)
    return expected


def _has_entry_points(project: Mapping[str, object]) -> bool:
    return bool(_expected_entry_points(project))


def _license_file_paths(root: Path, project: Mapping[str, object]) -> list[str]:
    patterns = project.get("license-files", [])
    if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
        raise DistributionVerificationError("pyproject project.license-files must be a list")

    paths: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for license_path in sorted(root.glob(pattern)):
            if not license_path.is_file():
                continue
            relative = license_path.relative_to(root).as_posix()
            if relative not in seen:
                paths.append(relative)
                seen.add(relative)
    return paths


def _wheel_metadata(
    root: Path,
    project: Mapping[str, object],
) -> tuple[str, dict[str, bytes], set[str]]:
    distribution = _distribution_component(str(project["name"]))
    version = _version_component(str(project["version"]))
    dist_info = f"{distribution}-{version}.dist-info"
    copied: dict[str, bytes] = {}
    for relative in _license_file_paths(root, project):
        copied[f"{dist_info}/licenses/{relative}"] = (root / relative).read_bytes()
    generated = {
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/WHEEL",
        f"{dist_info}/top_level.txt",
    }
    if _has_entry_points(project):
        generated.add(f"{dist_info}/entry_points.txt")
    return dist_info, copied, generated


def _sdist_metadata(project: Mapping[str, object]) -> tuple[str, set[str]]:
    distribution = _distribution_component(str(project["name"]))
    version = _version_component(str(project["version"]))
    egg_info = f"{distribution}.egg-info"
    generated = {
        "PKG-INFO",
        "setup.cfg",
        f"{egg_info}/PKG-INFO",
        f"{egg_info}/SOURCES.txt",
        f"{egg_info}/dependency_links.txt",
        f"{egg_info}/top_level.txt",
    }
    if _has_entry_points(project):
        generated.add(f"{egg_info}/entry_points.txt")
    if project.get("dependencies") or project.get("optional-dependencies"):
        generated.add(f"{egg_info}/requires.txt")
    return f"{distribution}-{version}", generated


def _expected_wheel_filename(project: Mapping[str, object]) -> str:
    distribution = _distribution_component(str(project["name"]))
    version = _version_component(str(project["version"]))
    return f"{distribution}-{version}-py3-none-any.whl"


def _expected_sdist_filename(project: Mapping[str, object]) -> str:
    distribution = _distribution_component(str(project["name"]))
    version = _version_component(str(project["version"]))
    return f"{distribution}-{version}.tar.gz"


def _verify_wheel_filename(
    path: Path,
    project: Mapping[str, object],
    *,
    label: str,
) -> None:
    expected_filename = _expected_wheel_filename(project)
    if path.name != expected_filename:
        raise DistributionVerificationError(
            f"{label} filename drift: expected={expected_filename!r}, actual={path.name!r}"
        )
    try:
        distribution, version, build, tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as error:
        raise DistributionVerificationError(f"{label} has invalid filename {path.name!r}") from error
    expected_distribution = canonicalize_name(str(project["name"]))
    expected_version = _project_version(project)
    expected_tags = frozenset({Tag("py3", "none", "any")})
    if (
        distribution != expected_distribution
        or version != expected_version
        or build
        or tags != expected_tags
    ):
        raise DistributionVerificationError(
            f"{label} filename identity drift: distribution={distribution!r}, "
            f"version={version!s}, build={build!r}, tags={sorted(map(str, tags))}"
        )


def _verify_sdist_filename(path: Path, project: Mapping[str, object]) -> None:
    expected_filename = _expected_sdist_filename(project)
    if path.name != expected_filename:
        raise DistributionVerificationError(
            f"sdist filename drift: expected={expected_filename!r}, actual={path.name!r}"
        )
    try:
        distribution, version = parse_sdist_filename(path.name)
    except InvalidSdistFilename as error:
        raise DistributionVerificationError(f"sdist has invalid filename {path.name!r}") from error
    if (
        distribution != canonicalize_name(str(project["name"]))
        or version != _project_version(project)
    ):
        raise DistributionVerificationError(
            f"sdist filename identity drift: distribution={distribution!r}, version={version!s}"
        )
    with path.open("rb") as stream:
        magic = stream.read(2)
    if magic != b"\x1f\x8b":
        raise DistributionVerificationError("sdist is not gzip-compressed")


def _parse_metadata(content: bytes, *, label: str) -> Message:
    try:
        message = BytesParser(policy=policy.default).parsebytes(content)
    except (UnicodeError, ValueError) as error:
        raise DistributionVerificationError(f"{label} is not valid email metadata") from error
    if message.defects:
        raise DistributionVerificationError(f"{label} has malformed metadata: {message.defects}")
    return message


def _single_header(
    message: Message,
    name: str,
    *,
    label: str,
    required: bool = True,
) -> str | None:
    values = message.get_all(name, [])
    if required and len(values) != 1:
        raise DistributionVerificationError(
            f"{label} must contain exactly one {name} header, found {len(values)}"
        )
    if not required:
        if values:
            raise DistributionVerificationError(f"{label} has unexpected {name} header")
        return None
    return str(values[0]).strip()


def _requirement_without_marker(requirement: Requirement) -> str:
    value = requirement.name
    if requirement.extras:
        value += f"[{','.join(sorted(requirement.extras))}]"
    if requirement.url:
        value += f" @ {requirement.url}"
    else:
        value += str(requirement.specifier)
    return value


def _parse_requirement(value: str, *, label: str) -> Requirement:
    try:
        return Requirement(value)
    except InvalidRequirement as error:
        raise DistributionVerificationError(f"{label} has invalid requirement {value!r}") from error


def _expected_requirements(project: Mapping[str, object]) -> Counter[Requirement]:
    expected: Counter[Requirement] = Counter()
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise DistributionVerificationError("pyproject project.dependencies must be a list")
    for value in dependencies:
        if not isinstance(value, str):
            raise DistributionVerificationError("pyproject dependency must be a string")
        expected[_parse_requirement(value, label="pyproject")] += 1

    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise DistributionVerificationError(
            "pyproject project.optional-dependencies must be a table"
        )
    for raw_extra, values in optional.items():
        if not isinstance(raw_extra, str) or not isinstance(values, list):
            raise DistributionVerificationError("pyproject optional dependency group is invalid")
        extra = canonicalize_name(raw_extra)
        for value in values:
            if not isinstance(value, str):
                raise DistributionVerificationError("pyproject dependency must be a string")
            requirement = _parse_requirement(value, label="pyproject")
            extra_marker = f'extra == "{extra}"'
            if requirement.marker is not None:
                marker = f"({requirement.marker}) and {extra_marker}"
            else:
                marker = extra_marker
            rendered = f"{_requirement_without_marker(requirement)}; {marker}"
            expected[_parse_requirement(rendered, label="pyproject")] += 1
    return expected


def _render_requirements(requirements: Counter[Requirement]) -> list[str]:
    rendered: list[str] = []
    for requirement, count in requirements.items():
        rendered.extend([str(requirement)] * count)
    return sorted(rendered)


def _optional_static_header(
    message: Message,
    name: str,
    expected: str | None,
    *,
    label: str,
) -> None:
    actual = _single_header(message, name, label=label, required=expected is not None)
    if actual != expected:
        raise DistributionVerificationError(
            f"{label} {name} drift: expected={expected!r}, actual={actual!r}"
        )


def _string_list(project: Mapping[str, object], name: str) -> list[str]:
    values = project.get(name, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise DistributionVerificationError(f"pyproject project.{name} must be a list")
    return values


def _expected_people(project: Mapping[str, object], name: str) -> tuple[str | None, str | None]:
    people = project.get(name, [])
    if not isinstance(people, list):
        raise DistributionVerificationError(f"pyproject project.{name} must be a list")

    names: list[str] = []
    addresses: list[str] = []
    for person in people:
        if not isinstance(person, dict):
            raise DistributionVerificationError(f"pyproject project.{name} entry is invalid")
        display_name = person.get("name")
        email = person.get("email")
        if display_name is not None and not isinstance(display_name, str):
            raise DistributionVerificationError(f"pyproject project.{name} name is invalid")
        if email is not None and not isinstance(email, str):
            raise DistributionVerificationError(f"pyproject project.{name} email is invalid")
        if display_name is None and email is None:
            raise DistributionVerificationError(f"pyproject project.{name} entry is empty")
        if email is None:
            names.append(display_name or "")
        elif display_name is None:
            addresses.append(email)
        else:
            try:
                addresses.append(str(Address(display_name=display_name, addr_spec=email)))
            except ValueError as error:
                raise DistributionVerificationError(
                    f"pyproject project.{name} email is invalid"
                ) from error
    return ", ".join(names) or None, ", ".join(addresses) or None


def _expected_readme(root: Path, project: Mapping[str, object]) -> tuple[bytes, str | None]:
    readme = project.get("readme")
    if readme is None:
        return b"", None

    content_type: object
    try:
        if isinstance(readme, str):
            suffix = PurePosixPath(readme).suffix.lower()
            content_types = {
                ".md": "text/markdown",
                ".rst": "text/x-rst",
                ".txt": "text/plain",
            }
            content_type = content_types.get(suffix)
            if content_type is None:
                raise DistributionVerificationError(
                    "pyproject project.readme has an unsupported file extension"
                )
            description = (root / readme).read_text(encoding="utf-8")
        elif isinstance(readme, dict):
            content_type = readme.get("content-type")
            if not isinstance(content_type, str):
                raise DistributionVerificationError(
                    "pyproject project.readme table must define content-type"
                )
            file = readme.get("file")
            text = readme.get("text")
            if isinstance(file, str) and text is None:
                description = (root / file).read_text(encoding="utf-8")
            elif isinstance(text, str) and file is None:
                description = text
            else:
                raise DistributionVerificationError(
                    "pyproject project.readme table must define exactly one of file or text"
                )
        else:
            raise DistributionVerificationError("pyproject project.readme is invalid")
    except (OSError, UnicodeError) as error:
        raise DistributionVerificationError("could not read pyproject project.readme") from error

    if description and not description.endswith("\n"):
        description += "\n"
    return description.encode("utf-8"), str(content_type)


def _verify_core_metadata(
    content: bytes,
    root: Path,
    project: Mapping[str, object],
    *,
    label: str,
) -> None:
    message = _parse_metadata(content, label=label)
    metadata_version = _single_header(message, "Metadata-Version", label=label)
    if metadata_version != "2.4":
        raise DistributionVerificationError(
            f"{label} Metadata-Version drift: expected='2.4', actual={metadata_version!r}"
        )

    actual_name = _single_header(message, "Name", label=label)
    if actual_name is None or canonicalize_name(actual_name) != canonicalize_name(
        str(project["name"])
    ):
        raise DistributionVerificationError(
            f"{label} name drift: expected={project['name']!r}, actual={actual_name!r}"
        )

    actual_version = _single_header(message, "Version", label=label)
    try:
        parsed_version = Version(actual_version or "")
    except InvalidVersion as error:
        raise DistributionVerificationError(
            f"{label} has invalid Version {actual_version!r}"
        ) from error
    if parsed_version != _project_version(project):
        raise DistributionVerificationError(
            f"{label} version drift: expected={project['version']!r}, actual={actual_version!r}"
        )

    expected_python = project.get("requires-python")
    actual_python = _single_header(
        message,
        "Requires-Python",
        label=label,
        required=expected_python is not None,
    )
    if expected_python is not None:
        try:
            expected_specifier = SpecifierSet(str(expected_python))
            actual_specifier = SpecifierSet(actual_python or "")
        except InvalidSpecifier as error:
            raise DistributionVerificationError(
                f"{label} has invalid Requires-Python {actual_python!r}"
            ) from error
        if actual_specifier != expected_specifier:
            raise DistributionVerificationError(
                f"{label} Requires-Python drift: expected={expected_specifier!s}, "
                f"actual={actual_specifier!s}"
            )

    actual_requirements: Counter[Requirement] = Counter()
    for value in message.get_all("Requires-Dist", []):
        actual_requirements[_parse_requirement(str(value), label=label)] += 1
    expected_requirements = _expected_requirements(project)
    if actual_requirements != expected_requirements:
        raise DistributionVerificationError(
            f"{label} dependency drift: expected={_render_requirements(expected_requirements)}, "
            f"actual={_render_requirements(actual_requirements)}"
        )

    optional = project.get("optional-dependencies", {})
    expected_extras = (
        {canonicalize_name(str(extra)) for extra in optional} if isinstance(optional, dict) else set()
    )
    actual_extras = [canonicalize_name(str(value)) for value in message.get_all("Provides-Extra", [])]
    if len(actual_extras) != len(set(actual_extras)) or set(actual_extras) != expected_extras:
        raise DistributionVerificationError(
            f"{label} extras drift: expected={sorted(expected_extras)}, "
            f"actual={sorted(actual_extras)}"
        )

    description = project.get("description")
    if description is not None and not isinstance(description, str):
        raise DistributionVerificationError("pyproject project.description must be a string")
    _optional_static_header(message, "Summary", description, label=label)

    author, author_email = _expected_people(project, "authors")
    _optional_static_header(message, "Author", author, label=label)
    _optional_static_header(message, "Author-email", author_email, label=label)

    license_expression = project.get("license")
    if license_expression is not None and not isinstance(license_expression, str):
        raise DistributionVerificationError(
            "pyproject project.license must be an SPDX expression string"
        )
    _optional_static_header(message, "License-Expression", license_expression, label=label)
    _optional_static_header(message, "License", None, label=label)

    expected_license_files = _license_file_paths(root, project)
    actual_license_files = [str(value).strip() for value in message.get_all("License-File", [])]
    if actual_license_files != expected_license_files:
        raise DistributionVerificationError(
            f"{label} License-File inventory drift: expected={expected_license_files}, "
            f"actual={actual_license_files}"
        )

    keywords = _string_list(project, "keywords")
    _optional_static_header(message, "Keywords", ",".join(keywords) or None, label=label)

    expected_classifiers = _string_list(project, "classifiers")
    actual_classifiers = [str(value).strip() for value in message.get_all("Classifier", [])]
    if actual_classifiers != expected_classifiers:
        raise DistributionVerificationError(
            f"{label} Classifier drift: expected={expected_classifiers}, "
            f"actual={actual_classifiers}"
        )

    urls = project.get("urls", {})
    if not isinstance(urls, dict) or not all(
        isinstance(name, str) and isinstance(url, str) for name, url in urls.items()
    ):
        raise DistributionVerificationError("pyproject project.urls must be a string table")
    expected_urls = [f"{name}, {url}" for name, url in urls.items()]
    actual_urls = [str(value).strip() for value in message.get_all("Project-URL", [])]
    if actual_urls != expected_urls:
        raise DistributionVerificationError(
            f"{label} Project-URL drift: expected={expected_urls}, actual={actual_urls}"
        )

    expected_body, expected_content_type = _expected_readme(root, project)
    _optional_static_header(
        message,
        "Description-Content-Type",
        expected_content_type,
        label=label,
    )
    actual_body = message.get_payload(decode=True)
    if not isinstance(actual_body, bytes) or actual_body != expected_body:
        raise DistributionVerificationError(f"{label} README description body drift")

    expected_dynamic = ["license-file"] if expected_license_files else []
    actual_dynamic = [str(value).strip() for value in message.get_all("Dynamic", [])]
    if actual_dynamic != expected_dynamic:
        raise DistributionVerificationError(
            f"{label} Dynamic drift: expected={expected_dynamic}, actual={actual_dynamic}"
        )


def _verify_wheel_metadata(
    wheel_files: Mapping[str, bytes],
    dist_info: str,
    root: Path,
    project: Mapping[str, object],
    *,
    label: str,
) -> None:
    _verify_core_metadata(
        wheel_files[f"{dist_info}/METADATA"],
        root,
        project,
        label=f"{label} METADATA",
    )
    wheel_headers = _parse_metadata(wheel_files[f"{dist_info}/WHEEL"], label=f"{label} WHEEL")
    expected = {
        "Wheel-Version": "1.0",
        "Root-Is-Purelib": "true",
    }
    for name, expected_value in expected.items():
        actual = _single_header(wheel_headers, name, label=f"{label} WHEEL")
        if actual != expected_value:
            raise DistributionVerificationError(
                f"{label} WHEEL {name} drift: expected={expected_value!r}, actual={actual!r}"
            )
    tags = [str(value).strip() for value in wheel_headers.get_all("Tag", [])]
    if tags != ["py3-none-any"]:
        raise DistributionVerificationError(
            f"{label} WHEEL tag drift: expected=['py3-none-any'], actual={tags}"
        )


def _verify_record(
    wheel_files: Mapping[str, bytes],
    dist_info: str,
    *,
    label: str,
) -> None:
    record_path = f"{dist_info}/RECORD"
    try:
        text = wheel_files[record_path].decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeError, csv.Error) as error:
        raise DistributionVerificationError(f"{label} RECORD is not valid CSV") from error

    entries: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise DistributionVerificationError(f"{label} RECORD row must have three fields: {row}")
        raw_path, digest, size = row
        member_path = _member_path(raw_path, label=f"{label} RECORD")
        normalized = member_path.as_posix()
        if normalized in entries:
            raise DistributionVerificationError(f"{label} RECORD has duplicate member {normalized}")
        entries[normalized] = (digest, size)

    missing = sorted(set(wheel_files) - set(entries))
    unexpected = sorted(set(entries) - set(wheel_files))
    if missing or unexpected:
        raise DistributionVerificationError(
            f"{label} RECORD membership drift: missing={missing}, unexpected={unexpected}"
        )

    for path, content in wheel_files.items():
        digest, size = entries[path]
        if path == record_path:
            if digest:
                raise DistributionVerificationError(
                    f"{label} RECORD must leave its own hash empty"
                )
            expected_size = str(len(content))
            if size not in {"", expected_size}:
                raise DistributionVerificationError(
                    f"{label} RECORD size drift for {path!r}: "
                    f"expected empty or {expected_size!r}, actual={size!r}"
                )
            continue

        algorithm, separator, encoded = digest.partition("=")
        allowed_algorithms = hashlib.algorithms_guaranteed - {"md5", "sha1"}
        if not separator or algorithm not in allowed_algorithms:
            raise DistributionVerificationError(
                f"{label} RECORD has unsupported hash algorithm for {path!r}: "
                f"{algorithm or '<missing>'!r}"
            )
        if algorithm.startswith("shake_"):
            try:
                padding = b"=" * (-len(encoded) % 4)
                decoded = base64.b64decode(
                    encoded.encode("ascii") + padding,
                    altchars=b"-_",
                    validate=True,
                )
            except (UnicodeError, ValueError, binascii.Error) as error:
                raise DistributionVerificationError(
                    f"{label} RECORD has invalid digest for {path!r}"
                ) from error
            if not decoded:
                raise DistributionVerificationError(
                    f"{label} RECORD has invalid digest for {path!r}"
                )
            if algorithm == "shake_128":
                raw_digest = hashlib.shake_128(content).digest(len(decoded))
            else:
                raw_digest = hashlib.shake_256(content).digest(len(decoded))
        else:
            raw_digest = hashlib.new(algorithm, content).digest()
        expected_encoded = base64.urlsafe_b64encode(raw_digest).rstrip(b"=").decode()
        expected_digest = f"{algorithm}={expected_encoded}"
        expected_size = str(len(content))
        if digest != expected_digest:
            raise DistributionVerificationError(
                f"{label} RECORD integrity drift for {path!r}: "
                f"expected={expected_digest!r}, actual={digest!r}"
            )
        if size not in {"", expected_size}:
            raise DistributionVerificationError(
                f"{label} RECORD size drift for {path!r}: "
                f"expected empty or {expected_size!r}, actual={size!r}"
            )


def _verify_sdist_metadata(
    sdist_files: Mapping[str, bytes],
    root: Path,
    project: Mapping[str, object],
) -> None:
    distribution = _distribution_component(str(project["name"]))
    for path in ("PKG-INFO", f"{distribution}.egg-info/PKG-INFO"):
        _verify_core_metadata(sdist_files[path], root, project, label=f"sdist {path}")


def _wheel_rebuild_payload(files: Mapping[str, bytes]) -> dict[str, bytes]:
    return {
        path: content
        for path, content in files.items()
        if not path.endswith(".dist-info/RECORD")
    }


def _verify_entry_points(root: Path, wheel_files: Mapping[str, bytes], label: str) -> None:
    entry_files = [path for path in wheel_files if path.endswith(".dist-info/entry_points.txt")]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = _expected_entry_points(pyproject["project"])
    expected_count = 1 if expected else 0
    if len(entry_files) != expected_count:
        raise DistributionVerificationError(
            f"{label} entry_points.txt inventory drift: "
            f"expected={expected_count}, actual={entry_files}"
        )
    if not expected:
        return

    class CaseSensitiveConfigParser(configparser.ConfigParser):
        def optionxform(self, optionstr: str) -> str:
            return optionstr

    parser = CaseSensitiveConfigParser(
        delimiters=("=",),
        interpolation=None,
        strict=True,
    )
    try:
        parser.read_string(wheel_files[entry_files[0]].decode("utf-8"))
    except (UnicodeError, configparser.Error) as error:
        raise DistributionVerificationError(
            f"{label} entry_points.txt is invalid"
        ) from error
    if parser.defaults():
        raise DistributionVerificationError(
            f"{label} entry_points.txt must not define DEFAULT entries"
        )
    actual = {
        section: dict(parser.items(section, raw=True))
        for section in parser.sections()
    }
    if actual != expected:
        raise DistributionVerificationError(
            f"{label} entry-point drift: expected={expected}, actual={actual}"
        )


def _github_anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", markdown, re.MULTILINE):
        plain = re.sub(r"<[^>]+>", "", heading)
        plain = plain.replace("`", "").strip().lower()
        slug = re.sub(r"[^\w\- ]", "", plain)
        slug = re.sub(r"\s", "-", slug)
        duplicate = counts.get(slug, 0)
        counts[slug] = duplicate + 1
        anchors.add(slug if duplicate == 0 else f"{slug}-{duplicate}")
    return anchors


def _verify_markdown_links(root: Path, packaged_paths: set[str]) -> None:
    markdown_paths = sorted(
        [root / "README.md", *root.glob("docs/**/*.md")],
        key=lambda path: path.as_posix(),
    )
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = raw_target.strip()
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")]
            else:
                target = target.split(maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            relative_target = unquote(parsed.path)
            if relative_target:
                resolved = (markdown_path.parent / relative_target).resolve()
                try:
                    packaged_target = resolved.relative_to(root.resolve()).as_posix()
                except ValueError as error:
                    raise DistributionVerificationError(
                        f"{markdown_path.relative_to(root)} link escapes the sdist: {target}"
                    ) from error
            else:
                resolved = markdown_path
                packaged_target = markdown_path.relative_to(root).as_posix()
            target_is_packaged_directory = any(
                path.startswith(f"{packaged_target.rstrip('/')}/") for path in packaged_paths
            )
            if packaged_target not in packaged_paths and not target_is_packaged_directory:
                raise DistributionVerificationError(
                    f"{markdown_path.relative_to(root)} has unpackaged local link: {target}"
                )
            if parsed.fragment and resolved.suffix.lower() == ".md":
                anchors = _github_anchors(resolved.read_text(encoding="utf-8"))
                fragment = unquote(parsed.fragment).lower()
                if fragment not in anchors:
                    raise DistributionVerificationError(
                        f"{markdown_path.relative_to(root)} has missing anchor: {target}"
                    )


def verify_distributions(
    source_root: Path,
    wheel: Path,
    sdist: Path,
    rebuilt_wheel: Path | None = None,
) -> None:
    root = source_root.resolve()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    package_files = _expected_package_files(root)
    documentation = _source_files(root, "docs/**/*.md")
    notebooks = _source_files(root, "notebooks/**/*.ipynb")
    top_level = {
        name: (root / name).read_bytes()
        for name in (
            "CHANGELOG.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "MANIFEST.in",
            "PRODUCTION_READINESS.md",
            "README.md",
            "RELEASING.md",
            "SECURITY.md",
            "pyproject.toml",
        )
    }
    sdist_expected = {
        **top_level,
        **documentation,
        **notebooks,
        **package_files,
        **_source_files(root, "references/**/*.json"),
        **_source_files(root, "scripts/**/*.py"),
        **_source_files(root, "tests/**/*.py"),
    }

    _verify_wheel_filename(wheel, project, label="wheel")
    _verify_sdist_filename(sdist, project)
    dist_info, copied_wheel_metadata, generated_wheel_metadata = _wheel_metadata(root, project)
    sdist_root, generated_sdist_metadata = _sdist_metadata(project)
    wheel_files, wheel_directories = _read_wheel(wheel)
    sdist_files, sdist_directories = _read_sdist(sdist, sdist_root)
    _require_exact_archive(
        "wheel",
        wheel_files,
        {**package_files, **copied_wheel_metadata},
        generated_wheel_metadata,
        wheel_directories,
    )
    _require_exact_archive(
        "sdist",
        sdist_files,
        sdist_expected,
        generated_sdist_metadata,
        sdist_directories,
    )
    _verify_wheel_metadata(wheel_files, dist_info, root, project, label="wheel")
    _verify_record(wheel_files, dist_info, label="wheel")
    _verify_sdist_metadata(sdist_files, root, project)
    _verify_entry_points(root, wheel_files, "wheel")
    _verify_markdown_links(root, set(sdist_files))

    if rebuilt_wheel is not None:
        if rebuilt_wheel.resolve() == wheel.resolve():
            raise DistributionVerificationError(
                "sdist-rebuilt wheel must be a distinct artifact from the direct wheel"
            )
        _verify_wheel_filename(rebuilt_wheel, project, label="sdist-rebuilt wheel")
        rebuilt_files, rebuilt_directories = _read_wheel(rebuilt_wheel)
        _require_exact_archive(
            "sdist-rebuilt wheel",
            rebuilt_files,
            {**package_files, **copied_wheel_metadata},
            generated_wheel_metadata,
            rebuilt_directories,
        )
        _verify_wheel_metadata(
            rebuilt_files,
            dist_info,
            root,
            project,
            label="sdist-rebuilt wheel",
        )
        _verify_record(rebuilt_files, dist_info, label="sdist-rebuilt wheel")
        _verify_entry_points(root, rebuilt_files, "sdist-rebuilt wheel")
        direct_payload = _wheel_rebuild_payload(wheel_files)
        rebuilt_payload = _wheel_rebuild_payload(rebuilt_files)
        missing = sorted(set(direct_payload) - set(rebuilt_payload))
        unexpected = sorted(set(rebuilt_payload) - set(direct_payload))
        changed = sorted(
            path
            for path in direct_payload.keys() & rebuilt_payload.keys()
            if direct_payload[path] != rebuilt_payload[path]
        )
        if missing or unexpected or changed:
            raise DistributionVerificationError(
                "sdist-rebuilt wheel payload differs: "
                f"missing={missing}, unexpected={unexpected}, changed={changed}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify wheel/sdist contents against the release checkout."
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--rebuilt-wheel", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        verify_distributions(args.source_root, args.wheel, args.sdist, args.rebuilt_wheel)
    except (DistributionVerificationError, OSError, tarfile.TarError) as error:
        raise SystemExit(f"distribution verification failed: {error}") from error
    print("distribution artifacts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
