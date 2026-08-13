from __future__ import annotations

import base64
import csv
import hashlib
import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scripts.verify_distribution_artifacts import (
    DistributionVerificationError,
    verify_distributions,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


TOP_LEVEL = (
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "PRODUCTION_READINESS.md",
    "README.md",
    "RELEASING.md",
    "SECURITY.md",
)

DEFAULT_ENTRY_POINTS_TOML = '''[project.scripts]
fixture = "nano_deepseek_v4:main"
'''


def _write_source(
    root: Path,
    *,
    version: str = "1.0.0",
    entry_points_toml: str = DEFAULT_ENTRY_POINTS_TOML,
) -> None:
    root.mkdir()
    for name in TOP_LEVEL:
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Fixture\n\n[Guide](docs/guide.md#details)\n[Docs](docs)\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        f'''[project]
name = "fixture"
version = "{version}"
description = "A distribution verification fixture."
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
license-files = ["LICENSE"]
authors = [
    {{ name = "Fixture Contributors" }},
    {{ name = "Alice Example", email = "alice@example.com" }},
    {{ email = "ops@example.com" }},
]
keywords = ["fixture", "packaging"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]
dependencies = [
    "dep-one>=1.2",
    "platform-dep; python_version < '3.13'",
]

[project.optional-dependencies]
test = [
    "pytest>=7",
    "colorama; sys_platform == 'win32'",
]

{entry_points_toml}

[project.urls]
Homepage = "https://example.invalid/fixture"
Repository = "https://example.invalid/fixture/repository"
''',
        encoding="utf-8",
    )
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# Details\n", encoding="utf-8")
    (root / "nano_deepseek_v4").mkdir()
    (root / "nano_deepseek_v4" / "__init__.py").write_text(
        "def main():\n    return 0\n", encoding="utf-8"
    )
    (root / "nano_deepseek_v4" / "py.typed").write_bytes(b"")
    (root / "references").mkdir()
    (root / "references" / "receipt.json").write_text("{}\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "helper.py").write_text("pass\n", encoding="utf-8")


def _project(root: Path) -> dict[str, object]:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]


def _version(root: Path) -> str:
    return str(Version(str(_project(root)["version"])))


def _wheel_name(root: Path) -> str:
    return f"fixture-{_version(root)}-py3-none-any.whl"


def _sdist_name(root: Path) -> str:
    return f"fixture-{_version(root)}.tar.gz"


def _requirement_without_marker(requirement: Requirement) -> str:
    value = requirement.name
    if requirement.extras:
        value += f"[{','.join(sorted(requirement.extras))}]"
    if requirement.url:
        return f"{value} @ {requirement.url}"
    return value + str(requirement.specifier)


def _metadata(
    root: Path,
    *,
    body: bytes | None = None,
    **overrides: str | list[str],
) -> bytes:
    project = _project(root)
    authors = project.get("authors", [])
    assert isinstance(authors, list)
    author_names: list[str] = []
    author_emails: list[str] = []
    for author in authors:
        assert isinstance(author, dict)
        name = author.get("name")
        email = author.get("email")
        if email is None:
            author_names.append(str(name))
        elif name is None:
            author_emails.append(str(email))
        else:
            author_emails.append(f"{name} <{email}>")
    urls = project.get("urls", {})
    assert isinstance(urls, dict)
    keywords = project.get("keywords", [])
    classifiers = project.get("classifiers", [])
    assert isinstance(keywords, list)
    assert isinstance(classifiers, list)
    fields: dict[str, str | list[str]] = {
        "Metadata-Version": "2.4",
        "Name": str(project["name"]),
        "Version": _version(root),
        "Summary": str(project["description"]),
        "Author": ", ".join(author_names),
        "Author-email": ", ".join(author_emails),
        "License-Expression": str(project["license"]),
        "Project-URL": [f"{name}, {url}" for name, url in urls.items()],
        "Keywords": ",".join(str(value) for value in keywords),
        "Classifier": [str(value) for value in classifiers],
        "Requires-Python": str(project["requires-python"]),
        "Description-Content-Type": "text/markdown",
        "License-File": ["LICENSE"],
        "Dynamic": ["license-file"],
    }
    dependencies = project.get("dependencies", [])
    assert isinstance(dependencies, list)
    requirements = [str(value) for value in dependencies]
    extras: list[str] = []
    optional = project.get("optional-dependencies", {})
    assert isinstance(optional, dict)
    for raw_extra, values in optional.items():
        extra = canonicalize_name(str(raw_extra))
        extras.append(extra)
        assert isinstance(values, list)
        for value in values:
            requirement = Requirement(str(value))
            marker = f'extra == "{extra}"'
            if requirement.marker is not None:
                marker = f"({requirement.marker}) and {marker}"
            requirements.append(f"{_requirement_without_marker(requirement)}; {marker}")
    fields["Requires-Dist"] = requirements
    fields["Provides-Extra"] = extras
    fields.update(overrides)

    lines: list[str] = []
    for name, value in fields.items():
        values = value if isinstance(value, list) else [value]
        lines.extend(f"{name}: {item}" for item in values)
    description = (root / "README.md").read_bytes() if body is None else body
    return ("\n".join(lines) + "\n\n").encode() + description


def _entry_points(root: Path) -> bytes | None:
    project = _project(root)
    groups: dict[str, dict[str, str]] = {}
    for project_key, group in (("scripts", "console_scripts"), ("gui-scripts", "gui_scripts")):
        entries = project.get(project_key, {})
        assert isinstance(entries, dict)
        if entries:
            groups[group] = {str(name): str(value) for name, value in entries.items()}
    custom_groups = project.get("entry-points", {})
    assert isinstance(custom_groups, dict)
    for group, entries in custom_groups.items():
        assert isinstance(entries, dict)
        if entries:
            groups[str(group)] = {str(name): str(value) for name, value in entries.items()}
    if not groups:
        return None
    sections = [
        "\n".join([f"[{group}]", *(f"{name} = {value}" for name, value in entries.items())])
        for group, entries in groups.items()
    ]
    return ("\n\n".join(sections) + "\n").encode()


def _wheel_headers(**overrides: str | list[str]) -> bytes:
    fields: dict[str, str | list[str]] = {
        "Wheel-Version": "1.0",
        "Generator": "fixture-builder",
        "Root-Is-Purelib": "true",
        "Tag": "py3-none-any",
    }
    fields.update(overrides)
    lines: list[str] = []
    for name, value in fields.items():
        values = value if isinstance(value, list) else [value]
        lines.extend(f"{name}: {item}" for item in values)
    return ("\n".join(lines) + "\n").encode()


def _record(
    entries: dict[str, bytes],
    *,
    algorithm: str = "sha256",
    include_sizes: bool = True,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for member, content in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.new(algorithm, content).digest())
        encoded = digest.rstrip(b"=").decode()
        size = str(len(content)) if include_sizes else ""
        writer.writerow((member, f"{algorithm}={encoded}", size))
    record_path = next(member for member in entries if member.endswith(".dist-info/RECORD"))
    # RECORD is initially a placeholder so that its path participates in archive construction.
    rows = output.getvalue().splitlines()
    rows = [row for row in rows if not row.startswith(f"{record_path},")]
    rows.append(f"{record_path},,")
    return ("\n".join(rows) + "\n").encode()


def _build_wheel(
    root: Path,
    path: Path,
    *,
    changed_package: bool = False,
    extra_files: dict[str, bytes] | None = None,
    omit: set[str] | None = None,
    metadata_overrides: dict[str, str | list[str]] | None = None,
    metadata_body: bytes | None = None,
    wheel_overrides: dict[str, str | list[str]] | None = None,
    mutate_record: Callable[[bytes], bytes] | None = None,
    record_algorithm: str = "sha256",
    record_sizes: bool = True,
) -> None:
    omitted = omit or set()
    version = _version(root)
    dist_info = f"fixture-{version}.dist-info"
    entries: dict[str, bytes] = {}
    package = root / "nano_deepseek_v4"
    for source in sorted(package.rglob("*")):
        if source.is_file():
            content = source.read_bytes()
            if changed_package and source.name == "__init__.py":
                content = b"changed\n"
            entries[source.relative_to(root).as_posix()] = content
    entries.update(
        {
            f"{dist_info}/METADATA": _metadata(
                root,
                body=metadata_body,
                **(metadata_overrides or {}),
            ),
            f"{dist_info}/RECORD": b"",
            f"{dist_info}/WHEEL": _wheel_headers(**(wheel_overrides or {})),
            f"{dist_info}/licenses/LICENSE": (root / "LICENSE").read_bytes(),
            f"{dist_info}/top_level.txt": b"nano_deepseek_v4\n",
        }
    )
    entry_points = _entry_points(root)
    if entry_points is not None:
        entries[f"{dist_info}/entry_points.txt"] = entry_points
    entries.update(extra_files or {})
    entries = {member: content for member, content in entries.items() if member not in omitted}
    record_path = f"{dist_info}/RECORD"
    if record_path in entries:
        record = _record(
            entries,
            algorithm=record_algorithm,
            include_sizes=record_sizes,
        )
        entries[record_path] = mutate_record(record) if mutate_record else record
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for member, content in entries.items():
            archive.writestr(member, content)


def _add_tar_bytes(archive: tarfile.TarFile, member: str, content: bytes) -> None:
    info = tarfile.TarInfo(member)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _requires_txt(root: Path) -> bytes:
    project = _project(root)
    dependencies = project.get("dependencies", [])
    assert isinstance(dependencies, list)
    lines = [str(value) for value in dependencies]
    optional = project.get("optional-dependencies", {})
    assert isinstance(optional, dict)
    for extra, values in optional.items():
        lines.extend(("", f"[{extra}]"))
        assert isinstance(values, list)
        lines.extend(str(value) for value in values)
    return ("\n".join(lines) + "\n").encode()


def _build_sdist(
    root: Path,
    path: Path,
    *,
    extra_files: dict[str, bytes] | None = None,
    extra_directories: set[str] | None = None,
    omit: set[str] | None = None,
    metadata_overrides: dict[str, str | list[str]] | None = None,
    metadata_body: bytes | None = None,
    compression: bool = True,
) -> None:
    archive_root = f"fixture-{_version(root)}"
    omitted = omit or set()
    mode: Literal["w:gz", "w"] = "w:gz" if compression else "w"
    with tarfile.open(path, mode) as archive:
        archive.add(root, arcname=archive_root)
        generated = {
            "PKG-INFO": _metadata(
                root,
                body=metadata_body,
                **(metadata_overrides or {}),
            ),
            "setup.cfg": b"[egg_info]\ntag_build =\n",
            "fixture.egg-info/PKG-INFO": _metadata(
                root,
                body=metadata_body,
                **(metadata_overrides or {}),
            ),
            "fixture.egg-info/SOURCES.txt": b"pyproject.toml\n",
            "fixture.egg-info/dependency_links.txt": b"\n",
            "fixture.egg-info/requires.txt": _requires_txt(root),
            "fixture.egg-info/top_level.txt": b"nano_deepseek_v4\n",
        }
        entry_points = _entry_points(root)
        if entry_points is not None:
            generated["fixture.egg-info/entry_points.txt"] = entry_points
        for relative, content in {**generated, **(extra_files or {})}.items():
            if relative not in omitted:
                _add_tar_bytes(archive, f"{archive_root}/{relative}", content)
        for relative in extra_directories or set():
            info = tarfile.TarInfo(f"{archive_root}/{relative}")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)


def _artifacts(
    tmp_path: Path,
    *,
    version: str = "1.0.0",
    entry_points_toml: str = DEFAULT_ENTRY_POINTS_TOML,
) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "source"
    _write_source(root, version=version, entry_points_toml=entry_points_toml)
    wheel = tmp_path / "direct" / _wheel_name(root)
    rebuilt = tmp_path / "rebuilt" / _wheel_name(root)
    sdist = tmp_path / _sdist_name(root)
    return root, wheel, sdist, rebuilt


def test_distribution_verifier_accepts_matching_wheel_and_sdist(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_keeps_notebook_in_sdist_only(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path)
    notebooks = root / "notebooks"
    notebooks.mkdir()
    notebook = notebooks / "tutorial.ipynb"
    notebook.write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n',
        encoding="utf-8",
    )
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)

    with ZipFile(wheel) as archive:
        assert not any(name.endswith(".ipynb") for name in archive.namelist())
    with tarfile.open(sdist, "r:gz") as archive:
        member = archive.extractfile("fixture-1.0.0/notebooks/tutorial.ipynb")
        assert member is not None
        assert member.read() == notebook.read_bytes()


def test_project_manifest_includes_notebook_sources():
    root = Path(__file__).parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "recursive-include notebooks *.ipynb" in manifest


@pytest.mark.parametrize(
    ("source_version", "normalized_version"),
    [
        ("1.0-rc1", "1.0rc1"),
        ("1!2.0", "1!2.0"),
        ("1.0+LOCAL-1", "1.0+local.1"),
        ("1!2.0-rc1+LOCAL-1", "1!2.0rc1+local.1"),
    ],
)
def test_distribution_verifier_accepts_pep440_normalized_version(
    tmp_path: Path,
    source_version: str,
    normalized_version: str,
):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path, version=source_version)
    assert wheel.name == f"fixture-{normalized_version}-py3-none-any.whl"
    assert sdist.name == f"fixture-{normalized_version}.tar.gz"
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_accepts_no_entry_points(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path, entry_points_toml="")
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_preserves_mixed_case_entry_point_names(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(
        tmp_path,
        entry_points_toml='''[project.scripts]
FixtureTool = "nano_deepseek_v4:main"
''',
    )
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_accepts_gui_and_custom_entry_point_groups(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(
        tmp_path,
        entry_points_toml='''[project.gui-scripts]
FixtureGUI = "nano_deepseek_v4:main"

[project.entry-points."fixture.plugins"]
FixturePlugin = "nano_deepseek_v4"
''',
    )
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_rejects_unexpected_entry_point_group(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(
        root,
        wheel,
        extra_files={
            "fixture-1.0.0.dist-info/entry_points.txt": (
                b"[console_scripts]\nfixture = nano_deepseek_v4:main\n\n"
                b"[fixture.plugins]\nextra = nano_deepseek_v4\n"
            )
        },
    )
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="entry-point drift"):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize(
    "entry_points",
    [
        b"\xff",
        b"[console_scripts]\nfixture: nano_deepseek_v4:main\n",
    ],
)
def test_distribution_verifier_wraps_invalid_entry_point_metadata(
    tmp_path: Path,
    entry_points: bytes,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(
        root,
        wheel,
        extra_files={"fixture-1.0.0.dist-info/entry_points.txt": entry_points},
    )
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="entry_points.txt is invalid"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_requires_distinct_rebuilt_wheel(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="must be a distinct artifact"):
        verify_distributions(root, wheel, sdist, rebuilt_wheel=wheel)


def test_distribution_verifier_rejects_changed_rebuilt_wheel(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_wheel(root, rebuilt, changed_package=True)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="sdist-rebuilt wheel payload drift"):
        verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_rejects_changed_wheel_payload(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, changed_package=True)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="wheel payload drift"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_broken_packaged_markdown_link(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    (root / "README.md").write_text("# Fixture\n\n[Missing](docs/missing.md)\n")
    _build_wheel(root, wheel)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="unpackaged local link"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_unexpected_wheel_member(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, extra_files={"sitecustomize.pth": b"import payload\n"})
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match=r"unexpected=.*sitecustomize\.pth"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_missing_wheel_metadata(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, omit={"fixture-1.0.0.dist-info/METADATA"})
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match=r"missing=.*METADATA"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_unexpected_sdist_file(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist, extra_files={"payload.bin": b"unexpected"})

    with pytest.raises(DistributionVerificationError, match=r"unexpected=.*payload\.bin"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_unexpected_sdist_directory(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist, extra_directories={"empty-payload"})

    with pytest.raises(
        DistributionVerificationError,
        match=r"unexpected_directories=.*empty-payload",
    ):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_missing_sdist_generated_metadata(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist, omit={"fixture.egg-info/SOURCES.txt"})

    with pytest.raises(DistributionVerificationError, match=r"missing=.*SOURCES\.txt"):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize(
    ("bad_name", "label"),
    [
        ("fixture-1.0.0-py2-none-any.whl", "wheel filename drift"),
        ("Fixture-1.0.0-py3-none-any.whl", "wheel filename drift"),
        ("fixture-1.0-py3-none-any.whl", "wheel filename drift"),
    ],
)
def test_distribution_verifier_rejects_noncanonical_wheel_filename(
    tmp_path: Path,
    bad_name: str,
    label: str,
):
    root, _wheel, sdist, _rebuilt = _artifacts(tmp_path)
    wheel = tmp_path / bad_name
    _build_wheel(root, wheel)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match=label):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize("bad_name", ["fixture-1.0.tar.gz", "Fixture-1.0.0.tar.gz"])
def test_distribution_verifier_rejects_noncanonical_sdist_filename(
    tmp_path: Path,
    bad_name: str,
):
    root, wheel, _sdist, _rebuilt = _artifacts(tmp_path)
    sdist = tmp_path / bad_name
    _build_wheel(root, wheel)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="sdist filename drift"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_uncompressed_sdist(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist, compression=False)

    with pytest.raises(DistributionVerificationError, match="not gzip-compressed"):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize(
    "member",
    ["nano_deepseek_v4//alias.py", "nano_deepseek_v4/./alias.py", "dir\\alias.py"],
)
def test_distribution_verifier_rejects_noncanonical_wheel_member(
    tmp_path: Path,
    member: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, extra_files={member: b"alias\n"})
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="non-canonical member"):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize("member", ["docs//alias.md", "docs/./alias.md", "docs\\alias.md"])
def test_distribution_verifier_rejects_noncanonical_sdist_member(
    tmp_path: Path,
    member: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist, extra_files={member: b"alias\n"})

    with pytest.raises(DistributionVerificationError, match="non-canonical member"):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"Metadata-Version": "2.3"}, "Metadata-Version drift"),
        ({"Name": "other"}, "name drift"),
        ({"Version": "2.0"}, "version drift"),
        ({"Summary": "A different summary."}, "Summary drift"),
        ({"Author": "Another Author"}, "Author drift"),
        ({"Author-email": "other@example.com"}, "Author-email drift"),
        ({"License-Expression": "MIT"}, "License-Expression drift"),
        ({"License": "Apache License"}, "unexpected License header"),
        ({"License-File": ["LICENSE", "NOTICE"]}, "License-File inventory drift"),
        ({"Keywords": "fixture,other"}, "Keywords drift"),
        (
            {
                "Classifier": [
                    "Operating System :: OS Independent",
                    "Programming Language :: Python :: 3",
                ]
            },
            "Classifier drift",
        ),
        (
            {"Project-URL": ["Homepage, https://example.invalid/other"]},
            "Project-URL drift",
        ),
        ({"Description-Content-Type": "text/plain"}, "Description-Content-Type drift"),
        ({"Dynamic": []}, "Dynamic drift"),
        ({"Dynamic": ["license-file", "summary"]}, "Dynamic drift"),
        ({"Requires-Python": ">=3.12"}, "Requires-Python drift"),
        ({"Requires-Dist": ["dep-one>=1.2"]}, "dependency drift"),
        ({"Provides-Extra": ["other"]}, "extras drift"),
    ],
)
def test_distribution_verifier_rejects_poisoned_wheel_metadata(
    tmp_path: Path,
    overrides: dict[str, str | list[str]],
    message: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, metadata_overrides=overrides)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match=message):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_changed_readme_metadata_body(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, metadata_body=b"# Substituted project\n")
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="README description body drift"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_poisoned_sdist_pkg_info(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(root, sdist, metadata_overrides={"Requires-Dist": ["malware"]})

    with pytest.raises(DistributionVerificationError, match="sdist PKG-INFO dependency drift"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_checks_sdist_egg_info_metadata(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel)
    _build_sdist(
        root,
        sdist,
        extra_files={
            "fixture.egg-info/PKG-INFO": _metadata(root, Summary="A different summary.")
        },
    )

    with pytest.raises(
        DistributionVerificationError,
        match=r"sdist fixture\.egg-info/PKG-INFO Summary drift",
    ):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"Wheel-Version": "2.0"}, "Wheel-Version drift"),
        ({"Root-Is-Purelib": "false"}, "Root-Is-Purelib drift"),
        ({"Tag": "cp313-cp313-manylinux_2_28_x86_64"}, "tag drift"),
        ({"Tag": ["py3-none-any", "py2-none-any"]}, "tag drift"),
    ],
)
def test_distribution_verifier_rejects_poisoned_wheel_headers(
    tmp_path: Path,
    overrides: dict[str, str | list[str]],
    message: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, wheel_overrides=overrides)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match=message):
        verify_distributions(root, wheel, sdist)


def _replace_record_line(record: bytes, suffix: str, replacement: str) -> bytes:
    lines = record.decode().splitlines()
    for index, line in enumerate(lines):
        if line.split(",", 1)[0].endswith(suffix):
            lines[index] = replacement
            break
    return ("\n".join(lines) + "\n").encode()


def _replace_record_size(record: bytes, suffix: str, size: str) -> bytes:
    source = io.StringIO(record.decode(), newline="")
    rows = list(csv.reader(source))
    for row in rows:
        if row[0].endswith(suffix):
            row[2] = size
            break
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode()


def test_distribution_verifier_accepts_sha512_record(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, record_algorithm="sha512")
    _build_wheel(root, rebuilt, record_algorithm="sha512")
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_accepts_missing_record_sizes(tmp_path: Path):
    root, wheel, sdist, rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, record_sizes=False)
    _build_wheel(root, rebuilt, record_sizes=False)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist, rebuilt_wheel=rebuilt)


def test_distribution_verifier_accepts_exact_record_self_size(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)

    def add_self_size(record: bytes) -> bytes:
        size = str(len(record))
        for _ in range(3):
            updated = _replace_record_size(record, ".dist-info/RECORD", size)
            next_size = str(len(updated))
            if next_size == size:
                return updated
            size = next_size
        raise AssertionError("RECORD size did not stabilize")

    _build_wheel(root, wheel, mutate_record=add_self_size)
    _build_sdist(root, sdist)

    verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda record: _replace_record_line(
                record,
                "__init__.py",
                "nano_deepseek_v4/__init__.py,sha256=AAAA,27",
            ),
            "RECORD integrity drift",
        ),
        (
            lambda record: b"\n".join(record.splitlines()[1:]) + b"\n",
            "RECORD membership drift",
        ),
        (
            lambda record: record + b"ghost.py,sha256=AAAA,1\n",
            "RECORD membership drift",
        ),
        (
            lambda record: record + record.splitlines()[0] + b"\n",
            "RECORD has duplicate member",
        ),
        (
            lambda record: record.replace(
                b"nano_deepseek_v4/__init__.py,",
                b"nano_deepseek_v4//__init__.py,",
            ),
            "non-canonical member",
        ),
    ],
)
def test_distribution_verifier_rejects_invalid_record(
    tmp_path: Path,
    mutator: Callable[[bytes], bytes],
    message: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    _build_wheel(root, wheel, mutate_record=mutator)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match=message):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize("algorithm", ["md5", "sha1", "not-a-hash"])
def test_distribution_verifier_rejects_weak_or_unknown_record_hash(
    tmp_path: Path,
    algorithm: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    content = (root / "nano_deepseek_v4" / "__init__.py").read_bytes()
    if algorithm in hashlib.algorithms_available:
        raw_digest = hashlib.new(algorithm, content).digest()
        encoded = base64.urlsafe_b64encode(raw_digest).rstrip(b"=").decode()
    else:
        encoded = "AAAA"

    def replace_hash(record: bytes) -> bytes:
        return _replace_record_line(
            record,
            "__init__.py",
            f"nano_deepseek_v4/__init__.py,{algorithm}={encoded},{len(content)}",
        )

    _build_wheel(root, wheel, mutate_record=replace_hash)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="unsupported hash algorithm"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_requires_record_hash_for_every_file(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    content = (root / "nano_deepseek_v4" / "__init__.py").read_bytes()

    def remove_hash(record: bytes) -> bytes:
        return _replace_record_line(
            record,
            "__init__.py",
            f"nano_deepseek_v4/__init__.py,,{len(content)}",
        )

    _build_wheel(root, wheel, mutate_record=remove_hash)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="unsupported hash algorithm"):
        verify_distributions(root, wheel, sdist)


@pytest.mark.parametrize("size_kind", ["noncanonical", "wrong"])
def test_distribution_verifier_rejects_invalid_record_size(
    tmp_path: Path,
    size_kind: str,
):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)
    content = (root / "nano_deepseek_v4" / "__init__.py").read_bytes()
    size = f"0{len(content)}" if size_kind == "noncanonical" else str(len(content) + 1)

    def replace_size(record: bytes) -> bytes:
        return _replace_record_size(record, "__init__.py", size)

    _build_wheel(root, wheel, mutate_record=replace_size)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="RECORD size drift"):
        verify_distributions(root, wheel, sdist)


def test_distribution_verifier_rejects_hashed_record_self_entry(tmp_path: Path):
    root, wheel, sdist, _rebuilt = _artifacts(tmp_path)

    def hash_self(record: bytes) -> bytes:
        record_path = "fixture-1.0.0.dist-info/RECORD"
        return record.replace(f"{record_path},,\n".encode(), f"{record_path},sha256=AAAA,1\n".encode())

    _build_wheel(root, wheel, mutate_record=hash_self)
    _build_sdist(root, sdist)

    with pytest.raises(DistributionVerificationError, match="must leave its own hash"):
        verify_distributions(root, wheel, sdist)
