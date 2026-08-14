from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_release_source import (
    RELEASE_DOCUMENTS,
    TRANSITIONAL_FRAGMENTS,
    ReleaseSourceError,
    main,
    verify_release_source,
)

VERSION = "1.2.3"
PREVIOUS_VERSION = "1.2.2"


def _write_release_tree(root: Path) -> None:
    (root / "nano_deepseek_v4").mkdir(parents=True)
    (root / "docs" / "guides").mkdir(parents=True)
    (root / "docs" / "verification").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "nano-deepseek-v4"\nversion = "{VERSION}"\n\n'
        "[project.optional-dependencies]\n"
        "official = []\n"
        "parity = []\n"
        "dev = []\n"
        "notebook = []\n",
        encoding="utf-8",
    )
    (root / "nano_deepseek_v4" / "__init__.py").write_text(
        f'__version__ = "{VERSION}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\ntitle: fixture\nversion: {VERSION}\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# fixture\n\n"
        f"| Published PyPI `{VERSION}` | released commands |\n\n"
        f'python -m pip install "nano-deepseek-v4=={VERSION}"\n\n'
        f"nano-deepseek-v4 {VERSION}\n",
        encoding="utf-8",
    )
    (root / "docs" / "guides" / "installation.md").write_text(
        "# Installation\n\n"
        f"| `official` | PyPI {VERSION} and current source | Hub tools |\n"
        f"| `parity` | PyPI {VERSION} and current source | Parity tools |\n"
        f"| `dev` | PyPI {VERSION} and current source | Development tools |\n"
        f"| `notebook` | PyPI {VERSION} and current source | Jupyter |\n\n"
        f'python -m pip install "nano-deepseek-v4[official,parity]=={VERSION}"\n',
        encoding="utf-8",
    )
    for relative in (
        "docs/guides/official-checkpoints.md",
        "docs/guides/train-and-generate.md",
        "docs/verification/conformance.md",
        "docs/verification/dspark.md",
    ):
        (root / relative).write_text(f"# {Path(relative).stem}\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        f"## [{VERSION}] - 2026-08-12\n\n"
        "### Added\n\n- Current release.\n\n"
        f"## [{PREVIOUS_VERSION}] - 2026-07-31\n\n"
        "### Added\n\n- Previous release.\n\n"
        f"[Unreleased]: https://github.com/hebo1221/nano-deepseek-v4/compare/v{VERSION}...HEAD\n"
        f"[{VERSION}]: https://github.com/hebo1221/nano-deepseek-v4/compare/"
        f"v{PREVIOUS_VERSION}...v{VERSION}\n"
        f"[{PREVIOUS_VERSION}]: https://github.com/hebo1221/nano-deepseek-v4/"
        f"tree/v{PREVIOUS_VERSION}\n",
        encoding="utf-8",
    )


def test_release_source_accepts_consistent_release_tree(tmp_path: Path):
    _write_release_tree(tmp_path)

    assert verify_release_source(tmp_path, f"v{VERSION}") == VERSION


def test_release_source_rejects_wrong_tag(tmp_path: Path):
    _write_release_tree(tmp_path)

    with pytest.raises(ReleaseSourceError, match="release tag"):
        verify_release_source(tmp_path, "v9.9.9")


def test_release_source_rejects_wrong_project_identity(tmp_path: Path):
    _write_release_tree(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            'name = "nano-deepseek-v4"',
            'name = "nano-deepseek-v4-typo"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="project.name"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_duplicate_init_version(tmp_path: Path):
    _write_release_tree(tmp_path)
    init = tmp_path / "nano_deepseek_v4" / "__init__.py"
    init.write_text(init.read_text() + '__version__ = "9.9.9"\n', encoding="utf-8")

    with pytest.raises(ReleaseSourceError, match="one top-level string"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize(
    "additional_source",
    [
        'if True:\n    __version__ = "9.9.9"\n',
        'del __version__\n',
        'from package import value as __version__\n',
    ],
)
def test_release_source_rejects_runtime_version_rebinding(
    tmp_path: Path,
    additional_source: str,
):
    _write_release_tree(tmp_path)
    init = tmp_path / "nano_deepseek_v4" / "__init__.py"
    init.write_text(init.read_text() + additional_source, encoding="utf-8")

    with pytest.raises(ReleaseSourceError, match="must not write __version__"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_ignores_nested_citation_version(tmp_path: Path):
    _write_release_tree(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: 9.9.9\nauthors:\n  version: {VERSION}\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="CITATION.cff version"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_duplicate_top_level_citation_version(tmp_path: Path):
    _write_release_tree(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(citation.read_text() + f"version: {VERSION}\n", encoding="utf-8")

    with pytest.raises(ReleaseSourceError, match="one top-level version"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize("quote", ["'", '"'])
def test_release_source_accepts_quoted_citation_version(tmp_path: Path, quote: str):
    _write_release_tree(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text().replace(
            f"version: {VERSION}",
            f"version: {quote}{VERSION}{quote}",
        ),
        encoding="utf-8",
    )

    assert verify_release_source(tmp_path, f"v{VERSION}") == VERSION


def test_release_source_rejects_invalid_citation_yaml(tmp_path: Path):
    _write_release_tree(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text() + "broken: [unterminated\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="CITATION.cff is invalid"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize(
    ("relative", "fragment"),
    [
        (relative, fragment)
        for relative, fragments in TRANSITIONAL_FRAGMENTS.items()
        for fragment in fragments
    ],
)
def test_release_source_rejects_every_transition_claim(
    tmp_path: Path,
    relative: str,
    fragment: str,
):
    _write_release_tree(tmp_path)
    path = tmp_path / relative
    path.write_text(path.read_text() + f"\n{fragment}\n", encoding="utf-8")

    with pytest.raises(ReleaseSourceError, match="transition claims"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize("relative", RELEASE_DOCUMENTS)
def test_release_source_rejects_stale_pin_in_every_managed_document(
    tmp_path: Path,
    relative: str,
):
    _write_release_tree(tmp_path)
    document = tmp_path / relative
    document.write_text(
        document.read_text()
        + '\npython -m pip install "nano-deepseek-v4[official]==1.2.2"\n',
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="exact package versions"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize("project_name", ["nano_deepseek_v4", "Nano.DeepSeek_V4"])
def test_release_source_rejects_stale_pin_with_equivalent_project_name(
    tmp_path: Path,
    project_name: str,
):
    _write_release_tree(tmp_path)
    document = tmp_path / "docs" / "verification" / "conformance.md"
    document.write_text(
        document.read_text()
        + f"\npython -m pip install '{project_name}=={PREVIOUS_VERSION}'\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="exact package versions"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize("extra", ["official", "parity", "dev", "notebook"])
def test_release_source_rejects_missing_optional_extra_row(
    tmp_path: Path,
    extra: str,
):
    _write_release_tree(tmp_path)
    installation = tmp_path / "docs" / "guides" / "installation.md"
    installation.write_text(
        "\n".join(
            line
            for line in installation.read_text().splitlines()
            if not line.startswith(f"| `{extra}` |")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="optional-extra rows"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_duplicate_optional_extra_row(tmp_path: Path):
    _write_release_tree(tmp_path)
    installation = tmp_path / "docs" / "guides" / "installation.md"
    installation.write_text(
        installation.read_text()
        + f"\n| `official` | PyPI {VERSION} and current source | Duplicate |\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="repeats optional-extra row"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_stale_optional_extra_availability(tmp_path: Path):
    _write_release_tree(tmp_path)
    installation = tmp_path / "docs" / "guides" / "installation.md"
    installation.write_text(
        installation.read_text().replace(
            f"| `dev` | PyPI {VERSION} and current source |",
            "| `dev` | PyPI 1.2.2 and current source |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="exact package versions|availability"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_second_unreleased_section(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text() + "\n## [Unreleased]\n\n- Left behind.\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="exactly one Unreleased"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_nonempty_unreleased_section(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            "## [Unreleased]\n\n",
            "## [Unreleased]\n\n- Not released.\n\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="must be empty"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_empty_current_release_section(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace("### Added\n\n- Current release.\n\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="current release section"):
        verify_release_source(tmp_path, f"v{VERSION}")


@pytest.mark.parametrize(
    "replacement",
    [
        "<!-- no release notes -->\n\n",
        "### Added\n\n",
        "### Added\n\n<!-- no release notes -->\n\n",
    ],
)
def test_release_source_rejects_current_release_without_release_notes(
    tmp_path: Path,
    replacement: str,
):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            "### Added\n\n- Current release.\n\n",
            replacement,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="current release section"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_malformed_release_heading(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            f"## [{PREVIOUS_VERSION}] - 2026-07-31",
            f"## [{VERSION}] — 2026-08-11\n\n"
            "- Duplicate current release.\n\n"
            f"## [{PREVIOUS_VERSION}] - 2026-07-31",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="malformed level-two"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_impossible_release_date(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace("2026-08-12", "2026-99-99"),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="date is invalid"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_release_date_order_drift(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text()
        .replace("2026-08-12", "2026-07-01")
        .replace("2026-07-31", "2026-08-01"),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="reverse chronological"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_duplicate_link_definition(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text()
        + f"\n[Unreleased]: https://example.invalid/v{VERSION}...HEAD\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="repeats link definition"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_casefolded_duplicate_link_definition(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text() + "\n[unreleased]: https://example.invalid/wrong\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="repeats link definition"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_missing_historical_release_link(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            f"[{PREVIOUS_VERSION}]: https://github.com/hebo1221/nano-deepseek-v4/"
            f"tree/v{PREVIOUS_VERSION}\n",
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="comparison links"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_rejects_release_order_drift(tmp_path: Path):
    _write_release_tree(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            f"## [{PREVIOUS_VERSION}] - 2026-07-31",
            "## [2.0.0] - 2026-07-31",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseSourceError, match="strictly descending"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_wraps_invalid_toml(tmp_path: Path):
    _write_release_tree(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project\n", encoding="utf-8")

    with pytest.raises(ReleaseSourceError, match="pyproject.toml is invalid"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_reports_missing_required_file(tmp_path: Path):
    _write_release_tree(tmp_path)
    (tmp_path / "README.md").unlink()

    with pytest.raises(ReleaseSourceError, match="cannot read README.md"):
        verify_release_source(tmp_path, f"v{VERSION}")


def test_release_source_cli_reports_pass_and_failure(tmp_path: Path, capsys):
    _write_release_tree(tmp_path)

    assert main(["--root", str(tmp_path), "--tag", f"v{VERSION}"]) == 0
    assert capsys.readouterr().out == f"release source: PASS (v{VERSION})\n"

    assert main(["--root", str(tmp_path), "--tag", "v9.9.9"]) == 1
    assert "release source verification failed: release tag" in capsys.readouterr().err
