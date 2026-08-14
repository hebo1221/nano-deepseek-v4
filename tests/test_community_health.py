from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
BUG_FORM = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
ISSUE_CONFIG = ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml"
Q_AND_A_URL = "https://github.com/hebo1221/nano-deepseek-v4/discussions/categories/q-a"
IDEAS_URL = "https://github.com/hebo1221/nano-deepseek-v4/discussions/categories/ideas"
SECURITY_URL = "https://github.com/hebo1221/nano-deepseek-v4/security/advisories/new"
FIELD_TYPES = {"checkboxes", "dropdown", "input", "markdown", "textarea"}
FIELD_ATTRIBUTES = {
    "checkboxes": {"description", "label", "options"},
    "dropdown": {"description", "label", "multiple", "options"},
    "input": {"description", "label", "placeholder"},
    "markdown": {"value"},
    "textarea": {"description", "label", "placeholder", "render", "value"},
}


def _assert_nonempty_string(value: object) -> None:
    assert isinstance(value, str)
    assert value.strip()


def _assert_valid_issue_form(form: object) -> None:
    assert isinstance(form, dict)
    assert set(form) <= {"assignees", "body", "description", "labels", "name", "title"}
    for key in ("name", "description"):
        _assert_nonempty_string(form[key])
    if "title" in form:
        _assert_nonempty_string(form["title"])
    for key in ("assignees", "labels"):
        if key not in form:
            continue
        values = form[key]
        assert isinstance(values, list)
        assert all(isinstance(value, str) and value.strip() for value in values)
        assert len(values) == len(set(values))

    assert isinstance(form["body"], list)
    assert form["body"]
    ids: list[str] = []
    labels: list[str] = []
    for field in form["body"]:
        assert isinstance(field, dict)
        assert set(field) <= {"attributes", "id", "type", "validations"}
        field_type = field["type"]
        assert field_type in FIELD_TYPES

        attributes = field["attributes"]
        assert isinstance(attributes, dict)
        assert set(attributes) <= FIELD_ATTRIBUTES[field_type]
        for attribute in ("description", "label", "placeholder", "render", "value"):
            if attribute in attributes:
                _assert_nonempty_string(attributes[attribute])
        if field_type == "markdown":
            assert "id" not in field
            assert set(attributes) == {"value"}
            continue

        field_id = field["id"]
        _assert_nonempty_string(field_id)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", field_id)
        ids.append(field_id)
        _assert_nonempty_string(attributes["label"])
        labels.append(attributes["label"])

        validations = field.get("validations", {})
        assert isinstance(validations, dict)
        assert set(validations) <= {"required"}
        if "required" in validations:
            assert isinstance(validations["required"], bool)

        if field_type == "dropdown":
            options = attributes["options"]
            assert isinstance(options, list)
            assert options
            assert all(isinstance(option, str) and option.strip() for option in options)
            assert len(options) == len(set(options))
            if "multiple" in attributes:
                assert isinstance(attributes["multiple"], bool)
        elif field_type == "checkboxes":
            options = attributes["options"]
            assert isinstance(options, list)
            assert options
            option_labels: list[str] = []
            for option in options:
                assert isinstance(option, dict)
                assert set(option) <= {"label", "required"}
                _assert_nonempty_string(option["label"])
                option_labels.append(option["label"])
                if "required" in option:
                    assert isinstance(option["required"], bool)
            assert len(option_labels) == len(set(option_labels))

    assert len(ids) == len(set(ids))
    assert len(labels) == len(set(labels))


def test_bug_report_form_matches_github_issue_forms_schema_contract():
    _assert_valid_issue_form(yaml.safe_load(BUG_FORM.read_text(encoding="utf-8")))


def test_bug_report_form_collects_a_reproducible_environment():
    form = yaml.safe_load(BUG_FORM.read_text(encoding="utf-8"))

    assert form["name"] == "Bug report"
    assert form["labels"] == ["bug"]
    fields = {item["id"]: item for item in form["body"] if "id" in item}
    assert {
        "preflight",
        "surface",
        "install_origin",
        "version",
        "environment",
        "reproduction",
        "expected",
        "actual",
        "output",
        "hub_revision",
        "additional_context",
    } == set(fields)

    required = {
        name
        for name, field in fields.items()
        if field.get("validations", {}).get("required") is True
    }
    assert required == {
        "surface",
        "install_origin",
        "version",
        "environment",
        "reproduction",
        "expected",
        "actual",
        "output",
    }
    assert all(option["required"] is True for option in fields["preflight"]["attributes"]["options"])

    surfaces = fields["surface"]["attributes"]["options"]
    for expected_surface in (
        "Installation or packaging",
        "Training",
        "Generation",
        "Native bundle save, inspect, or load",
        "Inference-cache persistence",
        "Hugging Face Hub or official checkpoint handling",
    ):
        assert expected_surface in surfaces

    text = BUG_FORM.read_text(encoding="utf-8")
    assert "from importlib.metadata import version" in text
    assert "git rev-parse HEAD" in text
    assert "resolved commit SHA" in text
    assert "Do not paste tokens" in text
    assert Q_AND_A_URL in text
    assert IDEAS_URL in text
    assert SECURITY_URL in text


def test_issue_picker_routes_non_bug_reports_to_the_right_channel():
    config = yaml.safe_load(ISSUE_CONFIG.read_text(encoding="utf-8"))

    assert config["blank_issues_enabled"] is False
    links = {item["name"]: item["url"] for item in config["contact_links"]}
    assert links == {
        "Usage questions": Q_AND_A_URL,
        "Feature ideas": IDEAS_URL,
        "Security vulnerability": SECURITY_URL,
    }


def test_public_contributor_docs_link_the_report_channels():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    for document in (readme, contributing):
        assert "https://github.com/hebo1221/nano-deepseek-v4/issues/new?template=bug_report.yml" in document
        assert Q_AND_A_URL in document
        assert IDEAS_URL in document
        assert SECURITY_URL in document
