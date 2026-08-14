# Contributing

Contributions that improve architectural fidelity, correctness, tests,
documentation, or small-scale usability are welcome. Frontier-scale serving and
distributed training infrastructure remain outside this reference
implementation's scope.

## Reporting and proposing work

Search existing issues and discussions before opening something new. Use the
[structured bug form](https://github.com/hebo1221/nano-deepseek-v4/issues/new?template=bug_report.yml)
for reproducible incorrect behavior, including the exact install origin,
version or commit, environment, command, output, and expected result. Ask usage
questions in
[Q&A](https://github.com/hebo1221/nano-deepseek-v4/discussions/categories/q-a)
and propose scoped additions in
[Ideas](https://github.com/hebo1221/nano-deepseek-v4/discussions/categories/ideas).
Report suspected vulnerabilities only through the
[private security form](https://github.com/hebo1221/nano-deepseek-v4/security/advisories/new).

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.4"
python -m pip install -e ".[dev,official,parity]"
```

This is the CPU development setup. CUDA contributors should select and install
the matching PyTorch build first, then run the editable-install command. See the
[installation guide](docs/guides/installation.md) for both paths; a bare Linux
editable install can otherwise resolve an unintended CUDA dependency stack.

During development, run the fast checks plus the tests that cover your change:

```bash
ruff check .
mypy nano_deepseek_v4 scripts tests
pytest tests/path_to_relevant_test.py
```

Before opening a pull request, run the complete local gate when practical:

```bash
yamllint .github .yamllint.yml
pytest --cov=nano_deepseek_v4 --cov-report=term-missing --cov-fail-under=75
```

Architecture, cache, routing, MTP, rotary, or tensor-layout changes must also
run the independent conformance checks:

```bash
nano-deepseek-v4 attention-reach
nano-deepseek-v4 dspark
nano-deepseek-v4 dspark-scheduler
nano-deepseek-v4 parity
```

Maintainers run the packaging checks for release-facing changes; contributors
are welcome to run them but documentation-only patches do not need a local
distribution build:

```bash
python -m build
python -m twine check --strict dist/*
```

CUDA tests are marked separately and skip automatically on CPU-only systems:

```bash
pytest -m gpu tests/test_accelerator.py
```

## Change expectations

- Add a regression test for behavior changes and bug fixes.
- Run the pinned parity verifier for architecture, cache, routing, MTP, or tensor-layout changes.
- Keep tiny CPU configurations fast; large official presets must not be
  instantiated in ordinary unit tests.
- Preserve public API compatibility or document intentional breaking changes in
  `CHANGELOG.md`.
- Validate checkpoint and cache inputs before allocating or mutating model
  state. Never add pickle-based loading for untrusted artifacts.
- Keep Hub inspection tests offline with deterministic API/header doubles. For
  release-facing changes, also dogfood one immutable public revision and record
  the exact selected-shard inventory without adding a flaky network CI gate.
- Keep unrelated formatting or generated files out of the change.

Pull requests should explain the problem, the chosen tradeoffs, and the exact
commands used for verification. Architecture changes should cite the relevant
technical-report section or another primary source.
