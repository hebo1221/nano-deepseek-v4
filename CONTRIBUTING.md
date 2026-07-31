# Contributing

Contributions that improve architectural fidelity, correctness, tests,
documentation, or small-scale usability are welcome. Frontier-scale serving and
distributed training infrastructure remain outside this reference
implementation's scope.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,official]"
```

Run the same quality gates used in CI before opening a pull request:

```bash
ruff check .
yamllint .github .yamllint.yml
mypy nano_deepseek_v4
pytest --cov=nano_deepseek_v4 --cov-report=term-missing --cov-fail-under=75
python -m build
python -m twine check --strict dist/*
```

CUDA tests are marked separately and skip automatically on CPU-only systems:

```bash
pytest -m gpu tests/test_accelerator.py
```

## Change expectations

- Add a regression test for behavior changes and bug fixes.
- Keep tiny CPU configurations fast; large official presets must not be
  instantiated in ordinary unit tests.
- Preserve public API compatibility or document intentional breaking changes in
  `CHANGELOG.md`.
- Validate checkpoint and cache inputs before allocating or mutating model
  state. Never add pickle-based loading for untrusted artifacts.
- Keep unrelated formatting or generated files out of the change.

Pull requests should explain the problem, the chosen tradeoffs, and the exact
commands used for verification. Architecture changes should cite the relevant
technical-report section or another primary source.
