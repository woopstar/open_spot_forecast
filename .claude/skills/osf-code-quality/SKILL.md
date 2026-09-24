---
name: osf-code-quality
description: Activate before every commit and PR to run the full OSF quality gate pipeline — lint, typing, quality checks, and tests.
---

# OSF Code Quality — Pre-Commit & Pre-PR Gates

Activate this skill **before every commit** and **before opening any PR**.

## Four Quality Gates — All Must Pass

Run these in order. If any fails, fix the issues before proceeding.

### Gate 1: Lint

```bash
./scripts/quality.sh lint
```

Runs: ruff format → ruff check. Auto-formats and checks for style violations.

### Gate 2: Type Checking

```bash
./scripts/quality.sh typing
```

Runs: mypy type checking. Must pass with **0 errors**.

Rules:

- `disable_error_code` is empty in `pyproject.toml` — never add new suppressions
- No `# type: ignore` without a comment justifying why

### Gate 3: Quality

```bash
./scripts/quality.sh quality
```

Runs: pyright + vulture static checks. Must pass with **0 errors**.

### Gate 4: Tests

```bash
./scripts/quality.sh test
```

Runs: pytest with coverage on Python 3.14.

For faster iteration during development:

```bash
./scripts/quality.sh test tests/test_module.py           # specific file
./scripts/quality.sh test tests/test_module.py::test_fn   # specific test
```

### All Gates at Once

```bash
./scripts/quality.sh all
```

## File Size Check

Hard limit: **30 KB AND 1000 lines** per file across the codebase. Check before PR:

```bash
wc -c custom_components/open_spot_forecast/**/*.py
wc -l custom_components/open_spot_forecast/**/*.py
```

If a file exceeds either limit, split it before adding more features.

## Pre-Commit Hook (Optional but Recommended)

```bash
pre-commit run --all-files
```

## Code Standards Quick Reference

- **Float comparisons**: epsilon guard in production (`abs(x) > 1e-9`), `pytest.approx()` in tests
- **Type hints**: every public function has parameter and return type annotations; use `| None`, not `Optional[...]`
- **Docstrings**: Google-style for public modules, classes, functions, methods
- **`@override`**: every method that overrides a base class method
- **Import order**: standard library → third-party → `homeassistant.*` → `custom_components.open_spot_forecast.*`
- **String formatting**: f-strings for runtime, `%`-formatting for logging
- **Encoding**: `open(file, encoding='utf-8')` in text mode
- **Paths**: `pathlib` over `os.path`
- **Modular code**: DRY principle, break into components

## Verify Before Commit

```bash
git --no-optional-locks status
```

Only intended changes should appear. If any of the four gates fail, fix them before committing. Do not submit a PR with formatting, linting, typing, or test failures.
