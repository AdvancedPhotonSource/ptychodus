---
name: pre-push
description: Run the ptychodus CI gate locally before pushing — ruff check, ruff format --check, mypy, pytest, and the Markdown linter, matching what .github/workflows/python-package.yml runs on PR. Use when the user says "check before push", "run CI locally", "pre-push", or after a batch of code changes when they're about to open/update a PR.
---

# pre-push

Runs the four commands `.github/workflows/python-package.yml` runs on every PR, in the same order, so a green run here means CI will pass — plus a Markdown lint pass that CI does not run but the project's documentation style depends on.

## Steps

Run these sequentially. Stop at the first failure and report clearly; do not proceed to the next step until the current one is clean.

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ptychodus scripts
uv run pytest
uv run pymarkdown scan $(git ls-files '*.md')
```

## Handling failures

- **`ruff check` failure:** show the violations. If they are auto-fixable (`--fix` would resolve them), ask the user before running `uv run ruff check --fix .`. Never auto-fix `N` (naming) violations — those often involve deliberate `# noqa: N…` on physical-quantity names per CLAUDE.md.
- **`ruff format --check` failure:** offer to run `uv run ruff format .` and re-run the check. This is almost always safe to apply.
- **`mypy` failure:** show the errors with file:line. Do not attempt fixes here — hand back to the user; typing changes often need real thought.
- **`pytest` failure:** show the failing test name(s) and a compact traceback. Ask the user how to proceed before rerunning.
- **`pymarkdown` failure:** show the `file:line: MDxxx` findings and apply the rule each message names — the house Markdown style is codified in CLAUDE.md's Conventions section. Do not relax `[tool.pymarkdown]` in `pyproject.toml` to silence a finding; that config records deliberate carve-outs (`MD013` line length, `MD014` shell prompts) and widening it hides real drift.

## Scope shortcuts

If the user's diff only touches one subpackage, you can offer to run the module-scoped variant instead:

```sh
uv run ruff check src/ptychodus/<subpackage>
uv run ruff format --check src/ptychodus/<subpackage>
uv run mypy src/ptychodus/<subpackage>
uv run pytest tests/test_<subpackage>.py  # if a matching test file exists
```

Use this only when the user asks for a fast local check on WIP; the full sweep above is what CI actually runs.

## Notes

- Ruff rules for this repo: `F, N, NPY`; single-quoted strings; 100-char lines; py311 target (see `pyproject.toml`).
- `mypy` targets `src/ptychodus` and the top-level `scripts/` tree. `src/ptychodus_store` is not currently in CI's mypy job — check `pyproject.toml`/`.github/workflows/python-package.yml` before assuming coverage.
- `pymarkdown` is a local-only gate; there is no Markdown job in CI. Scan via `git ls-files` so `.venv/`, `docs/build/`, and other untracked trees stay out of scope.
- Do not add `--no-verify` or skip hooks to work around a failure; investigate and fix instead.
- Widget tests in `tests/view/` need `QT_QPA_PLATFORM=offscreen` or they `SIGABRT` in `QApplication([])` on a headless machine. That default is now set inside `tests/view/conftest.py` via `os.environ.setdefault(...)`, so plain `uv run pytest` is safe from any invocation path (IDE, CI, terminal). A developer with a live display who wants to actually see the widgets can pre-set `QT_QPA_PLATFORM=xcb` — `setdefault` won't override it. CI itself never collects these tests: its job installs ptychodus without `--extra gui`, so the outer `tests/conftest.py` drops the whole subtree.
