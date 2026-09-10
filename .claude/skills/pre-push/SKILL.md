---
name: pre-push
description: Run the ptychodus CI gate locally before pushing — ruff check, ruff format --check, mypy, pytest, and the Markdown linter, matching what .github/workflows/python-package.yml runs on PR. Runs as two concurrent tiers so a lint slip fails in seconds. Use when the user says "check before push", "run CI locally", "pre-push", or after a batch of code changes when they're about to open/update a PR.
---

# pre-push

Runs the four commands `.github/workflows/python-package.yml` runs on every PR, so a green run here means CI will pass — plus a Markdown lint pass that CI does not run but the project's documentation style depends on.

The five checks are split into two tiers by cost. Within a tier they run **concurrently**; the tiers themselves are sequential. Concurrent `uv run` invocations do not contend on a venv lock, so this is safe.

## Steps

### Tier 1 — lint and docs (about 3s)

Everything here is sub-3-second. Run all three at once and **stop the whole gate if any fails** — a formatting slip should cost seconds, not minutes.

```sh
d=$(mktemp -d)
run() { local n=$1; shift; ( "$@" >"$d/$n.log" 2>&1; echo $? >"$d/$n.rc" ) & }

run ruff-check   uv run ruff check .
run ruff-format  uv run ruff format --check .
run pymarkdown   uv run pymarkdown scan $(git ls-files '*.md')
wait

for f in "$d"/*.rc; do printf '%s %s\n' "$(cat "$f")" "$(basename "$f" .rc)"; done
echo "logs: $d"
```

Print the log only for entries whose code is non-zero. If all three are `0`, continue to Tier 2.

### Tier 2 — types and tests (about 100s)

Reached only when Tier 1 is clean. Probe for `pytest-xdist` first so this still works in a bare `pip install .` environment, where the flag would otherwise be an error:

```sh
uv run python -c "import xdist" 2>/dev/null && PAR="-n auto --dist loadfile" || PAR=""

run mypy   uv run mypy src/ptychodus scripts
run pytest uv run pytest -q $PAR
wait

for f in "$d"/*.rc; do printf '%s %s\n' "$(cat "$f")" "$(basename "$f" .rc)"; done
```

Measured on a 16-core machine: mypy 55s, pytest 187s serial or 99s under xdist, so Tier 2 is bounded by pytest at about 100s and the whole gate lands near 102s against 245s fully sequential.

`--dist loadfile` distributes whole test **files**, so same-file tests keep sharing module state and fixture ordering. Per-test distribution (`--dist load`) is faster on paper but is not what this suite has been verified against.

## Handling failures

A tier reports **every** check that failed, not just the first — that is the point of running them together. Fix them as a batch.

- **`ruff check` failure:** show the violations. If they are auto-fixable (`--fix` would resolve them), ask the user before running `uv run ruff check --fix .`. Never auto-fix `N` (naming) violations — those often involve deliberate `# noqa: N…` on physical-quantity names per CLAUDE.md.
- **`ruff format --check` failure:** offer to run `uv run ruff format .` and re-run the check. This is almost always safe to apply.
- **`mypy` failure:** show the errors with file:line. Do not attempt fixes here — hand back to the user; typing changes often need real thought.
- **`pytest` failure:** show the failing test name(s) and a compact traceback. Ask the user how to proceed before rerunning.
- **`pytest` failure that only reproduces in parallel:** re-run serially with `uv run pytest -q` before believing it. A test that passes serially and fails under xdist is a genuine finding about shared state — report it as such rather than treating the parallel run as flaky.
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
- `pytest-xdist` is in `[dependency-groups] dev`, so `uv sync` installs it. It is deliberately **not** in `[tool.pytest.ini_options] addopts`: CI installs with `pip install . pytest` and has no xdist, so a global `addopts` would break every CI run. Parallelism stays a command-line flag the skill passes.
- `mypy` targets `src/ptychodus` and the top-level `scripts/` tree. `src/ptychodus_store` is not currently in CI's mypy job — check `pyproject.toml`/`.github/workflows/python-package.yml` before assuming coverage.
- `pymarkdown` is a local-only gate; there is no Markdown job in CI. Scan via `git ls-files` so `.venv/`, `docs/build/`, and other untracked trees stay out of scope.
- If pty-chi is installed editable (it is not on any registry, so `uv sync` cannot resolve it), add `--no-sync` to every `uv run` above or uv will try to re-resolve and fail.
- Do not add `--no-verify` or skip hooks to work around a failure; investigate and fix instead.
- Widget tests in `tests/view/` need `QT_QPA_PLATFORM=offscreen` or they `SIGABRT` in `QApplication([])` on a headless machine. That default is now set inside `tests/view/conftest.py` via `os.environ.setdefault(...)`, so plain `uv run pytest` is safe from any invocation path (IDE, CI, terminal), xdist workers included. A developer with a live display who wants to actually see the widgets can pre-set `QT_QPA_PLATFORM=xcb` — `setdefault` won't override it. CI itself never collects these tests: its job installs ptychodus without `--extra gui`, so the outer `tests/conftest.py` drops the whole subtree.
