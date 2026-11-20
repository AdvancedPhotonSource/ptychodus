---
name: curate-tests
description: Audit the ptychodus test suite so every test is correct, adds nontrivial value, runs efficiently and passes deterministically — vacuous and existence-only assertions, duplicate bodies, tests of third-party code, figure generators, non-hermetic paths, cheating fakes, production code only tests use, sleeps, unseeded randomness, unexplained loose tolerances, unjustified slow tests, unclassified warnings, order dependence and public-API coverage gaps — plus a protocol for de-flaking. Reports every finding, then asks once per category before changing anything. Use when the user says "curate the tests", "test suite QC", "remove low-value tests", "why is the suite slow", "fix the flaky test", or after a stretch of feature work has grown the suite.
---

# curate-tests

A test earns its place by being **correct** (it fails when the behaviour it names breaks), adding **nontrivial value** (no sibling, linter or guard test already pins the same thing), being **efficient** (its runtime buys an invariant nothing cheaper can observe), and being **robust** (it passes or fails the same way in any order, on any machine, every time). This skill audits `tests/` against those four, reports every finding as `tests/file.py::test` with a reason, and changes nothing until the user approves a category.

**Suite time is a tracked number.** Keeping it as small as is practical is a goal in its own right, so every run of this skill measures collection, serial and parallel wall time before curation (check 13) and again after it, and the final report carries the before/after table even when nothing was changed.

It owns judgment about the suite, not the gate. `pre-push` runs ruff, mypy and pytest; `pre-release` §10 runs the suite against a bare wheel install; the **Testing** section of [CLAUDE.md](../../../CLAUDE.md) decides which `view/` and `controller/` code is tested at all. Point at those rather than repeating them.

Operating rules for every step below:

- Invoke pytest as `uv run --no-sync python -m pytest -p no:cacheprovider`. A plain `uv run` syncs first, which silently replaces an editable pty-chi or PtychoPINN with the published one and drops anything installed outside the extras.
- Run plugins the project does not depend on through `uv run --no-sync --with <plugin>`. It builds an ephemeral overlay and leaves `uv.lock` and `.venv` untouched.
- Files a test left behind are reported, never deleted without asking.

## Phase A — static checks

Seconds each, no test run. The AST checks live in [audit_tests.py](audit_tests.py) beside this file; every hit is a lead to triage, not a verdict.

### 1. Vacuous tests

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py assertions | grep '^VACUOUS'
```

A test with no `assert`, no `pytest.raises`/`warns`/`approx`, no `assert_*` call and no call into a helper that asserts (followed transitively across `tests/`). Give it an assertion rather than deleting it — each one guards something real that currently cannot fail. A "does not raise" test of a validator is legitimate when a sibling proves the same validator raises; say so in its docstring.

### 2. Existence-only assertions

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py assertions | grep '^WEAK'
```

The test's sole assertion is `isinstance(...)`, `... is not None` or `len(...) > 0`. Delete it when a sibling makes the same call and checks content; strengthen it (shape, values, a round trip) when it is the only coverage. Keep it when the type **is** the contract — a dispatcher or factory whose job is to choose a class. `x is None` is not flagged: it asserts an exact value.

### 3. Duplicate bodies

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py assertions | grep '^DUPLICATE'
```

Identical normalized bodies, docstrings ignored. Within one file it is a true duplicate — merge, or parametrize if the names carry distinct intent. Across files the same body can bind different module-level names (one section table per library) and test different things; confirm before merging.

### 4. Tests of something else

```sh
grep -rnE 'savefig|matplotlib|^\s*print\(' tests/ --include='*.py'
```

- A test whose assertions only exercise a third-party library — matplotlib's colormap registry, `out.exists()` after `savefig` — tests that library. Delete it.
- A figure generator posing as a test has no assertion and writes a picture. If the figure is still worth having, move it to `scripts/` as `scripts/plot_basis_modes.py` was; otherwise delete it.
- A debug `print(` goes. Text a test hands to a child interpreter is not one.

### 5. Out of scope

```sh
git ls-files 'tests/view/*' 'tests/controller/*'
```

Compare against the allowlist in the CLAUDE.md **Testing** section. Anything outside it is deleted, not fixed.

### 6. Not hermetic

```sh
grep -rnE "['\"]/(net|home|gdata|mnt|local)/" tests/ --include='*.py'
grep -rnE '\b(httpx|urllib\.request|socket)\.(get|post|urlopen|create_connection)\(' tests/ --include='*.py'
```

A test reads only `tests/data/` and `tmp_path`, never a beamline mount or a home directory, and never reaches the network (refusing `socket.connect` with `monkeypatch` is how a test proves code stays offline). Writes outside `tmp_path` surface dynamically in check 13.

### 7. Fakes that cheat

```sh
grep -rnE '\bcast\(' tests/ --include='*.py'
grep -rnE '# type: ignore(\s|$)|# type: ignore\[[^]]+\]\s*$' tests/ --include='*.py'
```

A test may not pass `None`, a duck-typed stand-in or a `cast()` where the signature names a real type — that tests a call production never makes. Build the real object, or one that raises if touched (a dataset whose reads fail proves a code path never reads it, which is strictly stronger). A `# type: ignore` needs an error code and a comment, and belongs only in a deliberate negative test; mypy covers `tests/` locally, so nothing else is silenced.

### 8. Production code only tests use

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py test-only
```

Public, undecorated `src/` definitions whose name appears in no production file but its own, while a test names it. Names an `api/` module lists in `__all__` are library surface and skipped. Triage each: a convenience overload or accessor kept for tests is deleted and the tests moved onto the production path; a function awaiting its first caller is the user's call.

### 9. Timing dependence

```sh
grep -rnE '\b(time|asyncio)\.sleep\(|^\s*sleep\(' tests/ --include='*.py'
```

A sleep that orders events races on a loaded machine. Replace it with a gate (`threading.Event`, a blocking fake that releases on signal) or a poll against a deadline that returns as soon as the condition holds. A child that sleeps so a test can kill it, and `asyncio.sleep(0)` yielding to the loop, are deliberate.

Most such sleeps do not fail when they lose the race; the test passes without reaching its scenario. A sleep before a call that would produce the same outcome anyway (a `stop()` that also cuts the scan, a final estimate that also satisfies "estimates are ordered") hides exactly that. Poll for the scenario's own evidence — a counter, an event the recorder saw — and assert it was reached. A sleep that needs no replacement, because ordering already comes from a queue, is deleted.

### 10. Randomness

```sh
grep -rnE '(numpy|np)\.random\.[a-z_]+\(|default_rng\(\)|^\s*(import|from) random\b|torch\.(rand|randn|randint)\(' tests/ --include='*.py' \
  | grep -vE '(numpy|np)\.random\.default_rng\([^)]'
```

Every draw comes from `numpy.random.default_rng(<seed>)`. Empty is the expected result; keep running it so it stays that way. A test that asserts a qualitative difference between two random draws states in its docstring why the difference is not just a different seed.

### 11. Tolerance audit

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py tolerances
```

`rtol`, `atol`, `rel` or `abs` at or above `1e-3` with no comment or docstring on or within three lines above. A tolerance states where it comes from — noise level, discretization order, an iterative solver's stopping rule, float32 accumulation — so a reader can tell a derived bound from one loosened until green. The second is a finding: derive the bound, or tighten it to what the computation achieves and note why. A reason in the test's own docstring, or a named constant whose comment explains it, satisfies the check even when the scan cannot see it. `1e-12` versus `1.0e-12` is not a finding.

Measure before tightening. [tolerance_recorder.py](tolerance_recorder.py) is a pytest plugin that logs the largest error each call site reaches, and `--achieved` puts it beside each flagged bound:

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py tolerances --loose 0.05 | cut -d: -f1 | sort -u > "$out-loose.txt"
CURATE_TOLERANCE_LOG="$out-achieved.json" PYTHONPATH=.claude/skills/curate-tests \
  uv run --no-sync python -m pytest -p no:cacheprovider -p tolerance_recorder -q $(cat "$out-loose.txt")
python3 -I .claude/skills/curate-tests/audit_tests.py tolerances --loose 0.05 --achieved "$out-achieved.json"
```

Then set each bound by what produced the error:

- **Deterministic** — discretization, a fixed grid, a seeded but noise-free construction: about twice the achieved error, rounded up to a 1-2-5 step. Exact results (achieved 0) take the default `approx`, or a roundoff bound with a note saying why the result is exact.
- **Seeded noise** — a median, a mean or a fit over random draws: a few standard errors of the estimator, computed and stated (the median of `N` unit normals has standard error `1.2533 / sqrt(N)`), never twice what one seed happened to give. A bound fitted to one seed breaks on the next seed change.
- **Iterative or device-dependent** — a reconstruction run, torch reductions: leave a margin for nondeterminism, and say what a real defect would look like (a gauge error is a factor, not a percent).

### 12. Skips and markers

```sh
grep -rnE 'pytest\.skip\(\s*\)|mark\.skip(\(\s*\))?\s*$|mark\.xfail|mark\.asyncio' tests/ --include='*.py'
grep -nE 'importorskip' tests/conftest.py tests/*/conftest.py
```

- Every skip states its reason.
- `@pytest.mark.asyncio` is redundant under `asyncio_mode = "auto"` in `pyproject.toml`; remove it.
- An `xfail` needs `strict=True` and a reason naming the defect, or it hides the day the defect is fixed.
- `importorskip` in a conftest is reported as an error, not a skip — the conftest uses `collect_ignore` instead. A comment saying so is not a hit.

## Phase B — dynamic checks

### 13. Baseline: suite time and hermeticity

```sh
out="${TMPDIR:-/tmp}/curate-$(date +%Y%m%dT%H%M%S)"
git status --porcelain > "$out-status.txt"
uv run --no-sync python -m pytest -p no:cacheprovider -q --collect-only 2>&1 | tail -1
uv run --no-sync python -m pytest -p no:cacheprovider --durations=25 -rs -W default 2>&1 \
  | tee "$out-serial.log" | tail -60
uv run --no-sync python -m pytest -p no:cacheprovider -q -n auto --dist loadfile 2>&1 | tail -1
git status --porcelain | diff "$out-status.txt" -
```

Record the three times pytest prints — collection (`collected in`), serial and parallel (`passed … in`) — with pass/skip counts, the top 25 durations and the warnings summary; this is the "before" column. Durations come from the serial run, because under xdist they measure contention. The parallel time is what a developer waits for at `pre-push`, the serial time what CI's bare job pays; report both. Any line the final `diff` prints is a test writing into the checkout. Measure on an otherwise idle machine and say if it was not — a before/after comparison across different loads is noise.

### 14. Durations

Every top-25 entry above about a second carries a docstring saying why it is slow, and the accepted reason is that it spawns a process because the invariant cannot be observed in-process — a GPU context the parent must never hold, a pickled child, a cold interpreter. `--durations` reports setup, call and teardown separately; a slow `setup` row is a fixture, and a module- or session-scoped fixture paid once is cheaper than it looks. For anything else:

1. Profile it: `time.perf_counter()` around the fixture and the call, or `python -X faulthandler` with `faulthandler.dump_traceback_later` to find what blocks.
2. Stub an expensive probe nothing in the module reads (device discovery, a subprocess capability check) with an autouse `monkeypatch` **at the scope of the fixture that triggers it**. A module-scoped fixture runs before a function-scoped patch, so the stub must be module-scoped too. A probe the tests do read, asked again by each test, is asked once by a module-scoped fixture.
3. A spawned child imports the module holding its work before it runs anything. Keep child-work modules free of heavy imports, splitting work that needs `ptychodus.api.product` or a framework from work that does not, as `subprocess_child_fixtures` and `subprocess_reconstructor_fixtures` are split; time `python -c 'import <module>'` from `tests/` to see what each child pays.
4. Widen a fixture's scope when its value is immutable and tests only read it, and move work out of a slow or gated fixture.
5. Shrink arrays and iteration counts to the smallest the invariant holds at — a reconstruction test proves convergence on a 32-pixel object as well as on a 256-pixel one, unless resolution is what it asserts.
6. Collapse a parametrization whose cases exercise the same branch; keep one case per branch and per boundary.
7. Re-run check 13 and confirm the top 25 holds only justified entries.

### 14b. Collection time

```sh
uv run --no-sync python -X importtime -m pytest -p no:cacheprovider -q -s --collect-only 2> "$out-importtime.log" >/dev/null
sort -t'|' -k2 -n -r "$out-importtime.log" | head -25
```

`-s` is required: without it pytest captures file descriptor 2 during collection and the log loses every test-module import.

Collection is paid on every run, including a single `-k` test, and every xdist worker pays it again. Import cost is the usual culprit: a test module or `*_fixtures.py` importing torch, pty-chi or a plugin package at module scope when only a few of its tests need it. Move that import into the tests that need it, behind a function-level `pytest.importorskip` where the dependency is optional. Never move it into a conftest, which every module then pays.

Pytest plugins that third-party packages install and pytest loads automatically (logfire, hydra) appear in the same ranking. Their cost belongs to the environment, not the suite: report it, and leave `addopts` empty, as `pre-push` explains, rather than adding `-p no:<plugin>` to `pyproject.toml`.

### 15. Warnings ratchet

Classify every warning in the baseline summary:

- **The warning is the behaviour under test** (inf from comparing identical arrays, an evanescent mode): assert it with `pytest.warns`, or mark that test `@pytest.mark.filterwarnings('ignore:<message>:<Category>')`, or use `numpy.errstate` in the source where inf or NaN is a documented return.
- **The input is avoidable** (a propagator built at zero distance): fix it at the source. Suppressing it deletes the only signal.

Never filter globally — a blanket `ignore::RuntimeWarning` hides the next real divide-by-zero. Report the count against the last run; once none is unclassified, propose `filterwarnings = ["error"]` under `[tool.pytest.ini_options]` as a separate decision.

Python reports a warning once per source line per process, so a full run shows only the first test to reach each line, and fixing one cause can uncover another at a neighbouring line. After fixing, re-run the files that warned on their own with `-W default` and classify whatever surfaces. A warning raised inside a third-party library usually has its cause in the arguments ptychodus passes it (a tensor where the library copies with `torch.tensor`).

### 16. Order independence

```sh
for run in 1 2; do
  uv run --no-sync --with pytest-randomly python -m pytest -p no:cacheprovider -n auto --dist loadfile 2>&1 \
    | grep -E 'randomly-seed|passed|failed|error'
done
```

Under xdist the shuffle reorders files across workers and tests within each file, at a parallel run's cost. pytest-randomly loads from its entry point and prints `Using --randomly-seed=<seed>`; no `-q`, or the seed line is lost. A failure reproduces with `--randomly-seed=<seed>`. Bisect by running the failing test after halves of the tests that preceded it until one predecessor remains — that is the test leaking state. pytest-randomly also reseeds the global RNGs before each test, which can mask unseeded draws; check 10 covers those statically.

### 17. Public-API coverage gaps

```sh
python3 -I .claude/skills/curate-tests/audit_tests.py gaps
```

Report only. Public module-level `api/` functions and behavioural classes no test names; dataclasses and enums are skipped as covered through what returns them, and an abstract `FileReader`/`FileWriter` interface through its plugins. Flag the ones whose failure would be silent or costly — URL and credential guards, physical generators — and leave writing tests to a separate request.

## Correctness protocol

Applies to every test this skill writes, strengthens or rewrites, and to any test whose correctness is in doubt.

- **It can fail.** Break the behaviour it guards — stub the function, revert the fix, disable the branch — and watch it fail. Assert a value that differs from the default: a logger already at `WARNING` makes asserting `WARNING` pass either way. If an assertion cannot catch the bug at hand, say so in its docstring rather than letting it imply coverage.
- **It reaches its scenario.** Assert the precondition as well as the outcome (`assert not slow.timed_out`) so a fixture change cannot quietly turn the test into a different, easier one. A generator closed before its first `next()` runs none of its body.
- **Its name and comments are true.** A name claims only what the assertions check. A comment that contradicts the behaviour is fixed to the behaviour once the behaviour is confirmed right; a test that pins broken behaviour is a bug report, not coverage.
- **Its reference is independent.** Prefer a closed form that matches the discrete case exactly (a Dirichlet kernel, not the continuous sinc, for a sampled aperture), conservation laws and round trips. Compare complex values or phase, not only magnitudes — a unit-modulus phase error is invisible to `abs()`. Never compute the expected value with the implementation's own convention.
- **It asserts the code, not the framework.** Pytest's logging plugin re-adds its own handler, so "exactly one handler" is brittle.

## De-flaking protocol

1. **Measure.** Count failures over N full-suite runs, serial and under xdist, and state the rate. To show a flake predates the current change, run the baseline from a `git worktree add` of `HEAD`, never by stashing a dirty tree. Contention is a conclusion the numbers have to support, not a starting guess.
2. **Find the mechanism.** A native crash: `python -X faulthandler`, then `gdb --args python -m pytest …` until it reproduces, and read the C stack. Lazy initialization (a device context, a Qt event): wrap the initializer to print its caller's stack and run with `-s`. Confirm with `--collect-only -q` that a `-k` filter did not deselect the test under study. Drop each theory the evidence contradicts.
3. **Fix the cause.**
   - Stop the leak — global state, a device context, a widget shown and never closed.
   - Assert process-global state (no GPU context, nothing imported) in a freshly spawned child, so test order cannot reach it.
   - Give a Qt fixture `yield` and a teardown: `close()`, `deleteLater()`, `processEvents()`, `sendPostedEvents(None, QEvent.DeferredDelete)`, with a docstring saying the teardown is load-bearing.
   - Replace a sleep with a gate; write the ordering argument into the test.
   - Never retry, never add a rerun plugin, never loosen the assertion.
4. **Verify.** For an observed failure rate `p`, require `K` consecutive clean runs with `(1 - p)^K <= 0.05`, about `K = 3/p`, serial and under xdist. Confirm the test still fails with the cause reinstated. If a commit is requested before that finishes, report the fix as diagnosed, not proven.

## Approval and reporting

1. Report one table per category — `tests/file.py::test`, the reason, the proposed action and, for anything slow, the seconds it would save — and the baseline from check 13.
2. Ask one `AskUserQuestion` per category with a recommended option first — delete / strengthen / leave; for slow tests justify / speed up / move to a spawned probe; for figures move to `scripts/` / delete.
3. Apply the approved categories, run the correctness protocol on everything touched, then re-run check 13 for the "after" column.
4. Close with the suite-time table, always, whatever was approved:

   | Suite | Before | After | Change |
   | --- | --- | --- | --- |
   | Tests passed / skipped | 4555 / 4 | 4540 / 4 | −15 / 0 |
   | Collection | 30.1 s | 18.4 s | −11.7 s |
   | Serial | 187 s | 151 s | −36 s (−19 %) |
   | Parallel (`-n auto --dist loadfile`) | 99 s | 82 s | −17 s (−17 %) |
   | Slowest test | 57.0 s | 57.0 s | justified: spawns a child |
   | Warnings | 2 | 0 | −2 |

   The numbers above are illustrative. Name the changes that bought most of the saving, and the slow tests left in place with their justification.
5. Report defects in `src/` found along the way as separate items. Keep the change to the tests the user approved.

## Do not

- Commit.
- Add pytest plugins to `pyproject.toml`; the dynamic checks run them ephemerally.
- Filter warnings globally, or fix a flake by retrying.
- Add tests for `view/` or `controller/` code outside the CLAUDE.md allowlist.
- Silence mypy in tests with a bare `# type: ignore` or a `cast()`.
- Keep or add production code that exists only for a test.
- Delete a figure test whose plot is still worth having; move it to `scripts/`.
- Re-run what `pre-push` or `pre-release` already checks.
