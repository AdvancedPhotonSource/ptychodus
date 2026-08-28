---
name: pre-release
description: Run a comprehensive pre-release verification of the ptychodus repository — API docs coverage, minimal docstring presence, reader-plugin docs, README/CLAUDE.md/pyproject.toml consistency, install-instruction freshness, Markdown hygiene, zero-warning Sphinx build, full CI gate, entry-point smoke tests, and distribution-artifact contents. Report every finding and offer per-issue fixes. Use when the user says "pre-release check", "release audit", "before we cut a release", or "verify the repo is release-ready".
---

# pre-release

Comprehensive gate for cutting a release. Runs nine verification sections in order, reports pass/fail for each, and — for every failure — proposes a specific fix and asks the user before applying it. **Never fixes silently. Never commits.**

## How to run the check

The sections split into two phases by whether they touch the environment. Run Phase A concurrently, then Phase B strictly in order. Capture pass/fail and any findings for each section. At the end, print the summary table (last section) in its numeric order regardless of execution order — it doubles as reading order. Only then walk the user through fixing failures one at a time.

### Phase A — concurrent, read-only

Sections 1, 2, 3, 4a-c, 5a-c, 5.5, and 8 are pure inspection and share nothing. Fan them out and collect the results:

```sh
d=$(mktemp -d)
run() { local n=$1; shift; ( "$@" >"$d/$n.log" 2>&1; echo $? >"$d/$n.rc" ) & }
```

Launch each section's command with `run <section-name> <command>`, then `wait`, then read the logs. This takes about 15s against roughly 60s sequential.

### Phase B — sequential, environment-mutating

Sync **once**, naming every extra the gate depends on, then run Sections 6, 7, and 9 in order:

```sh
uv sync --extra docs --extra gui --extra globus --extra ptychi --extra store
```

Sections 6, 7, and 9 must not overlap — with each other or with Phase A. See "Why the phases" below.

---

### Section 1 — API modules present in `docs/source/api.md`

For every `src/ptychodus/api/**/*.py` file, confirm `docs/source/api.md` contains a matching `.. automodule:: ptychodus.api.<dotted.name>` directive. The scan is recursive, so subpackage modules (`preprocess/`, `simulate/`) are covered under their dotted names. A path component starting with `_` is skipped at any depth — that covers `__init__.py`, `simulate/_phase_unwrap.py`, and any future private subpackage.

Do not enumerate with a flat `ls src/ptychodus/api/*.py`, and do not match the directive with a `[a-z_]+` character class: the former misses every subpackage module, and the latter has no `.` so `preprocess.diffraction` truncates to `preprocess`. The two errors cancel out and the section reports `PASS` while blind to an undocumented subpackage module.

```sh
uv run python -c "
import pathlib, re
root = pathlib.Path('src/ptychodus/api')
have = set()
for p in sorted(root.rglob('*.py')):
    rel = p.relative_to(root).with_suffix('')
    if any(part.startswith('_') for part in rel.parts):
        continue
    have.add('.'.join(rel.parts))
doc = set(re.findall(r'\.\. automodule:: ptychodus\.api\.([A-Za-z_][A-Za-z_0-9.]*)',
                     pathlib.Path('docs/source/api.md').read_text()))
print(f'modules={len(have)} documented={len(doc)}')
for m in sorted(have - doc):
    print(f'MISSING FROM DOCS: ptychodus.api.{m}')
for m in sorted(doc - have):
    print(f'STALE DOC ENTRY:   ptychodus.api.{m}')
print('PASS' if have == doc else 'FAIL')
"
```

`PASS` if the script prints `PASS`. `FAIL` with the list of `MISSING FROM DOCS` and `STALE DOC ENTRY` lines.

A `STALE DOC ENTRY` is a module that `api.md` documents but that no longer exists. Sphinx catches this downstream as an autodoc import failure under Section 6's `-W`, but naming it here gives a far better message. The fix is to delete the orphaned stanza from `docs/source/api.md`.

**Suggested fix per missing module**: append this stanza to `docs/source/api.md`, matching the existing pattern (title-case `##` heading, then the autodoc directive as raw reStructuredText inside an `{eval-rst}` block):

````markdown
## <Title Case Name>

```{eval-rst}
.. automodule:: ptychodus.api.<stem>
   :members:
   :undoc-members:
   :show-inheritance:
```
````

Ask the user to confirm the heading text before writing.

---

### Section 2 — Minimal docstring coverage in `src/ptychodus/api/`

For each non-private module in `src/ptychodus/api/`, verify:

- Module has a top-of-file docstring.
- Every top-level `class` has a docstring on its first statement.
- Every top-level `def` whose name does NOT start with `_` has a docstring on its first statement.

Run this AST scan:

```sh
uv run python -c "
import ast, pathlib
missing = []
for p in sorted(pathlib.Path('src/ptychodus/api').glob('*.py')):
    if p.name.startswith('_'):
        continue
    tree = ast.parse(p.read_text())
    if not (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str)):
        missing.append(f'{p}:1: module docstring')
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('_'):
                continue
            if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                kind = type(node).__name__.replace('Def','').lower()
                missing.append(f'{p}:{node.lineno}: {kind} {node.name!r}')
for m in missing:
    print(m)
print(f'--- {len(missing)} missing ---')
"
```

`PASS` if 0 missing. `FAIL` otherwise, with the list of `file:line` findings.

**Suggested fix per finding**: propose a one-line docstring derived from the identifier name (e.g. for `class DiffractionPattern:` propose `"""A diffraction pattern captured by the detector."""`) and let the user accept, edit, or skip. Do not batch these — one at a time, so the user can improve the wording as you go.

---

### Section 3 — Reader plugins represented in `docs/source/readers.md`

`docs/source/readers.md` is a curated bullet list of user-facing names, not a 1:1 file mapping. Cross-check it against the plugin registry rather than against filenames: every plugin already declares a user-facing `display_name` in its `register_plugins` hook, and those strings are the authoritative source.

Drive the check from `PluginRegistry.load_plugins()` so it reuses the exact discovery path the application uses. This matters because `pkgutil.iter_modules` is non-recursive: it yields `aps33id_velociprobe` as a *package* and calls the `register_plugins` in its `__init__.py`. A flat `ls src/ptychodus/plugins/*.py` misses that package entirely, and a hand-maintained keyword table drifts silently as new plugins land — do not reintroduce either.

Comparison is by facility token and beamline token, in two tiers:

- **FAIL** — a token a plugin registers that `readers.md` never mentions. This is the real drift risk: a new beamline reader lands and the doc is not updated.
- **REVIEW** — a token the doc claims that no plugin display name mentions. Report-only, because the doc legitimately uses friendlier names than the plugins do. Making this a hard failure would produce false alarms.

```sh
uv run python -c "
import logging, re, pathlib

skipped = []
class _H(logging.Handler):
    def emit(self, record):
        m = record.getMessage()
        if m.startswith(('Skipping ', 'Failed to register ')):
            skipped.append(m)
_lg = logging.getLogger('ptychodus.api.plugins')
_lg.addHandler(_H())
_lg.setLevel(logging.WARNING)

from ptychodus.api.plugins import PluginRegistry
registry = PluginRegistry.load_plugins()
names = {p.display_name
         for a in vars(registry) if a.endswith('_file_readers')
         for p in getattr(registry, a)}
doc = pathlib.Path('docs/source/readers.md').read_text()

FACILITIES = ['APS', 'CNM', 'LCLS', 'MAX IV', 'NSLS-II', 'SLAC', 'SLS']
BEAMLINE = re.compile(r'\b\d+-ID\b')

def tokens(text):
    found = {f for f in FACILITIES if re.search(rf'\b{re.escape(f)}\b', text)}
    return found | set(BEAMLINE.findall(text))

plug = tokens(' | '.join(names))
docs = tokens(doc)
print(f'reader plugins registered: {len(names)}; modules skipped at load: {len(skipped)}')
for m in skipped:
    print(f'  {m}')
fails = sorted(plug - docs)
for t in fails:
    print(f'FAIL   in plugins, missing from readers.md: {t}')
for t in sorted(docs - plug):
    print(f'REVIEW in readers.md, no plugin names it: {t}')
print('PASS' if not fails else 'FAIL')
"
```

Two details in that script are load-bearing:

- Facility matching uses `\b...\b`, not substring. A plain `'SLS' in text` also matches **NSLS-II**, which would silently mask a missing SLS entry.
- The script reports `modules skipped at load`. An optional-dependency plugin that fails to import registers nothing, which shrinks the plugin token set and weakens the check in the safe direction — under-reporting, never a false `FAIL`. Run the release gate in an environment with the full extras, and treat a nonzero skip count as a caveat on this section's coverage.

`PASS` if the script prints `PASS` (no `FAIL` lines). `REVIEW` lines do not fail the gate, but report them so a human can adjudicate.

**Suggested fix per finding**: propose a bullet insertion under the correct facility heading, matching the existing style (`- <Beamline name> (<abbreviation>)`). Ask the user to confirm the heading and wording — the doc uses friendly names, not filenames.

---

### Section 4 — README, CLAUDE.md, pyproject.toml consistency

**4a. Extras named in `README.md` all exist in `pyproject.toml`.**

```sh
# Extract extras from README.md install commands.
# Anchor on `ptychodus[...]` so Markdown link labels ([Ptychodus], [uv]) don't match:
grep -oE 'ptychodus\[[a-z,]+\]' README.md | sed 's/.*\[//; s/\]//' | tr ',' '\n' | sort -u

# Extras declared in pyproject.toml:
uv run python -c "
import tomllib
with open('pyproject.toml','rb') as f:
    p = tomllib.load(f)
for k in sorted(p['project']['optional-dependencies']):
    print(k)
"
```

`PASS` if the README extras are a subset of the pyproject extras. `FAIL` with the missing extras.

**Suggested fix**: either add the extra to `pyproject.toml` (rare — usually intentional) or remove/rename the extra reference in `README.md`.

**4b. Every `[project.scripts]` entry is documented or referenced.**

```sh
uv run python -c "
import tomllib
with open('pyproject.toml','rb') as f:
    p = tomllib.load(f)
for name in sorted(p['project'].get('scripts', {})):
    print(name)
"
```

For each script name, confirm it appears in `CLAUDE.md` OR `docs/source/getting_started.md`. `PASS` if all are referenced.

**Suggested fix**: add a `uv run <script>` example under the "Common Commands" section of CLAUDE.md (project convention — see the existing block).

**4c. Python version claim matches pyproject.**

- CLAUDE.md says "Python ≥3.11".
- Confirm `pyproject.toml` `requires-python` matches. `PASS`/`FAIL` accordingly.

---

### Section 5 — Installation instructions are fresh

**5a. Dockerfile variants referenced still exist.** Every Dockerfile name mentioned in `docs/source/getting_started.md` and `CLAUDE.md` should be a real file in `containers/`:

```sh
grep -hoE 'Dockerfile\.[a-z]+' docs/source/getting_started.md CLAUDE.md | sort -u | while read f; do
    [ -f "containers/$f" ] || echo "MISSING: containers/$f"
done
```

`PASS` if empty. `FAIL` lists missing Dockerfiles.

**5b. Extras named in `docs/source/getting_started.md` install commands exist in pyproject.**

Same technique as 4a but against getting_started.md.

**5c. The `uv sync` command in README.md uses currently-supported extras.**

Parse the `uv sync --extra <x> --extra <y> ...` line in README.md and confirm each extra is in `pyproject.toml`.

---

### Section 5.5 — Markdown hygiene

Docs are MyST Markdown; no reStructuredText should reappear, and every tracked `.md` must satisfy the house style codified in `CLAUDE.md`.

```sh
# No stray reStructuredText anywhere in the repo:
git ls-files '*.rst'

# House style, mechanically enforced:
uv run pymarkdown scan $(git ls-files '*.md')

# Shell fences use `sh`, not `bash`:
git ls-files '*.md' | xargs grep -l '^```bash'
```

`PASS` if the first and third commands print nothing and `pymarkdown` exits 0. `FAIL` otherwise, listing each offending file.

**Suggested fix**: for `.rst` files, convert to MyST Markdown and update every reference. For linter findings, apply the rule the message names — do not widen the `[tool.pymarkdown]` config in `pyproject.toml` to silence a real violation.

---

### Section 6 — Sphinx build with zero warnings

```sh
make -C docs clean
make -C docs html SPHINXOPTS="-W --keep-going"
```

The sync happens once at the top of Phase B, not here. A bare `uv sync --extra docs` at this point would uninstall `gui`, `globus`, `ptychi`, and `store` before Section 7 runs — see "Why the phases".

`PASS` if exit code 0. `FAIL` on any warning or error.

**Suggested fix**: hand the first warning to the user. Common causes: undocumented cross-references, autodoc import failures (usually a missing extra — retry with `uv sync --extra docs --extra ptychi --extra globus --extra gui --extra store` if autodoc can't import a module), duplicate labels. Not auto-fixable.

*Note*: `-W` is strict, and the prose sources are expected to build clean — treat any warning originating in `docs/source/*.md` as a regression introduced by the change under review. Warnings raised from `src/ptychodus/api/*.py` docstrings (reStructuredText syntax errors surfaced by autodoc) are a known pre-existing backlog; report the count and the first offender, but do not fold fixing them into the release gate.

---

### Section 7 — Full CI gate (chain `pre-push`)

Invoke the `pre-push` skill. Both of its tiers must be green — Tier 1 (ruff check, ruff format --check, pymarkdown) and Tier 2 (mypy, pytest). Let it run its own two-tier schedule; do not re-run its checks here.

`PASS` if `pre-push` finishes clean. `FAIL` on any failing step; hand the failure back to the user without attempting fixes here (the `pre-push` skill knows how to offer format fixes; deeper fixes belong outside the release gate).

---

### Section 8 — Entry-point smoke test

For each name in `[project.scripts]`, run `uv run <script> --version`; if the command doesn't implement `--version`, fall back to `--help`. Exit 0 counts as PASS; non-zero or import error counts as FAIL.

```sh
uv run python -c "
import tomllib
with open('pyproject.toml','rb') as f:
    print('\n'.join(tomllib.load(f)['project']['scripts']))
" | xargs -P 8 -I{} sh -c '
    s=$1
    if uv run "$s" --version >/dev/null 2>&1; then
        echo "PASS: $s"
    elif uv run "$s" --help >/dev/null 2>&1; then
        echo "PASS: $s (--help)"
    else
        echo "FAIL: $s"
    fi
' _ {}
```

`PASS` if every script exits 0. `FAIL` lists the broken entry points — usually caused by import-time errors introduced by an unrelated change.

### Section 9 — Distribution artifacts

The release artifacts are what PyPI users actually get, and nothing else in this audit inspects them. In particular the `ptychodus-store` web UI is compiled TypeScript that is gitignored, so it can silently go missing from a build.

First, fail if a stale `build/` tree is present — `setuptools` reuses `build/lib/`, so a leftover from an earlier build can ship outdated modules:

```sh
test -d build && echo "FAIL: stale build/ present — rm -rf build dist src/*.egg-info" || echo "PASS: no stale build/"
```

Then build both artifacts into a temp directory with the UI requirement enforced, and confirm the compiled UI landed in each. Do not build into `./dist/` — that is the maintainer's release output.

```sh
out=$(mktemp -d)
PTYCHODUS_STORE_REQUIRE_UI_BUILD=1 uv build --no-sources --out-dir "$out" || echo "FAIL: uv build"
ts=$(find src/ptychodus_store/ui/src -name '*.ts' | wc -l)
sdist=$(tar -tzf "$out"/ptychodus-*.tar.gz | grep -c 'ui/dist/.*\.js$')
wheel=$(unzip -Z1 "$out"/ptychodus-*.whl | grep -c 'ui/dist/.*\.js$')
echo "ts=$ts sdist=$sdist wheel=$wheel"
[ "$ts" = "$sdist" ] && [ "$ts" = "$wheel" ] && echo "PASS: UI in both artifacts" || echo "FAIL: compiled UI missing or incomplete"
uvx twine check "$out"/* || echo "FAIL: twine check"
```

`FAIL` on a count mismatch means the `sdist`/`build_py` hooks in `setup.py` did not compile the UI — check that `tsc` is on `PATH`. Report the temp directory path so the user can inspect the artifacts, and do not copy them into `./dist/`.

---

## Final report

Print a summary in this exact format after all sections have run:

```text
1. API modules in docs         PASS   (<covered>/<total>)
2. Docstrings in api           PASS|FAIL   (<n> missing)
3. Reader plugins in docs      PASS|FAIL   (<n> orphans)
4. README/CLAUDE/pyproject     PASS|FAIL   (<one-line summary>)
5. Install instructions        PASS|FAIL   (<one-line summary>)
5.5 Markdown hygiene           PASS|FAIL   (<n> rst files, <n> lint findings)
6. Sphinx build (zero warns)   PASS|FAIL   (first warning if any)
7. Full CI gate                PASS|FAIL
8. Entry-point smoke tests     PASS|FAIL   (<pass>/<total>)
9. Distribution artifacts      PASS|FAIL   (ui .js sdist/wheel vs .ts count)
```

Then, and only then, walk through each `FAIL` with the user: state the finding, propose the fix, ask "apply it?" per item, and Edit/Write only after they confirm.

Once the gate is clean (or the user accepts the outstanding findings), run the **Draft release notes** step below.

## Draft release notes

Advisory step, not a gate. Synthesize a bulleted draft the user pastes into `git tag -a` or the PR description — never commit, never tag, never write to a file.

**Step 1 — Read prior tags to internalize the style.** This is the single most important instruction; every release note in this repo follows the convention set by prior tags, so seeing them fresh anchors the draft:

```sh
for tag in $(git tag --sort=-creatordate | head -5); do
    echo "===== $tag ====="
    git tag -l --format='%(contents)' "$tag"
    echo
done
```

The convention you should see: title line `<Short title> (#<PR-number>)`, blank line, flat `-` bullets, one user-visible change per bullet. Past-tense or imperative voice. Backticks on paths, extras, CLI names, module names, file names. No sub-bullets. Bugfix-only tags (`v1.4.1`) legitimately have zero bullets — just the title. Big releases (`v1.4.0`, `v1.5.0`) run 20–30 bullets.

**Step 2 — Gather raw material since the previous tag:**

```sh
prev=$(git describe --tags --abbrev=0)
echo "Since $prev"
git log --oneline "$prev..HEAD"
git log "$prev..HEAD"
git diff --stat "$prev..HEAD" -- CLAUDE.md README.md docs/source/
```

The `--oneline` view gives you the merge granularity (usually one bullet per merged PR); the full log gives you the "what" behind each merge; the doc `--stat` flags the notable user-facing changes.

**Step 3 — Optionally consult Claude Code session transcripts** under `~/.claude/projects/-home-beams0-SHENKE-Ptychography-ptychodus/*.jsonl` when a commit message is thin — the *why* often lives in a conversation, not the message. Read-only; sample only when needed to explain a specific commit.

**Step 4 — Draft the bullet list applying the convention:**

- Past-tense or imperative voice, backticked identifiers, flat structure, no sub-bullets, one short line per bullet.
- **Frame every bullet in terms of value to the user** — the new capability, the fixed symptom, the new supported hardware/format, the faster or simpler workflow. Ask "what does this let the user do or stop worrying about?" and lead with that. Do not paraphrase the commit message or the file-level diff. Compare: the actual `v1.5.0` tag says *"GPU contexts are now acquired only inside reconstruction subprocesses, so the parent process no longer holds device state"* (observable behavior, with the "so that" the user cares about); a bad draft would say *"Moved GPU context acquisition into `ReconstructorLibrary._acquire_context`"* (file-level restatement of a diff).
- **Keep each bullet as terse as it can be while still conveying the value.** Target the median density of `v1.4.0` and `v1.5.0` — a phrase or a short sentence. A longer bullet is warranted only when the scope genuinely needs it (headline features spanning multiple surfaces, like the `v1.5.0` opening bullet). A bullet that reads like a full paragraph is a signal to split it or to drop the implementation detail. Do not pad with rationale that a reader can infer from the phrase itself.
- Skip release-noise commits (version bumps, formatting, typo fixes) and internal refactors the user cannot observe. A pure rename earns a bullet only when downstream code or documentation had to change. A dependency swap earns a bullet only when a downstream packager or user notices — e.g., `v1.5.0`'s *"HTTP client is `httpx` everywhere (`requests` removed)"* is worth it because transitive pins move.
- Scale total density to the release: a bugfix-only diff may warrant just the title with zero bullets.

**Step 5 — Print the draft** to the terminal inside a fenced code block so the user can copy it verbatim. Do not write it to a file, do not commit, do not `git tag`.

## Why the phases

Three constraints force Phase B to be serial. Do not re-flatten the phases without addressing them:

- **`uv sync` is exact by default.** It uninstalls anything outside the extras named on that one invocation (`--inexact` opts out). A narrow `uv sync --extra docs` therefore strips `gui`, `globus`, `ptychi`, and `store` — which silently narrows Section 7's pytest collection and inflates Section 3's "modules skipped at load" count. That is why the sync is hoisted to the top of Phase B and names every extra at once.
- **`uv build` and `uv sync` contend on the same project lock.** Section 9 must not overlap Section 6 or the hoisted sync.
- **Section 9's stale-`build/` guard only means something if nothing else is building.** A concurrent build would create the very directory the guard is checking for.

Phase A is safe to fan out because every section in it only reads: filesystem scans, an AST walk, a plugin-registry load, and `--version`/`--help` invocations of the entry points.

One harmless overlap is left alone: Section 5.5 and Section 7's Tier 1 both run `pymarkdown scan` over the same file set, about 2.3s, kept so the two sections stay independently runnable.

## Do not

- Do not `git commit` — release commits are the user's call.
- Do not bump the version — that's a deliberate release step, not drift.
- Do not fix warnings in code you didn't write for the release — surface them and let the user decide scope.
- Do not run this skill against a dirty working tree unless the user explicitly wants to include their WIP in the audit.
