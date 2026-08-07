---
name: pre-release
description: Run a comprehensive pre-release verification of the ptychodus repository — API docs coverage, minimal docstring presence, reader-plugin docs, README/CLAUDE.md/pyproject.toml consistency, install-instruction freshness, Markdown hygiene, zero-warning Sphinx build, full CI gate, and entry-point smoke tests. Report every finding and offer per-issue fixes. Use when the user says "pre-release check", "release audit", "before we cut a release", or "verify the repo is release-ready".
---

# pre-release

Comprehensive gate for cutting a release. Runs eight verification sections in order, reports pass/fail for each, and — for every failure — proposes a specific fix and asks the user before applying it. **Never fixes silently. Never commits.**

## How to run the check

Execute the sections in order. After each, capture pass/fail and any findings. At the end, print the summary table (last section). Only then walk the user through fixing failures one at a time.

---

### Section 1 — API modules present in `docs/source/api.md`

For every `src/ptychodus/api/*.py` file (excluding `_*.py` private modules), confirm `docs/source/api.md` contains a matching `.. automodule:: ptychodus.api.<stem>` directive.

```sh
# Modules that should be documented:
ls src/ptychodus/api/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -v '^_' | grep -v '^__' | sort

# Modules currently documented:
grep -oE '\.\. automodule:: ptychodus\.api\.[a-z_]+' docs/source/api.md | sed 's|.*ptychodus\.api\.||' | sort
```

Compute the diff. `PASS` if empty. `FAIL` with the list of missing module names.

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

`docs/source/readers.md` is a curated bullet list, not a 1:1 file mapping. Cross-check by keyword: for every distinct beamline/format hint in `src/ptychodus/plugins/`, ensure the readers doc mentions it.

Keyword map (edit as new plugins land):

| Plugin filename substring | Expected mention in readers.md |
| --- | --- |
| `aps02id_` | "2-ID-D Bionanoprobe" or "2-ID-D Microprobe" or "2-ID-E Microprobe" |
| `aps04id_polar_` | "4-ID" and "Polar" (Polarization Modulation Spectroscopy) |
| `aps09id_cssi_` | "9-ID" and "CSSI" |
| `aps12id_` | "12-ID" and "SAXS" |
| `aps19id_isn_` | "19-ID" and "ISN" |
| `aps31id_lynx_` | "31-ID" and "LYNX" |
| `aps33id_velociprobe` | "33-ID" and "Velociprobe" |
| `lcls_` | "LCLS" |
| `max_iv_nanomax_` | "MAX IV" or "NanoMAX" |
| `fold_slice_` | "fold_slice" |
| `cxi_` | "CXI" or "*.cxi" |
| `csv_` | "CSV" or "Comma-Separated" |
| `mda_` | "MDA" or "*.mda" |
| `npy_` | "NumPy" or "*.npy" |
| `delimited_position_` | "Space-Separated" or "*.txt" |

```sh
# List all plugin file stems:
ls src/ptychodus/plugins/*.py | xargs -n1 basename | sed 's/\.py$//'

# For each keyword above, verify readers.md contains the expected mention:
for kw in "2-ID-D Bionanoprobe" "4-ID" "9-ID" "12-ID" "19-ID" "31-ID" "33-ID" "LCLS" "NanoMAX" "fold_slice" "CXI" "CSV" "MDA" "NumPy" "Space-Separated"; do
    grep -q "$kw" docs/source/readers.md || echo "MISSING: $kw"
done
```

Also flag *the reverse*: any bullet in `readers.md` whose beamline has no matching plugin file — that indicates a stale doc entry.

`PASS` if every plugin has a doc mention and every doc mention has a plugin. `FAIL` with lists of orphans in either direction.

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

**5a. Dockerfile variants referenced still exist.** Every Dockerfile name mentioned in `docs/source/getting_started.md` and `CLAUDE.md` should be a real file at the repo root:

```sh
grep -hoE 'Dockerfile\.[a-z]+' docs/source/getting_started.md CLAUDE.md | sort -u | while read f; do
    [ -f "$f" ] || echo "MISSING: $f"
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
uv sync --extra docs
make -C docs clean
make -C docs html SPHINXOPTS="-W --keep-going"
```

`PASS` if exit code 0. `FAIL` on any warning or error.

**Suggested fix**: hand the first warning to the user. Common causes: undocumented cross-references, autodoc import failures (usually a missing extra — retry with `uv sync --extra docs --extra ptychi --extra globus --extra gui --extra store` if autodoc can't import a module), duplicate labels. Not auto-fixable.

*Note*: `-W` is strict, and the prose sources are expected to build clean — treat any warning originating in `docs/source/*.md` as a regression introduced by the change under review. Warnings raised from `src/ptychodus/api/*.py` docstrings (reStructuredText syntax errors surfaced by autodoc) are a known pre-existing backlog; report the count and the first offender, but do not fold fixing them into the release gate.

---

### Section 7 — Full CI gate (chain `pre-push`)

Invoke the `pre-push` skill. All four steps (ruff check, ruff format --check, mypy, pytest) must be green.

`PASS` if `pre-push` finishes clean. `FAIL` on any failing step; hand the failure back to the user without attempting fixes here (the `pre-push` skill knows how to offer format fixes; deeper fixes belong outside the release gate).

---

### Section 8 — Entry-point smoke test

For each name in `[project.scripts]`, run `uv run <script> --version`; if the command doesn't implement `--version`, fall back to `--help`. Exit 0 counts as PASS; non-zero or import error counts as FAIL.

```sh
uv run python -c "
import tomllib
with open('pyproject.toml','rb') as f:
    print('\n'.join(tomllib.load(f)['project']['scripts']))
" | while read script; do
    if uv run "$script" --version >/dev/null 2>&1; then
        echo "PASS: $script"
    elif uv run "$script" --help >/dev/null 2>&1; then
        echo "PASS: $script (--help)"
    else
        echo "FAIL: $script"
    fi
done
```

`PASS` if every script exits 0. `FAIL` lists the broken entry points — usually caused by import-time errors introduced by an unrelated change.

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
```

Then, and only then, walk through each `FAIL` with the user: state the finding, propose the fix, ask "apply it?" per item, and Edit/Write only after they confirm.

## Do not

- Do not `git commit` — release commits are the user's call.
- Do not bump the version — that's a deliberate release step, not drift.
- Do not fix warnings in code you didn't write for the release — surface them and let the user decide scope.
- Do not run this skill against a dirty working tree unless the user explicitly wants to include their WIP in the audit.
