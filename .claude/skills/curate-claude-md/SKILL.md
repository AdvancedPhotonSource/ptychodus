---
name: curate-claude-md
description: Audit CLAUDE.md and .claude/skills/*/SKILL.md for drift against the source tree, for bloat, and for contradictions — stale symbol names, dead packages, closed enumerations that lost a member, embedded audit commands whose stated result changed, volatile phrasing, config scalars, local-artifact leakage, and oversized rationale. Proposes a fix per finding and applies none without confirmation. Use when the user says "check CLAUDE.md", "audit CLAUDE.md", "is CLAUDE.md still accurate", "trim CLAUDE.md", after a refactor that renames or moves symbols, or from the pre-release gate.
---

# curate-claude-md

CLAUDE.md and every `SKILL.md` description load into context on every turn, so each word costs tokens on every request and each stale claim misleads every session until someone notices. This skill audits both, reports `file:line` per finding with a proposed fix, and applies nothing without asking.

Scope is [CLAUDE.md](../../CLAUDE.md) and [.claude/skills/](../../.claude/skills/). `README.md` and `docs/source/` belong to `pre-release` §4a, §5 and §5.5 — do not duplicate those checks here.

## What belongs in CLAUDE.md

- **Invariants, architecture, conventions** — the non-obvious knowledge that cannot be derived by reading code. Nothing else.
- **Behavioral rules earn their place; facts do not.** A rule belongs here when Claude keeps getting it wrong. A fact findable in `pyproject.toml`, `git`, a linter config, a README, or a skill description does not.
- **No exhaustive enumerations.** They drift. Representative sample plus "and others", so an omission is not a defect by construction.
- **Prefer claims whose truth survives the most likely change.** "nine of them" dies the next time a beamline is added; "all but `reconstruct_pear.py` and `reconstruct_batch_lamni.py`" does not.
- **Make rules grep-auditable and embed the audit command in the rule.** A rule carrying its own one-line check can be verified forever; one that cannot is unenforceable. This is the single highest-value habit in the file.
- **Name roles, not instances.** When a rule needs an exemplar, expect the exemplar to be renamed before the rule changes.
- **Never surface deliberately undocumented on-disk artifacts** — `tike`, `ptychonn`, `ptycho_fm`, repo-root `.ini` files, `install.sh`, `TODO`.

## Steps

Run every check from the repo root. Report `OK` for each clean check and `file:line` for each finding. The order is by measured hit rate: **names drift; paths and config scalars do not.**

### 1. Symbol existence

Every backticked identifier must exist. Third-party and stdlib names (`find_spec`, `collect_ignore`, `importorskip`) surface here as unresolved — triage them, do not "fix" them.

```sh
grep -o '`[^`]*`' CLAUDE.md | tr -d '`' \
  | grep -oE '^([A-Z][a-z][A-Za-z0-9]*|[a-z][a-z0-9]*_[a-z0-9_]*|[A-Z][A-Z0-9_]{3,})$' \
  | grep -vxE 'None|True|False' | sort -u \
  | while read -r s; do
      grep -rqE "^[[:space:]]*(class|def) ${s}\b|^${s}[[:space:]]*[:=]" --include='*.py' src/ \
        || echo "UNRESOLVED  $s"
    done
```

### 2. `Class.method` lives in that class

A bare `def <method>` grep is not enough — it passes when the method exists on some unrelated class, which is exactly how a renamed exemplar survives review. Accept a dataclass field as well as a method, since CLAUDE.md cites both in `Owner.member` form.

```sh
grep -o '`[^`]*`' CLAUDE.md | tr -d '`' | sed 's/::/./' \
  | grep -oE '^[A-Z][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*$' | sort -u \
  | while IFS=. read -r cls meth; do
      files=$(grep -rlE "^class ${cls}\b" --include='*.py' src/)
      [ -z "$files" ] && { echo "MISSING class   ${cls}.${meth}"; continue; }
      echo "$files" | xargs awk -v c="$cls" -v m="$meth" '
        /^class /                 { inc = ($0 ~ "^class "c"[ (:]") }
        inc && ($0 ~ "def "m"\\(" || $0 ~ "^ +"m" *:")  { found = 1 }
        END { exit !found }' || echo "MISSING method  ${cls}.${meth}"
    done
```

### 3. Dead packages

Use git, never the filesystem: a package deleted from the index can linger on disk as a stale `__pycache__/`, so `test -e` reports it alive.

```sh
grep -o '`[^`]*/`' CLAUDE.md | tr -d '`' | grep -vE '^/|[ <{*]' | sort -u | while read -r d; do
  for c in "$d" "src/ptychodus/$d" "src/ptychodus/model/$d" "src/ptychodus/api/$d" "src/$d"; do
    git ls-files "$c" 2>/dev/null | grep -q . && continue 2
  done
  echo "NO TRACKED FILES  $d"
done
```

A bare `foo/` is resolved against the trees it could plausibly name, `model/` included — most subpackage references in CLAUDE.md are written relative to their parent. Expect one standing exception: `ui/dist/` is a gitignored build artifact, which CLAUDE.md says itself.

### 4. Closed enumerations

Produce the set difference; a human decides whether the list is closed. A list ending "and others" is open and an omission is fine. A list that reads as exhaustive and is not is a defect — **prefer reopening the list over appending to it.**

```sh
grep -rlE 'class [A-Za-z]+ReconstructorLibrary' src/ptychodus/model/*/*.py | xargs -n1 dirname | xargs -n1 basename | sort -u
ls -d src/ptychodus/model/*/ | xargs -n1 basename | sort
ls src/ptychodus/api/*.py | xargs -n1 basename | sed 's/\.py$//' | sort
grep -oE "^    [A-Z_]+ = " src/ptychodus/api/io.py | tr -d ' =' | sort -u
```

### 5. Universal claims

"Each X exposes a Y" is the most brittle sentence shape in the file — one new sibling falsifies it silently, and the exceptions are often *inside the list the sentence itself gives*.

```sh
for d in src/ptychodus/model/*/; do
  git ls-files "$d" | grep -q . || continue          # skip untracked and dead dirs
  grep -qE '^class [A-Za-z]*Core\b' "$d/core.py" 2>/dev/null || echo "NO *Core  $(basename "$d")"
done
```

Expected: only subpackages CLAUDE.md documents as exceptions. Check the named members first — a universal contradicted by an item in its own enumeration reads as authoritative and is the hardest kind to notice.

### 6. Embedded audit commands

Any rule that quotes a `grep`/`rg` incantation **and** a claim about its output: run it, compare. This is the only self-verifying claim class in the file, and it drifts silently because nobody re-runs it.

```sh
rg -Un --pcre2 'from ptychodus\.api\.[\w.]+ import (\([^)]*\b_|_)' src/ tests/ scripts/
```

Prefer rewriting a drifted claim so it states the rule and the command but **not** the current hit count.

### 7. Volatile phrasing

```sh
grep -nEoi '\b(currently|the only|only current|so far|as of now|at present|for now|at the moment)\b[^.;—]{0,50}|\b(one|two|three|four|five|six|seven|eight|nine|ten) of them|~?[0-9]+-line' CLAUDE.md
```

Re-verify each hit, then rewrite it to survive the most likely next change. Match `the only` rather than bare `only` — unqualified "only" is ordinary English and buries the three or four real hits under a dozen.

### 8. Config scalars

Cheapest check in the file and historically the most stable — but free to run.

```sh
python3 -c "
import tomllib; d = tomllib.load(open('pyproject.toml','rb'))
print('requires-python', d['project']['requires-python'])
print('extras   ', sorted(d['project']['optional-dependencies']))
print('scripts  ', sorted(d['project']['scripts']))
print('ruff     ', {k: v for k, v in d['tool']['ruff'].items() if k in ('line-length','target-version')})
"
grep -hoE 'Dockerfile\.[a-z0-9]+' CLAUDE.md | sort -u | while read -r f; do
  [ -f "containers/$f" ] || echo "MISSING  containers/$f"
done
```

Every `[project.scripts]` name should be documented or deliberately omitted, **and** every script CLAUDE.md names must still exist — check both directions; a command block naming a deleted script is the easier error to miss.

### 9. Local-artifact leakage

Every referenced repo path must be tracked. This one check subsumes the standing "no `install.sh`, no `TODO`, no repo-root `.ini`" rules, because all three are untracked by definition.

```sh
grep -o '\]([^)]*)' CLAUDE.md | sed 's/^](//; s/)$//' | sort -u | while read -r p; do
  git ls-files "$p" | grep -q . || echo "UNTRACKED  $p"
done
grep -nEi '\b(tike|ptychonn|ptycho_fm|install\.sh)\b' CLAUDE.md
```

`settings.ini` as a `StandardFileLayout` member is an I/O contract, not a repo-root file — not a finding.

### 10. Contradictions

Judgment, not grep. Check three directions:

- Within CLAUDE.md, and between CLAUDE.md and any `SKILL.md`.
- **Between a skill's frontmatter and its own body** — the description is written once and rarely re-read against the steps beneath it.
- Against the `feedback_*` memories. A memory that records *the state of the tree* rather than *a decision about it* becomes a false instruction once the tree moves, so **the memory can be the stale party** — correct it there rather than editing CLAUDE.md to match.

### 11. Bloat

```sh
awk '/^## Conventions/,/^## Repository/' CLAUDE.md \
  | awk '/^- /{n = split($0, a, " "); if (n > 80) printf "%4d words  %.70s...\n", n, $0}'
wc -w CLAUDE.md
```

Soft caps: ~80 words per bullet, ~2400 words total. Over either, **compress rationale — never delete a rule.** Each convention was added because Claude violated it, so dropping one regresses behavior. Cut instead: prose restating what a linter or `pyproject.toml` already enforces, narrative explaining how the code came to be this way, and any procedure a skill already owns. Keep the rule, its audit command, and one exemplar.

## Admitting a new rule

Before adding anything to CLAUDE.md:

1. Is it behavioral, or a fact derivable from the tree? Facts stay out.
2. Can a one-line grep audit it? If so, embed that command in the rule.
3. Does it duplicate a skill, `pyproject.toml`, or a linter? Then point at that instead.
4. Write it in the tone and length of its neighbours.
5. Offer options with a recommendation and let the user choose the rule.
6. Ship the CLAUDE.md edit in the same commit as the code change that motivated it.

## Cadence

Run after any refactor that renames or moves symbols, when `git log <last-CLAUDE.md-commit>..HEAD -- src/` has grown large, and from `pre-release`. The file ratchets: feature commits append clauses and nothing prunes, so the compression pass in check 11 needs a deliberate occasion.

```sh
git log -1 --format='%h %ad' --date=short -- CLAUDE.md
git log --oneline "$(git log -1 --format=%H -- CLAUDE.md)..HEAD" -- src/ | wc -l
```

## Reporting

- Clean: report "CLAUDE.md and skills OK" with the word count, and stop.
- Otherwise report each finding as `file:line`, what the tree actually says, and a proposed replacement.
- Walk the findings one at a time and ask before each edit.

## Do not

- Do not auto-fix. A stale exemplar needs a judgment call about which replacement best illustrates the rule.
- Do not delete a convention to meet the word budget.
- Do not complete a drifted enumeration by appending members; reopen it instead.
- Do not add a rule the user did not choose.
- Never commit.
