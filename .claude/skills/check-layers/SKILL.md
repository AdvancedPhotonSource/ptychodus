---
name: check-layers
description: Verify api/model/view+controller layer boundary compliance across the ptychodus source tree — no upward imports (api must not depend on model/view/controller; model must not depend on view/controller; ptychodus_store must not depend on ptychodus.model or ptychodus.view). Use when refactoring, before merging structural changes, or when the user asks to audit the architecture.
---

# check-layers

CLAUDE.md documents a strict three-layer separation:

- `src/ptychodus/api/` — pure domain, depends on nothing else in ptychodus.
- `src/ptychodus/model/` — application logic, depends only on `api/`.
- `src/ptychodus/view/` and `src/ptychodus/controller/` — GUI; may depend on `api/` and `model/`.
- `src/ptychodus_store/` — separate package; reads from `ptychodus.api` only, never from `ptychodus.model` or `ptychodus.view`.

This skill greps for violations and reports file:line for anything out of place.

## Steps

Run each check. Report `OK` for each clean rule and file:line for each violation.

### 1. api/ must not import model/, view/, or controller/

```sh
grep -rn --include='*.py' -E "^(from|import)[[:space:]]+ptychodus\.(model|view|controller)" src/ptychodus/api/
```

Expected: no output.

### 2. model/ must not import view/ or controller/

```sh
grep -rn --include='*.py' -E "^(from|import)[[:space:]]+ptychodus\.(view|controller)" src/ptychodus/model/
```

Expected: no output.

### 3. api/ and model/ must not import PyQt5

PyQt5 is a view-layer concern. Its presence in api/ or model/ is a smell even if no `ptychodus.view.*` import is used.

```sh
grep -rn --include='*.py' -E "^(from|import)[[:space:]]+PyQt5" src/ptychodus/api/ src/ptychodus/model/
```

Expected: no output. If found, the code should move to `view/` or `controller/`.

### 4. ptychodus_store/ must not import ptychodus.model or ptychodus.view

```sh
grep -rn --include='*.py' -E "^(from|import)[[:space:]]+ptychodus\.(model|view|controller)" src/ptychodus_store/
```

Expected: no output.

### 5. Circular-check inside model/ (informational)

Not a hard rule, but flag any `model/<A>/` file importing from `model/<B>/` where `B` is composed *after* `A` in `ModelCore.__init__` — that's an ordering bug waiting to happen.

```sh
grep -rn --include='*.py' -E "^from[[:space:]]+ptychodus\.model\.[a-z_]+[[:space:]]+import" src/ptychodus/model/
```

Report the cross-subpackage imports and ask the user to verify the composition order in `src/ptychodus/model/core.py` still respects them.

## Reporting

- If every check is clean: report "All layer boundaries OK" and stop.
- If any check finds violations: report each violation as a bullet with `file:line` and the offending line's content. Suggest the correct home for the code (e.g. "Move `X` from `model/foo/bar.py:42` to `controller/foo/bar.py` — this uses `PyQt5.QtCore.QObject` which is view-layer").
- Do not auto-fix. Layer violations often reflect a design choice the user needs to make (e.g. move code, extract an interface into api/, invert a dependency).

## When this catches things

- After a big refactor that moves code between layers.
- After adding a new subsystem via the `add-core` skill.
- Before merging structural PRs.
- When a new contributor's first PR touches multiple layers.
