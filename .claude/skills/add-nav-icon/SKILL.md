---
name: add-nav-icon
description: Add a navigation-bar icon to ptychodus in both the PyQt GUI and the ptychodus_store web UI. Covers SVG placement, Qt resource compilation with the manual typing fix that pyrcc5 clobbers, ViewCore wiring, nav.ts wiring, and the tsc rebuild. Use when the user says "add a nav icon", "add a toolbar icon", "wire up an icon for the new X panel", or is following up on add-core / a new panel.
---

# add-nav-icon

Ptychodus has two navigation surfaces that share icon assets: the PyQt5 toolbar in `ViewCore` and the browser nav bar in `ptychodus_store`. SVGs live once in [`src/ptychodus_store/ui/icons/`](../../src/ptychodus_store/ui/icons/); the PyQt side pulls them in via a Qt resource file compiled to `resources.py`, and the web UI pulls them straight from the FastAPI static mount. This skill wires a new icon through both.

## Step 1 — Add the SVG

Drop the SVG into [`src/ptychodus_store/ui/icons/`](../../src/ptychodus_store/ui/icons/). Do not put icons anywhere else — the `.qrc` file uses a relative path (`../../ptychodus_store/ui/icons/…`) to reach exactly this directory.

- Kebab-case filename (e.g. `my-icon.svg`).
- `fill="currentColor"` so CSS themes work.
- Font Awesome 7.1.0 (CC BY 4.0) is the house style. See [`icons/Font-Awesome-LICENSE.txt`](../../src/ptychodus_store/ui/icons/Font-Awesome-LICENSE.txt) for attribution and [`icons/README.md`](../../src/ptychodus_store/ui/icons/README.md) for the update procedure.

## Step 2 — Register in the Qt resource file

Add one line to [`src/ptychodus/view/resources.qrc`](../../src/ptychodus/view/resources.qrc), keeping the list alphabetized by `alias`:

```xml
<file alias="my-feature">../../ptychodus_store/ui/icons/my-icon.svg</file>
```

The `alias` is what code references as `:/icons/my-feature`. Convention: alias matches the panel concept, not the filename (e.g. `atom.svg` is aliased `fluorescence`).

## Step 3 — Regenerate `resources.py` and reapply the typing fix

`resources.py` is auto-generated binary. Regenerate from inside the venv:

```sh
cd src/ptychodus/view
uv run pyrcc5 -o resources.py resources.qrc
```

Fresh `pyrcc5` output writes the two module functions **unannotated and camelCase**, which fails both `mypy` (missing return type) and `ruff` (`N802`, function name should be lowercase). Reapply the manual typing fix — patch the tail of the file so the two functions look like this:

```python
def qInitResources() -> None:  # noqa: N802
    QtCore.qRegisterResourceData(
        rcc_version, qt_resource_struct, qt_resource_name, qt_resource_data
    )


def qCleanupResources() -> None:  # noqa: N802
    QtCore.qUnregisterResourceData(
        rcc_version, qt_resource_struct, qt_resource_name, qt_resource_data
    )


qInitResources()
```

Two edits per function: append `-> None` to the signature and `# noqa: N802` to the same line. Leave the `WARNING! All changes made in this file will be lost!` header — the noqa comments and return annotations are the only manual patches, and this pattern must survive every regeneration. (The import site at [`view/core.py:30`](../../src/ptychodus/view/core.py#L30) uses `from . import resources  # noqa` for the same reason — it's imported for side effects.)

## Step 4 — Wire it into `ViewCore`

Add a call to `self.navigation.add_panel(...)` in [`src/ptychodus/view/core.py`](../../src/ptychodus/view/core.py) `ViewCore.__init__`. Order matters — **the sequence of `add_panel` calls is the source of truth for the left/right stacked-panel indexes** (per CLAUDE.md). Mirror the fluorescence pattern at [`view/core.py:275-280`](../../src/ptychodus/view/core.py#L275-L280):

```python
self.my_feature_view = MyFeatureView()
self.my_feature_image_view = MyFeatureImageView()
self.my_feature_action = self.navigation.add_panel(
    QIcon(':/icons/my-feature'),
    'My Feature',
    left=self.my_feature_view,
    right=self.my_feature_image_view,
)
```

If the new panel belongs nested under a parent (Products or Processing today), add its action to the existing `add_subview_group` call at [`view/core.py:336-352`](../../src/ptychodus/view/core.py#L336-L352). Otherwise it renders as a top-level toolbar button.

`ControllerCore` uses stacked-widget indexes matching `ViewCore`'s `add_panel` order — inserting a panel in the middle shifts every downstream index. Prefer appending, or run the app afterward and confirm nothing underneath shifted.

## Step 5 — Wire it into the web UI

Add one entry to the `NAV` array in [`src/ptychodus_store/ui/src/nav.ts`](../../src/ptychodus_store/ui/src/nav.ts):

```typescript
{ route: 'my-feature', label: 'My Label', icon: 'my-icon.svg' },
```

- `route` — page identifier used by the front-end router.
- `label` — tooltip / text under the button.
- `icon` — **filename only**; the render code prefixes `/ui/icons/`.

No backend router change is needed. FastAPI already serves `src/ptychodus_store/ui/` at `/ui/` via the `StaticFiles` mount in [`src/ptychodus_store/app.py`](../../src/ptychodus_store/app.py), so a new SVG in `ui/icons/` is automatically reachable at `/ui/icons/my-icon.svg`.

## Step 6 — Rebuild the TypeScript

```sh
cd src/ptychodus_store/ui
tsc
```

No bundler, no framework — plain `tsc` compiles `src/**/*.ts` into `dist/` (see `tsconfig.json`). Restart the store server so the fresh compiled JS is loaded; `tsc --watch` is fine for iteration.

## Verify

- **PyQt:** `uv run ptychodus` — the new icon appears at the correct toolbar position (top-level or nested), highlights on click, and swaps in the associated left/right panels.
- **Web UI:** run the `store-dev` skill, load the browser UI, confirm the icon renders in the nav bar (network tab should show a 200 for `/ui/icons/my-icon.svg`).
- **CI gate:** run the `pre-push` skill. If `pyrcc5` wiped the typing fix, `ruff` will fail with `N802` on `qInitResources` / `qCleanupResources` — reapply Step 3 and re-run.
