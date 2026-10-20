---
name: add-reconstructor
description: Scaffold a new reconstructor backend library (a *ReconstructorLibrary that exposes Reconstructor / TrainableReconstructor implementations to ProcessingCore). Use when the user says "add a new reconstruction backend", "integrate <library> as a reconstructor", or "add a new ptychi/ptychopinn-style backend".
---

# add-reconstructor

Reconstructor backends are optional dependencies. Each lives under **`src/ptychodus/plugins/<backend>/`** and registers a `*ReconstructorLibrary` **instance** with the plugin registry, whose chooser `ModelCore` hands to `ProcessingCore`. A backend is discovered, never named in a composition root, so adding one touches no existing file except the settings-section table.

A backend holds no settings and reads none. It **declares** what it accepts as a nested pydantic object, and `model/processing/` generates the live parameters, the INI keys and the panel from that declaration.

## Steps

### 1. Directory layout

Create `src/ptychodus/plugins/<backend>/` with:

- `__init__.py` — `register_plugins(registry)`, registering one library instance.
- `core.py` — the `*ReconstructorLibrary` class.
- `options.py` — the pydantic declaration plus a `default_options()` factory.
- `reconstructor.py` — builds the `Reconstructor` / `TrainableReconstructor`. Imported only from `_build()`, never at module scope.
- `_subprocess.py` — the child-side entry points, if the backend touches a GPU.

Write no `settings.py`. The hand-written `*Settings` class is the retired idiom; a new backend declares instead.

### 2. Declare what the backend accepts

In `options.py`, one `BaseModel` per settings group plus a root that composes them. State bounds, titles and descriptions **on the field** — `controller/parameters.py` raises on a missing bound, and the generated panel takes its label and tool tip from `title` and `description`.

```python
class DataOptions(BaseModel):
    batch_size: int = Field(16, ge=1, title='Batch Size:', description='Patterns per step.')


class <Backend>Options(BaseModel):
    data: DataOptions = Field(default_factory=DataOptions)


def default_options() -> <Backend>Options:
    """Return the configuration ptychodus starts a <Backend> run from."""
    return <Backend>Options()
```

Use a `StrEnum` for a fixed set of values; the panel reads the members and builds a combo box, so no separate enumerators class is needed. Pass the declaration as an **instance**, not the class — a field built by a `default_factory` reports its class-level default as undefined.

### 3. The library class

Model on [src/ptychodus/plugins/ptychopinn/core.py](../../src/ptychodus/plugins/ptychopinn/core.py):

```python
from ...api.reconstruct import ReconstructorLibrary, TrainableReconstructor


class <Backend>ReconstructorLibrary(ReconstructorLibrary):
    """What <Backend> offers, holding nothing but the names it is built from."""

    def __init__(self) -> None:
        super().__init__('<logger-name>', '<distribution>', requires=('<module>', 'torch'))

    def _build(self) -> list[TrainableReconstructor]:
        # Imported lazily so this module's own import cost stays small.
        from .reconstructor import build_reconstructor

        return [build_reconstructor(mode) for mode in ('Algorithm1', 'Algorithm2')]
```

Key rules:

- **The constructor takes no arguments and does no work.** Registration happens in every process that loads plugins, including the many that never reconstruct.
- **`requires` names every module the backend cannot work without**, including transitive ones it imports directly. The base resolves them with `find_spec` — without importing, which keeps a GPU framework out of `sys.modules` as [tests/test_no_gpu_context.py](../../tests/test_no_gpu_context.py) enforces.
- **Write no availability logic.** The base calls `_build()` once on first iteration and announces either outcome through the library's own logger: the version on success, `<Backend> not found.` when a required module is absent or when `_build()` raises `ImportError`.
- **A library does not name itself.** It is named where it registers: `display_name` is what a person picks and what the section table and the parameter store are keyed on, `simple_name` is the form a settings file persists, and `distribution` is the PyPI package a version lives under — which is what the backend's own log lines report.

### 4. Build the reconstructor

In `reconstructor.py`, narrow the incoming options once and pass `default_options` to the reconstructor so the settings layer knows what to generate:

```python
def _declared(options: BaseModel) -> <Backend>Options:
    if not isinstance(options, <Backend>Options):
        raise TypeError(f'Expected <Backend>Options, got {type(options).__name__}.')

    return options
```

GPU work belongs in a spawned child: wrap it with `SubprocessReconstructor` (see [src/ptychodus/api/reconstruct.py](../../src/ptychodus/api/reconstruct.py)), passing `default_options=default_options` and the child entry points. Yield one `ReconstructOutput` per iteration so the GUI can stream progress; set `progress_goal` once the run knows its own iteration count.

### 5. Register the plugin

```python
def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        <Backend>ReconstructorLibrary(),
        simple_name='<Backend>',
        display_name='<Backend>',
    )
```

Nothing else is wired. `ModelCore` takes whatever registered, and `ControllerCore` builds every backend's panel from what it declares — there are no hand-written reconstructor panels left.

### 6. Say where the options are persisted

Add `src/ptychodus/model/processing/_<backend>_sections.py` exporting `sections()`, one `OptionsSection` per settings group, and list it in `_SECTIONS_BY_LIBRARY` in [src/ptychodus/model/processing/core.py](../../src/ptychodus/model/processing/core.py) keyed by the name the backend registers under. Entries are asked per reconstructor — `(name, declared_options) -> sections` — so wrap a table that is the same for every reconstructor a backend offers in `_regardless`; answer the arguments only when the groups differ per reconstructor, as pty-chi's do. A backend absent from that table registers no groups and runs on its declared defaults. The table is **transitional**, retiring with the flat INI layout.

Two guard tests cover the table: every name in it must match a registered library, because a rename would otherwise silently drop the backend's settings, and every name must also appear in the golden fixture's group-prefix table, which is what lets that fixture keep checking the rest of the tree when the backend is not installed.

### 7. Optional-dependency plumbing

```toml
[project.optional-dependencies]
<backend> = ["<backend-package>>=<min-version>"]
```

See the comment above `[project.optional-dependencies]` for a backend published on no index. If the install is not a plain `uv sync --extra <backend>`, say so in the backend's `core.py` docstring.

## Verify

- `uv sync --extra <backend>` — install the backend.
- `uv run ptychodus` — the backend appears in the reconstructor list with a generated panel. `--log-level 10` shows the version or the "not found" line.
- Without the extra, the GUI still opens, the backend offers nothing, and no other reconstructor is affected.
- `uv run pytest tests/test_settings_defaults.py` — the golden INI gains the new groups; regenerate the fixture deliberately and review the diff.
- `pre-push` — full CI gate green.
