---
name: add-reconstructor
description: Scaffold a new reconstruction library (a *ReconstructorLibrary that exposes Reconstructor / TrainableReconstructor implementations to ProcessingCore). Use when the user says "add a new reconstruction library", "integrate <library> as a reconstructor", or "add a new ptychi/ptychopinn-style library".
---

# add-reconstructor

Reconstruction libraries are optional dependencies. Each lives under **`src/ptychodus/plugins/<library>/`** and registers a `*ReconstructorLibrary` **instance** with the plugin registry, whose chooser `ModelCore` hands to `ProcessingCore`. A library is discovered, never named in a composition root, so adding one touches no existing file except the settings-section table.

A library holds no settings and reads none. It **declares** what it accepts as a nested pydantic object, and `model/processing/` generates the live parameters, the INI keys and the panel from that declaration.

## Steps

### 1. Directory layout

Create `src/ptychodus/plugins/<library>/` with:

- `__init__.py` — `register_plugins(registry)`, registering one library instance.
- `core.py` — the `*ReconstructorLibrary` class.
- `options.py` — the pydantic declaration plus a `default_options()` factory.
- `reconstructor.py` — the `Reconstructor` / `TrainableReconstructor` class and the computation it runs. Imported only from `_build()`, never at module scope.

Write no `settings.py`. The hand-written `*Settings` class is the retired idiom; a new library declares instead.

### 2. Declare what the library accepts

In `options.py`, one `BaseModel` per settings group plus a root that composes them. State bounds, titles and descriptions **on the field** — `controller/parameters.py` raises on a missing bound, and the generated panel takes its label and tool tip from `title` and `description`.

```python
class DataOptions(BaseModel):
    batch_size: int = Field(16, ge=1, title='Batch Size:', description='Patterns per step.')


class <Library>Options(BaseModel):
    data: DataOptions = Field(default_factory=DataOptions)


def default_options() -> <Library>Options:
    """Return the configuration ptychodus starts a <Library> run from."""
    return <Library>Options()
```

Use a `StrEnum` for a fixed set of values; the panel reads the members and builds a combo box, so no separate enumerators class is needed. Pass the declaration as an **instance**, not the class — a field built by a `default_factory` reports its class-level default as undefined.

### 3. The library class

Model on [src/ptychodus/plugins/ptychopinn/core.py](../../../src/ptychodus/plugins/ptychopinn/core.py):

```python
from ...api.reconstruct import ReconstructorLibrary, TrainableReconstructor


class <Library>ReconstructorLibrary(ReconstructorLibrary):
    """What <Library> offers, holding nothing but the names it is built from."""

    def __init__(self) -> None:
        super().__init__('<logger-name>', '<distribution>', requires=('<module>', 'torch'))

    def _build(self) -> list[TrainableReconstructor]:
        # Imported lazily so this module's own import cost stays small.
        from .reconstructor import <Library>Reconstructor

        return [<Library>Reconstructor(mode) for mode in ('Algorithm1', 'Algorithm2')]
```

Key rules:

- **The constructor takes no arguments and does no work.** Registration happens in every process that loads plugins, including the many that never reconstruct.
- **`requires` names every module the library cannot work without**, including transitive ones it imports directly. The base resolves them with `find_spec` — without importing, which keeps a GPU framework out of `sys.modules` as [tests/test_no_gpu_context.py](../../../tests/test_no_gpu_context.py) enforces.
- **Write no availability logic.** The base calls `_build()` once on first iteration and announces either outcome through the library's own logger: the version on success, `<Library> not found.` when a required module is absent or when `_build()` raises `ImportError`.
- **A library does not name itself.** It is named where it registers: `display_name` is what a person picks and what the section table and the parameter store are keyed on, `simple_name` is the form a settings file persists, and `distribution` is the PyPI package a version lives under — which is what the library's own log lines report.

### 4. Write the reconstructor

In `reconstructor.py`, a module-level class whose `reconstruct` (and, for a trainable reconstructor, `train`) is an ordinary generator. It knows nothing about processes: the application pickles the object and its arguments to a spawned child restricted to the selected devices, and iterates the generator there. That asks three things of the class:

- **It pickles.** Module-level class, plain attributes — no closures, lambdas or open handles. [tests/test_plugin_pickling.py](../../../tests/test_plugin_pickling.py) round-trips every registered reconstructor.
- **`reconstruct` and `train` leave `self` unchanged and import the framework inside the method.** What they change about themselves is lost with the child; only what they yield comes back. Ask torch which device is visible from inside the method too — the selection has already been applied.
- **Every other method imports no GPU framework.** `get_default_options`, `load_model` and `save_model` run in the application process. `load_model` records the path (and may unpack an archive); the model is loaded inside `reconstruct`.

```python
from ptychodus.api.options import expect_options


class <Library>Reconstructor(Reconstructor):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_default_options(self) -> BaseModel:
        return default_options()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        declared = expect_options(options, <Library>Options)
        import torch  # here, never at module scope
        ...
        yield ReconstructOutput(product=product, progress=epoch, progress_goal=num_epochs)
```

Validate the input at the top of `reconstruct`, before reaching for the framework, so a bad product fails as soon as possible. Yield one `ReconstructOutput` per iteration so the GUI can stream progress; set `progress_goal` once the run knows its own iteration count.

A trainable reconstructor's `train` writes into the output directory and names the model it wrote in its last `TrainOutput.trained_model`, in the form its `load_model` accepts. It does not load it: whoever asked for the training does. [src/ptychodus/plugins/ptychopinn/reconstructor.py](../../../src/ptychodus/plugins/ptychopinn/reconstructor.py) is the worked example.

### 5. Reconcile conventions at the boundary

Convert the product into the library's gauge and frame on the way in, and apply the inverse to **every** yielded `ReconstructOutput`, not just the last — a streaming library yields many. Two things to reconcile:

- **Scale.** ptychodus stores a probe in photons per pixel, so `sum|P|^2` is the photons incident per exposure. Measure the factor by pushing the probe and the object patches beneath it through the library's *own* forward propagator and comparing with the mean detector sum; do not derive it from the library's FFT normalization, which can vary with geometry. Keep the factor and divide by it on readback, so the round trip is exact rather than a second measurement of a probe the library has since moved. Scale only the probe: the object rides through untouched, which leaves the product the data constrains unchanged.
- **Coordinates.** ptychodus stores probe positions in meters; most libraries want object pixels. Map with the object geometry's `map_probe_positions_to_object_x_px` / `_y_px` and invert with `map_coordinates_object_to_probe`.

Both belong in pure functions beside the options alignment, so they are testable without the library installed and cannot drift from it. [src/ptychodus/plugins/ptychi/task.py](../../../src/ptychodus/plugins/ptychi/task.py) is the worked example: `align_probe_gauge_with_ptychi` measures, `apply_probe_gauge_to_options` restates the probe-power target in the same gauge, and `reconstruct_with_ptychi` applies and undoes it.

### 6. Register the plugin

```python
def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        <Library>ReconstructorLibrary(),
        simple_name='<Library>',
        display_name='<Library>',
    )
```

Nothing else is wired. `ModelCore` takes whatever registered, and `ControllerCore` builds every library's panel from what it declares — there are no hand-written reconstructor panels left.

### 7. Say where the options are persisted

Add `src/ptychodus/model/processing/_<library>_sections.py` exporting `sections()`, one `OptionsSection` per settings group, and list it in `_SECTIONS_BY_LIBRARY` in [src/ptychodus/model/processing/core.py](../../../src/ptychodus/model/processing/core.py) keyed by the name the library registers under. Entries are asked per reconstructor — `(name, declared_options) -> sections` — so wrap a table that is the same for every reconstructor a library offers in `_regardless`; answer the arguments only when the groups differ per reconstructor, as pty-chi's do. A library absent from that table registers no groups and runs on its declared defaults. The table is **transitional**, retiring with the flat INI layout.

Two guard tests cover the table: every name in it must match a registered library, because a rename would otherwise silently drop the library's settings, and every name must also appear in the golden fixture's group-prefix table, which is what lets that fixture keep checking the rest of the tree when the library is not installed.

### 8. Optional-dependency plumbing

```toml
[project.optional-dependencies]
<library> = ["<library-package>>=<min-version>"]
```

See the comment above `[project.optional-dependencies]` for a library published on no index. If the install is not a plain `uv sync --extra <library>`, say so in the library's `core.py` docstring.

## Verify

- `uv sync --extra <library>` — install the library.
- `uv run ptychodus` — the library appears in the reconstructor list with a generated panel. `--log-level 10` shows the version or the "not found" line.
- Without the extra, the GUI still opens, the library offers nothing, and no other reconstructor is affected.
- `uv run pytest tests/test_settings_tables.py` — the library's name in the section table matches the name it registers under.
- Reconstruct, save the product, then warm-start a *different* library from it: probe power and positions must come through comparable, which is the mixing invariant.
- `pre-push` — full CI gate green.
