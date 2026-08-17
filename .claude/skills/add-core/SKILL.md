---
name: add-core
description: Scaffold a new *Core subsystem (settings + parameters + observers + api) and wire it through ModelCore, ViewCore, and ControllerCore. Use when the user says "add a new analysis/processing feature", "add a new model component", or is introducing a subsystem that needs its own settings group and GUI panel.
---

# add-core

Every ptychodus subsystem follows the same shape: a `*Core` class in `src/ptychodus/model/<feature>/` owns its `SettingsRegistry` group, typed parameters, sub-components, and API; `model/core.py::ModelCore` composes it; and the view + controller layers mirror the layout. This skill encodes that shape.

## Layer boundaries (do not cross)

- `src/ptychodus/api/` — pure domain, no dependencies on model/view/controller.
- `src/ptychodus/model/` — logic; depends on `api/` only.
- `src/ptychodus/view/` — PyQt5 widgets; no logic.
- `src/ptychodus/controller/` — mediates view ↔ model.

Run the `check-layers` skill after wiring to confirm you haven't accidentally introduced a model→view import.

## Steps

### 1. Settings

Create `src/ptychodus/model/<feature>/settings.py`. Follow the pattern in [src/ptychodus/model/analysis/settings.py](../../src/ptychodus/model/analysis/settings.py):

```python
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class <Feature>Settings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('<Feature>')
        self._group.add_observer(self)

        # Typed parameters:
        self.num_iterations = self._group.create_integer_parameter(
            'NumberOfIterations', 1000, minimum=1
        )
        # create_real_parameter, create_boolean_parameter, create_string_parameter,
        # create_path_parameter — see api/parametric.py for the full menu.

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
```

Preserve unit suffixes on physical quantities (e.g. `probe_energy_eV`, `pixel_width_m`) — CLAUDE.md notes these are accepted via `# noqa: N815` and reviewers expect them.

### 2. Core

Create `src/ptychodus/model/<feature>/core.py`:

```python
class <Feature>Core:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        # ... other dependencies from ModelCore (repositories, other Cores)
    ) -> None:
        self.settings = <Feature>Settings(settings_registry)
        # ... construct sub-components, wire observers
```

For an example that composes many sub-components + `VisualizationEngine`s, see [src/ptychodus/model/analysis/core.py](../../src/ptychodus/model/analysis/core.py).

### 3. Compose in ModelCore

In [src/ptychodus/model/core.py](../../src/ptychodus/model/core.py), import your `<Feature>Core` and construct it in `ModelCore.__init__` in the correct dependency order — after everything it needs, before anything that needs it. `ModelCore` is the single composition root; do not construct components anywhere else.

### 4. View

Create `src/ptychodus/view/<feature>/` mirroring the model layout. PyQt5 widgets only — no logic, no imports from `model/` or `controller/`.

### 5. Controller

Create `src/ptychodus/controller/<feature>/` with the `*ViewController` that bridges the view widgets to `<Feature>Core`.

### 6. Wire into ViewCore and ControllerCore

`ViewCore` and `ControllerCore` (in `src/ptychodus/view/core.py` and `src/ptychodus/controller/core.py`) are composition roots. Register the new panel in both. **The navigation toolbar order in `ViewCore` is the source of truth for the left/right stacked-panel indexes** (per CLAUDE.md) — pick the toolbar position deliberately and ensure `ControllerCore` uses the matching stacked-widget index.

## PluginChooser binding for settings-driven selection

If your feature has a "current selected implementation" (e.g. current file reader), bind a `PluginChooser` to a `StringParameter` in your settings:

```python
self.file_reader_chooser.synchronize_with_parameter(self.settings.file_type)
```

`synchronize_with_parameter` in [src/ptychodus/api/plugins.py](../../src/ptychodus/api/plugins.py) sets up the two-way binding: parameter value ↔ current plugin.

## Verify

- `check-layers` — no boundary violations.
- `pre-push` — full CI gate green.
- Launch the GUI (`uv run ptychodus`) and confirm the new panel shows up in the correct toolbar position.
