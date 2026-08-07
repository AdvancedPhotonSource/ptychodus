---
name: add-reconstructor
description: Scaffold a new reconstructor backend library (a *ReconstructorLibrary that exposes Reconstructor / TrainableReconstructor implementations to ProcessingCore). Use when the user says "add a new reconstruction backend", "integrate <library> as a reconstructor", or "add a new ptychi/ptychopinn-style backend".
---

# add-reconstructor

Reconstructor backends are optional dependencies. Each lives under `src/ptychodus/model/<backend>/` and exposes a `*ReconstructorLibrary` that `ModelCore` composes and hands to `ProcessingCore`. The library must degrade gracefully when the backend package isn't installed — the GUI should still open.

## Steps

### 1. Directory layout

Create `src/ptychodus/model/<backend>/` with:

- `__init__.py` — re-exports the `*ReconstructorLibrary` class only.
- `core.py` — the `*ReconstructorLibrary` class.
- `settings.py` — one or more `*Settings` classes (see the `add-core` skill for the settings pattern).
- `reconstructor.py` (or several) — the actual `Reconstructor` / `TrainableReconstructor` implementations, importing the backend package at module top so a missing dep produces a clean `ModuleNotFoundError`.

### 2. Library class

Model on [src/ptychodus/model/ptychopinn/core.py](../../src/ptychodus/model/ptychopinn/core.py):

```python
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
import logging

from ptychodus.api.reconstructor import (
    NullReconstructor, Reconstructor, ReconstructorLibrary, TrainableReconstructor,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import <Backend>Settings

logger = logging.getLogger(__name__)


class <Backend>ReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('<backend_short_name>')
        self.settings = <Backend>Settings(settings_registry)
        self._reconstructors: list[Reconstructor] = []  # or list[TrainableReconstructor]

        try:
            from .reconstructor import <Backend>Reconstructor
        except ModuleNotFoundError:
            logger.info('<Backend> not found.')
            if is_developer_mode_enabled:
                for name in ('Algorithm1', 'Algorithm2'):
                    self._reconstructors.append(NullReconstructor(name))
        else:
            try:
                pkg_version = version('<backend_package>')
            except PackageNotFoundError:
                pkg_version = 'unknown'
            logger.info(f'<Backend> {pkg_version}')
            # Instantiate real reconstructors here.
            self._reconstructors.append(<Backend>Reconstructor('Algorithm1', self.settings))

    @property
    def name(self) -> str:
        return '<Backend>'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
```

Key rules:

- **The `try / except ModuleNotFoundError` for the backend import goes inside `__init__`**, gating the concrete reconstructor construction. Never make the top-level `core.py` fail on a missing backend.
- **In developer mode**, populate `NullReconstructor` stubs so the GUI still shows the algorithm names for testing. Outside developer mode, leave the list empty when the backend is missing.
- **Add `<backend_package>` as an optional extra** in `pyproject.toml` under `[project.optional-dependencies]`.

### 3. Implement the reconstructor

In `reconstructor.py`, subclass `Reconstructor` or `TrainableReconstructor` from [src/ptychodus/api/reconstructor.py](../../src/ptychodus/api/reconstructor.py). The core contract:

- `reconstruct(parameters) -> Iterator[ReconstructOutput]` — yield one `ReconstructOutput` per iteration so the GUI can stream progress.
- For `TrainableReconstructor`, also implement `train(...)` and the ingest/export hooks.

### 4. Wire into ModelCore

In [src/ptychodus/model/core.py](../../src/ptychodus/model/core.py):

1. Import `<Backend>ReconstructorLibrary`.
2. Construct it in `ModelCore.__init__` after `settings_registry` is available.
3. Include it in the list passed to `ProcessingCore(...)`.

Follow the existing pattern for `PtyChiReconstructorLibrary`, `PtychoNNReconstructorLibrary`, `PtychoPINNReconstructorLibrary`, `PtychoPINNTorchReconstructorLibrary`.

### 5. Optional-dependency plumbing

Add the backend to `pyproject.toml`:

```toml
[project.optional-dependencies]
<backend> = ["<backend-package>>=<min-version>"]
```

Document the install command in the backend's `core.py` docstring if the install isn't a plain `uv sync --extra <backend>` (e.g. requires a sibling checkout).

## Verify

- `uv sync --extra <backend>` — install the backend.
- `uv run ptychodus` — launch the GUI; the new backend should appear in the reconstructor list. Toggle `--log-level 10` to see the "found / not found" message.
- Without the extra (`uv sync` without `--extra <backend>`), the GUI still opens and other reconstructors still work; if developer mode is on, `NullReconstructor` stubs appear.
- `pre-push` — full CI gate green.
