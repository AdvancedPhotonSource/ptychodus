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
  **Being replaced.** A backend is moving toward declaring what it accepts as a nested pydantic
  object returned from `get_default_options()`, with `model/processing/` generating the settings
  from it; see `model/ptychopinn/options.py` for a declaration ptychodus owns and
  `model/processing/_ptycho_fm_sections.py` for an overlay on an upstream one. Until the backends
  are switched over, write the `*Settings` class — but keep bounds and display text out of the
  controller, because that is the half the migration cannot do for you.
- `reconstructor.py` (or several) — the actual `Reconstructor` / `TrainableReconstructor` implementations. `core.py` imports this module only after its `find_spec` probe has confirmed the backend is installed.

### 2. Library class

Model on [src/ptychodus/model/ptychopinn/core.py](../../src/ptychodus/model/ptychopinn/core.py):

```python
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Final
import logging

from ptychodus.api.reconstruct import (
    NullReconstructor, Reconstructor, ReconstructorLibrary, TrainableReconstructor,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import <Backend>Settings

logger = logging.getLogger(__name__)

_MODES: Final[tuple[str, ...]] = ('Algorithm1', 'Algorithm2')


def _<backend>_available() -> bool:
    """Return True iff the backend is importable, without importing it."""
    return all(find_spec(mod) is not None for mod in ('<backend_package>',))


class <Backend>ReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('<backend_short_name>')
        self.settings = <Backend>Settings(settings_registry)
        self._reconstructors: list[Reconstructor] = []  # or list[TrainableReconstructor]

        if not _<backend>_available():
            logger.info('<Backend> not found.')

            if is_developer_mode_enabled:
                for mode in _MODES:
                    self._reconstructors.append(NullReconstructor(mode))
            return

        try:
            pkg_version = version('<backend_package>')
        except PackageNotFoundError:
            pkg_version = 'unknown'
        logger.info(f'<Backend> {pkg_version}')

        # Imported only now, so a missing backend never reaches an import.
        from .reconstructor import build_reconstructor

        for mode in _MODES:
            self._reconstructors.append(build_reconstructor(mode, self.settings))

    @property
    def name(self) -> str:
        return '<Backend>'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
```

Key rules:

- **Detect the backend with `find_spec`, not by catching `ModuleNotFoundError` around the import.** A probe cannot be satisfied by a half-installed package, and it keeps the backend out of `sys.modules` entirely — which `tests/test_no_gpu_context.py` enforces for any backend that would otherwise pull in a GPU framework at composition-root time.
- **Share one tuple of mode names** between the real reconstructors and the developer-mode stubs, so the GUI advertises the same names either way.
- **In developer mode**, populate `NullReconstructor` stubs so the GUI still shows the algorithm names for testing. Outside developer mode, leave the list empty when the backend is missing.
- **Add `<backend_package>` as an optional extra** in `pyproject.toml` under `[project.optional-dependencies]`, unless it is published on no index — see the comment above `[project.optional-dependencies]` for backends installed from a checkout instead.

### 3. Implement the reconstructor

In `reconstructor.py`, subclass `Reconstructor` or `TrainableReconstructor` from [src/ptychodus/api/reconstructor.py](../../src/ptychodus/api/reconstructor.py). The core contract:

- `reconstruct(parameters) -> Iterator[ReconstructOutput]` — yield one `ReconstructOutput` per iteration so the GUI can stream progress.
- For `TrainableReconstructor`, also implement `train(...)` and the ingest/export hooks.

### 4. Wire into ModelCore

In [src/ptychodus/model/core.py](../../src/ptychodus/model/core.py):

1. Import `<Backend>ReconstructorLibrary`.
2. Construct it in `ModelCore.__init__` after `settings_registry` is available.
3. Include it in the list passed to `ProcessingCore(...)`.

Follow the existing pattern for `PtyChiReconstructorLibrary`, `PtychoPINNReconstructorLibrary`, `PtychoFMReconstructorLibrary`.

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
