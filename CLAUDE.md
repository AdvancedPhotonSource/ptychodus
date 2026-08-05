# CLAUDE.md

## Project

Ptychodus is a ptychography data-analysis application that ingests instrument data, prepares it for processing, and dispatches it through reconstruction libraries (PtyChi, PtychoPINN, PtychoPINN-Torch). It runs as a PyQt5 GUI, a headless batch CLI, and a streaming processor inside beamline pipelines (pvapy area-detector). Python ≥3.11, `uv` preferred.

## Common Commands

```sh
uv sync --extra globus --extra gui --extra ptychi   # dev install (extras: see pyproject.toml)
uv run ptychodus                                    # GUI
uv run ptychodus -b reconstruct -i <in> -o <out>    # batch — reads StandardFileLayout in <in>/
uv run pytest                                       # tests
uv run ruff check . && uv run ruff format --check . # lint + format
uv run mypy src/ptychodus                           # types
```

Container builds: `podman build -f Dockerfile.{cuda,cpu,rocm,xpu} .`. Docs: `make -C docs html`.

Other entry points (`convert-to-ptychodus`, `ptychodus-bdp`, `ptychodus-store`, `ptychodus-system-check`, `ptychodus-iri-tokens`, `ptychodus-transfer-tokens`, …) are listed in [pyproject.toml](pyproject.toml) `[project.scripts]`. The store service has its own docs: [src/ptychodus_store/README.md](src/ptychodus_store/README.md).

## Architecture

### Three-layer separation: api / model / view+controller

- **`src/ptychodus/api/`** — pure-Python domain layer with **no** GUI or model dependencies. Core data structures (`Product`, `ProbeSequence`, `Object`, `ProbePositionSequence`, `DiffractionDataset`, `AssembledDiffractionData`), abstract interfaces (`DiffractionFileReader/Writer`, `ProductFileReader/Writer`, `Reconstructor`, `TrainableReconstructor`, `WorkflowAPI`), and infrastructure (`Observable`/`Observer`, `Parameter`/`ParameterGroup`, `SettingsRegistry`, `PluginRegistry`/`PluginChooser`). This layer depends on nothing else in ptychodus.
- **`src/ptychodus/model/`** — application logic. Each subpackage (`diffraction/`, `product/`, `processing/`, `reconstructor/`, `fluorescence/`, `analysis/`, `globus/`, `genesis/`, `ptychi/`, `ptychopinn/`, and others) exposes a `*Core` class that owns its settings, repositories, and APIs. `model/core.py::ModelCore` is the composition root that constructs every `*Core` in dependency order. Used by the GUI *and* by `__main__.py` batch mode.
- **`src/ptychodus/view/`** (PyQt5 widgets, no logic) and **`src/ptychodus/controller/`** (mediates widgets↔model). Mirror the model layout. `view/core.py::ViewCore` and `controller/core.py::ControllerCore` are composition roots; the navigation toolbar order in `ViewCore` is the source of truth for left/right stacked-panel indexes.

The GUI is optional: `__main__.py` falls back to headless if PyQt5 is missing. `ptychodus_stream_processor.py` (the `PtychodusAdImageProcessor`) is a third entry mode, imported only when `pvapy` is available.

### Index-based pattern/position association

Diffraction patterns and probe positions are paired by **integer scan index**, never by array order. This is what makes streaming and mismatched-rate ingest robust.

- Producers: `DiffractionArray.get_indexes()` (`api/diffraction.py`); `ProbePosition.index` (`api/probe_positions.py`).
- Matcher: `AssembledDiffractionData.prepare_reconstruct_input` in `api/reconstructor.py` treats pattern indexes as authoritative — duplicate position indexes are averaged into anchors, pattern indexes inside the anchor range without an exact position are linearly interpolated, and pattern indexes outside the anchor range are dropped (no extrapolation).
- Round-trip: HDF5 and NPZ product writers persist position indexes via `ProductFileKeys.PROBE_POSITION_INDEXES`.

### Plugin system

`api/plugins.py::PluginRegistry.load_plugins()` walks `ptychodus.plugins.*` with `pkgutil.iter_modules` and calls each module's `register_plugins(registry)`. Module-load failures are logged and skipped — this is how optional-dependency plugins (LCLS, NSLS-II, …) silently disable themselves. Never make plugin loading fatal.

### Settings & observers

Settings flow through `api/parametric.py::Parameter[T]` and `ParameterGroup`. Each `*Core` calls `settings_registry.create_group(name)` and creates typed parameters. `SettingsRegistry` serializes the tree as INI. React to settings changes via `Observer`/`Observable` (`api/observer.py`); `PluginChooser.synchronize_with_parameter` is the typical bridge for "settings string → currently selected plugin." Hook observers rather than poll.

### Standard HDF5 layout (external interface — change carefully)

`api/io.py::StandardFileLayout` is the canonical contract for batch mode and remote workflows: `diffraction.h5`, `product-in.h5`, `product-out.h5`, `fluorescence-in.h5`, `fluorescence-out.h5`, `settings.ini`. Key names live in `DiffractionFileKeys` / `ProductFileKeys` enums — external consumers depend on them.

### Reconstructor libraries

Each backend (`model/ptychi/`, `model/ptychopinn/`, `model/ptychopinn_torch/`) exposes a `*ReconstructorLibrary`. Backends whose deps aren't installed must fail cleanly *at import inside their own `__init__.py`* — the `*ReconstructorLibrary` surface stays stable so `ModelCore` can always construct and pass it to `ProcessingCore`. Reconstructors yield `ReconstructOutput` per iteration so the GUI can stream progress.

### Remote compute

Both providers are optional-dep gated but constructed by `ModelCore` even when disabled:

- `model/globus/` — Globus Compute (original APS workflow).
- `model/genesis/` — IRI/AmSC HPC via `facility_adapters.py`, with per-facility scripts under `src/ptychodus/scripts/genesis/{alcf,nersc,olcf}/`.

`WorkflowAPI` (`api/workflow.py`, `model/workflow.py::ConcreteWorkflowAPI`) is the unified façade GUI, batch, and remote drive.

### ptychodus_store service

`src/ptychodus_store/` is a **separate package** adjacent to `ptychodus`, not a subpackage. It reads-only from `ptychodus.api` and **must not import `ptychodus.model` or `ptychodus.view`**. Surface: FastAPI REST under `/api/v1/*`, MCP server at `/mcp`, minimal TypeScript UI at `/ui/`, SQLite metadata cache reconciled by a watchdog observer. Composition root: `create_app()` in `app.py`. Deployment details in [src/ptychodus_store/README.md](src/ptychodus_store/README.md).

## Conventions

- Ruff: **single-quoted** strings, 100-char lines, py311 target. Enabled rules: F, N, NPY.
- Preserve unit suffixes on physical quantities (`probe_energy_eV`, `set_value_from_string`) — silence naming lints with `# noqa: N802/N806/N815`, don't rename.
- Type hints are mandatory. `pyproject.toml` lists modules whose missing stubs are intentionally ignored — keep new code typed and don't widen that list.
- `ModelCore.is_developer_mode_enabled` is `True` when the effective log level ≤ DEBUG (`--log-level 10`); some controllers gate features (Agent panel, probe-position analysis) behind it.
- HTTP-client code uses `httpx` throughout (sync `httpx.Client` for repeated calls, module-level `httpx.get/post` for one-shots). `requests` is not a dependency.
- Prefer affirmative conditionals when both branches do meaningful work — `if x.is_file(): ... else: ...`, not the negated form. Guard clauses that early-return are still idiomatic.
- The `ptychodus_store/ui/` frontend is TypeScript compiled to native ES modules with plain `tsc` — no bundler, no framework, no runtime npm deps. Wheel builds run `tsc` via a `build_py` cmdclass in `setup.py`; interactive dev needs `tsc` on PATH.

## Repository

CI targets `main`; local dev branches (e.g. `amsc`, `webservice`) are active — confirm the intended target before opening PRs.
