# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ptychodus is a ptychography data-analysis application that ingests instrument data, prepares it for processing, and dispatches it through several reconstruction libraries (PtyChi, PtychoPINN, PtychoPINN-Torch). It runs interactively as a PyQt5 GUI, headless via batch CLI, and as a streaming processor inside beamline pipelines (pvapy area-detector). Python ≥3.11.

## Common Commands

Project uses `uv` (preferred) and a developer install with extras.

```sh
# Dev install (preferred). Other available extras: docs, ptychopinn
uv sync --extra globus --extra gui --extra ptychi

# Launch GUI
uv run ptychodus

# Headless batch
uv run ptychodus -b reconstruct -i <input_dir> -o <output_dir>
uv run ptychodus -b train      -i <input_dir> -o <output_dir>
# Batch mode reads <input_dir>/settings.ini, <input_dir>/diffraction.h5, <input_dir>/product-in.h5
# (see StandardFileLayout in src/ptychodus/api/io.py)

# Beamline data-prep CLI
uv run ptychodus-bdp --product-name <name> --diffraction-input <h5> \
                     --probe-position-input <csv> --output-directory <dir> \
                     --settings <ini>

# ptychodus-store service (REST + MCP + browser UI). Full details in
# src/ptychodus_store/README.md.
uv sync --extra store
PTYCHODUS_STORE_STORAGE_ROOT=/path/to/artifacts uv run ptychodus-store serve
uv run ptychodus-store rebuild-index    # full DB reconciliation from disk

# Other entry points (see [project.scripts] in pyproject.toml)
uv run convert-to-ptychodus
uv run ptychodus-system-check
uv run ptychodus-iri-tokens          # Genesis/IRI auth
uv run ptychodus-transfer-tokens     # AmSC data-transfer auth
uv run ptychodus-ptychopinn-tf-test  # PtychoPINN TensorFlow smoke check
```

Tests, lint, types:

```sh
uv run pytest                                  # full suite (tests/)
uv run pytest tests/test_io.py                 # one file
uv run pytest tests/test_io.py::test_name      # one test
uv run ruff check .                            # lint  (rules: F, N, NPY)
uv run ruff format .                           # format (single quotes, line-length 100)
uv run mypy src/ptychodus                      # type check (py 3.11)
```

CI (`.github/workflows/python-package.yml`) runs four jobs on push/PR to `main`: a `pip install` + `ptychodus --version` smoke test (Py 3.11/3.12/3.13), `pytest tests/`, `ruff check` + `ruff format --check`, and `mypy src/ptychodus`. Run the local equivalents before pushing — CI will block the PR otherwise.

Container & docs:

```sh
podman build -f Dockerfile.cuda -t ptychodus:cuda13.0 .   # also: Dockerfile.cpu / .rocm / .xpu
docker build  -f Dockerfile.cuda -t ptychodus:cuda13.0 .
make -C docs html        # Sphinx docs into docs/build/
```

## High-Level Architecture

### Three-layer separation: api / model / view+controller

- **`src/ptychodus/api/`** — pure-Python domain layer with **no** GUI or model dependencies. Defines core data structures (`Product`, `ProbeSequence`, `Object`, `ProbePositionSequence`, `DiffractionDataset`, `AssembledDiffractionData`), abstract interfaces (`DiffractionFileReader/Writer`, `ProductFileReader/Writer`, `Reconstructor`, `TrainableReconstructor`, `WorkflowAPI`), and infrastructure (`Observable`/`Observer`, `Parameter`/`ParameterGroup`, `SettingsRegistry`, `PluginRegistry`/`PluginChooser`). All other layers depend on this; this layer depends on nothing else in ptychodus.
- **`src/ptychodus/model/`** — application logic. Each subpackage (`diffraction/`, `product/`, `processing/`, `reconstructor/`, `analysis/`, `fluorescence/`, `globus/`, `genesis/`, `automation/`, `agent/`, `visualization/`, `ptychi/`, `ptychopinn/`, `ptychopinn_torch/`) exposes a `*Core` class that owns its settings, repositories, and APIs. `model/core.py::ModelCore` is the composition root: it constructs every `*Core` in dependency order and wires them together. Used both by the GUI and by `__main__.py` batch mode.
- **`src/ptychodus/view/`** (PyQt5 widgets, no logic) and **`src/ptychodus/controller/`** (mediates between widgets and model). Mirror the model package layout. `view/core.py::ViewCore` and `controller/core.py::ControllerCore` are the composition roots; the navigation toolbar order in `ViewCore` is the source of truth for the left/right stacked-panel indexes.

The GUI is optional: `__main__.py` falls back to headless mode if PyQt5 is missing. `ptychodus_stream_processor.py` (the `PtychodusAdImageProcessor`) is the third entry mode and is only imported when `pvapy` is available.

### Index-based pattern/position association

Diffraction patterns and probe positions are paired by **integer scan index**, never by array order. This is what makes streaming and mismatched-rate ingest robust: patterns and positions can arrive from different files or different PV channels, with dropped or extra samples on either side, and the matcher still pairs them correctly.

- Producers: `DiffractionArray.get_indexes()` on the pattern side (`api/diffraction.py`); `ProbePosition.index` on the position side (`api/probe_positions.py`).
- Matcher: `AssembledDiffractionData.prepare_reconstruct_input` in `api/reconstructor.py` treats pattern indexes as authoritative — duplicate position indexes are averaged into anchors, pattern indexes inside the anchor range with no exact position are linearly interpolated, and pattern indexes outside the anchor range are dropped (no extrapolation). The `Product` is rebuilt from the resulting per-pattern positions.
- Round-trip: HDF5 and NPZ product writers persist position indexes via `ProductFileKeys.PROBE_POSITION_INDEXES`.

### Plugin system

File-format support is dynamic. `api/plugins.py::PluginRegistry.load_plugins()` walks `ptychodus.plugins.*` with `pkgutil.iter_modules` and calls each module's `register_plugins(registry)` function. Module-load failures are logged and skipped — this is how optional-dependency plugins (e.g., LCLS, NSLS-II) silently disable themselves.

To add a new file format: create a module under `src/ptychodus/plugins/`, implement the relevant `*FileReader`/`*FileWriter` from `ptychodus.api`, and define `register_plugins(registry)` calling the appropriate `registry.<category>.register_plugin(...)`. For product readers, prefer `register_product_file_reader_with_adapters` so the reader is also reachable as a probe/probe-position/object reader.

### Settings & observers

Settings flow through `api/parametric.py::Parameter[T]` and `ParameterGroup`. Each `*Core` calls `settings_registry.create_group(name)` and creates typed parameters on it. `SettingsRegistry` serializes the full tree as INI. Components react to settings changes through the `Observer`/`Observable` pattern in `api/observer.py`; `PluginChooser.synchronize_with_parameter` is the typical bridge for "settings string → currently selected plugin." When adding cross-component reactivity, hook observers rather than poll.

### Reconstructor libraries

Each reconstructor backend (`model/ptychi/`, `model/ptychopinn/`, `model/ptychopinn_torch/`) exposes a `*ReconstructorLibrary` class. They are constructed in `ModelCore` and passed as a list to `ProcessingCore`. Backends that aren't installed should fail cleanly at import inside their own `__init__.py` — keep the surface a stable `*ReconstructorLibrary` regardless. Reconstructors implement `Reconstructor`/`TrainableReconstructor` from `api/reconstructor.py` and yield `ReconstructOutput` per iteration so the GUI can stream progress.

### Standard HDF5 layout

`api/io.py::StandardFileLayout` is the canonical contract for batch mode and remote workflows: `diffraction.h5`, `product-in.h5`, `product-out.h5`, `fluorescence-in.h5`, `fluorescence-out.h5`, `settings.ini`. `load_diffraction_data` / `save_diffraction_data` and `load_product` / `save_product` round-trip these files; their key names live in `DiffractionFileKeys` / `ProductFileKeys` enums — change those carefully, they are an external interface.

### Remote compute

Two providers, both gated by optional dependencies and constructed by `ModelCore` even when disabled:

- `model/globus/` — Globus Compute submission, used by the original APS workflow.
- `model/genesis/` — IRI/AmSC HPC submission with facility adapters in `model/genesis/facility_adapters.py` and per-facility scripts under `src/ptychodus/scripts/genesis/{alcf,nersc,olcf}/`.

`WorkflowAPI` (see `api/workflow.py` and `model/workflow.py::ConcreteWorkflowAPI`) is the unified façade that GUI, batch mode, and remote workflows all drive.

### ptychodus_store service

`src/ptychodus_store/` is a **separate package** adjacent to `ptychodus`, not a subpackage. It reads-only from `ptychodus.api` (loaders, product schema, visualization types) and never imports `ptychodus.model` or `ptychodus.view`. It exposes:

- a FastAPI REST API under `/api/v1/*` (routers in `src/ptychodus_store/routers/`),
- an MCP server at `/mcp` (see `src/ptychodus_store/mcp_server.py`) with read-only tools mirroring the REST surface,
- a minimal TypeScript browser UI at `/ui/` (source in `src/ptychodus_store/ui/src/`, compiled by `tsc` to `ui/dist/`),
- a SQLite metadata cache reconciled against the on-disk storage root by a watchdog observer.

Composition root is `create_app()` in `src/ptychodus_store/app.py`. Settings load via pydantic-settings from `PTYCHODUS_STORE_*` env vars (see `src/ptychodus_store/config.py`). Full deployment docs: `src/ptychodus_store/README.md`.

## Conventions

- Ruff is configured for **single-quoted** strings, 100-char lines, py311 target. The selected lint rules (F, N, NPY) flag pyflakes errors, PEP-8 naming, and NumPy-specific issues. NumPy/Qt-style names (e.g., `probe_energy_eV`, `set_value_from_string`) are accepted via `# noqa: N802/N806/N815` — preserve the unit suffixes on physical quantities; reviewers expect them.
- Type hints are mandatory; `pyproject.toml` lists modules whose missing stubs are intentionally ignored. Keep new code typed and avoid widening that ignore list.
- `model/core.py::ModelCore.is_developer_mode_enabled` is `True` whenever the effective log level ≤ DEBUG (`--log-level 10`). Some controllers gate features (Agent panel, probe-position analysis) behind it.
- HTTP-client code uses `httpx` throughout (sync `httpx.Client` for repeated calls, module-level `httpx.get/post/...` for one-shots). `requests` is not a direct dependency.
- Prefer affirmative conditionals. For `if/else` pairs, put the positive case first (`if x.is_file(): ... else: ...`), not the negated one (`if not x.is_file(): ... else: ...`) — reading the affirmative branch up front reduces cognitive load. Guard clauses that early-return (`if not x: return`) are still idiomatic; this rule targets branching where both sides do meaningful work.
- The `ptychodus_store/ui/` frontend is authored in TypeScript, compiled to native ES modules with plain `tsc` (no bundler, no framework, no runtime npm deps). Wheel builds run `tsc` automatically via a `build_py` cmdclass in `setup.py`; interactive dev needs `tsc` on PATH.

## Repository Notes

- The git CI workflow targets `main`; local development branches such as `amsc` are active — confirm the intended target before opening PRs.
- Sample `.h5`/`.npy` data and any `dist/` output — including the compiled frontend at `src/ptychodus_store/ui/dist/` and packaging output at repo-root `dist/` — are untracked local artifacts. The frontend `dist/` is regenerated automatically at wheel build time; do not stage it in commits.
