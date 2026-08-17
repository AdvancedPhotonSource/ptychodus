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
uv run mypy src/ptychodus scripts                   # types
```

Container builds: `podman build -f Dockerfile.{cuda,cpu,rocm,xpu} .`. Docs: `make -C docs html`.

Other entry points (`convert-to-ptychodus`, `ptychodus-bdp`, `ptychodus-store`, `ptychodus-system-check`) are listed in [pyproject.toml](pyproject.toml) `[project.scripts]`; their modules live in [src/ptychodus/cli/](src/ptychodus/cli/). Unpackaged operator tooling — the podman wrapper and the per-facility HPC token/submit helpers — lives in the top-level [scripts/](scripts/) and is run from a checkout, e.g. `python scripts/genesis/ptychodus_iri_tokens.py`. The store service has its own docs: [src/ptychodus_store/README.md](src/ptychodus_store/README.md).

### Testing

Tests cover `api/`, `model/`, and `view/widgets/` only — **do not add tests for the rest of `view/` or for `controller/`**, including controller helper functions. Widget tests depend on PyQt5, which is an optional extra (`--extra gui`); the whole `tests/view/` subtree is dropped in [tests/conftest.py](tests/conftest.py) via `collect_ignore` when `find_spec('PyQt5')` returns `None`, mirroring the `ptychodus_store` gate. `pytest tests/` must pass on a bare `pip install .` (that is what CI runs), so anything reachable from `ptychodus.model` at import time must be free of optional dependencies: gate optional backends with the `find_spec` probe + deferred factory import used in each `model/*/core.py`. Guard a test that needs an optional backend with `pytest.importorskip('ptychi')`; drop a whole optional-extra directory with `collect_ignore` in [tests/conftest.py](tests/conftest.py) — `importorskip` in a conftest is reported as an error, not a skip.

## Architecture

### Three-layer separation: api / model / view+controller

- **`src/ptychodus/api/`** — pure-Python domain layer with **no** GUI or model dependencies. Core data structures (`Product`, `ProbeSequence`, `Object`, `ProbePositionSequence`, `DiffractionDataset`, `AssembledDiffractionData`), abstract interfaces (`DiffractionFileReader/Writer`, `ProductFileReader/Writer`, `Reconstructor`, `TrainableReconstructor`, `WorkflowAPI`), and infrastructure (`Observable`/`Observer`, `Parameter`/`ParameterGroup`, `SettingsRegistry`, `PluginRegistry`/`PluginChooser`). Module naming follows SciPy conventions: activity-verb form for verb-based operations (`interpolate`, `visualize`, `reconstruct`, `propagate`, `simulate/`, `preprocess/`); plural-collection form for infrastructure (`plugins`, `settings`, `parameters`, `constants`, `typing`); singular nouns for domain data types. `simulate/` hosts forward models (`simulate/diffraction`, `simulate/object`, `simulate/probe`, `simulate/probe_positions`); `preprocess/` hosts preprocessing pipelines (`preprocess/diffraction`, `preprocess/probe_positions` — the latter owns `AffineTransform` and the RANSAC estimator). This layer depends on nothing else in ptychodus.
- **`src/ptychodus/model/`** — application logic. Each subpackage (`diffraction/`, `product/`, `processing/`, `reconstructor/`, `fluorescence/`, `analysis/`, `globus/`, `genesis/`, `ptychi/`, `ptychopinn/`, and others) exposes a `*Core` class that owns its settings, repositories, and APIs. `model/core.py::ModelCore` is the composition root that constructs every `*Core` in dependency order. Used by the GUI *and* by `__main__.py` batch mode.
- **`src/ptychodus/view/`** (PyQt5 widgets, no logic) and **`src/ptychodus/controller/`** (mediates widgets↔model). Mirror the model layout. `view/core.py::ViewCore` and `controller/core.py::ControllerCore` are composition roots; the navigation toolbar order in `ViewCore` is the source of truth for left/right stacked-panel indexes.

The GUI is optional: `__main__.py` falls back to headless if PyQt5 is missing. `ptychodus_stream_processor.py` (the `PtychodusAdImageProcessor`) is a third entry mode, imported only when `pvapy` is available.

### Index-based pattern/position association

Diffraction patterns and probe positions are paired by **integer scan index**, never by array order. This is what makes streaming and mismatched-rate ingest robust.

- Producers: `DiffractionArray.get_indexes()` (`api/diffraction.py`); `ProbePosition.index` (`api/probe_positions.py`).
- Matcher: `prepare_reconstruct_input(assembled_data, product, ...)` in `api/reconstruct.py` treats pattern indexes as authoritative — duplicate position indexes are averaged into anchors, pattern indexes inside the anchor range without an exact position are linearly interpolated, and pattern indexes outside the anchor range are dropped (no extrapolation).
- Round-trip: HDF5 and NPZ product writers persist position indexes via `ProductFileKeys.PROBE_POSITION_INDEXES`.

### Plugin system

`api/plugins.py::PluginRegistry.load_plugins()` walks `ptychodus.plugins.*` with `pkgutil.iter_modules` and calls each module's `register_plugins(registry)`. Module-load failures are logged and skipped — this is how optional-dependency plugins (LCLS, NSLS-II, …) silently disable themselves. Never make plugin loading fatal.

### Settings & observers

Settings flow through `api/parameters.py::Parameter[T]` and `ParameterGroup`. Each `*Core` calls `settings_registry.create_group(name)` and creates typed parameters. `SettingsRegistry` serializes the tree as INI. React to settings changes via `Observer`/`Observable` (`api/observer.py`); `PluginChooser.synchronize_with_parameter` is the typical bridge for "settings string → currently selected plugin." Hook observers rather than poll.

### Standard HDF5 layout (external interface — change carefully)

`api/io.py::StandardFileLayout` is the canonical contract for batch mode and remote workflows: `diffraction.h5`, `product-in.h5`, `product-out.h5`, `fluorescence-in.h5`, `fluorescence-out.h5`, `settings.ini`. Key names live in `DiffractionFileKeys` / `ProductFileKeys` enums — external consumers depend on them.

### Reconstructor libraries

Each backend (`model/ptychi/`, `model/ptychopinn/`, `model/ptychopinn_torch/`) exposes a `*ReconstructorLibrary`. Backends whose deps aren't installed must fail cleanly *at import inside their own `__init__.py`* — the `*ReconstructorLibrary` surface stays stable so `ModelCore` can always construct and pass it to `ProcessingCore`. Reconstructors yield `ReconstructOutput` per iteration so the GUI can stream progress.

### Remote compute

Both providers are optional-dep gated but constructed by `ModelCore` even when disabled:

- `model/globus/` — Globus Compute (original APS workflow).
- `model/genesis/` — IRI/AmSC HPC via `facility_adapters.py`, with per-facility scripts under `scripts/genesis/{alcf,nersc,olcf}/`.

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
- **numpy → Python scalar in `api/`.** Cast only at the boundary — the `return` of a function annotated `-> float`/`-> int`/`-> bool`, or assignment to a Python-typed dataclass field. Everywhere else let numpy scalars propagate through arithmetic. When you do cast, use the constructor (`float(...)`, `int(...)`, `bool(...)`, `complex(...)`), never `.item()` — mypy sees the target type, and it matches the HDF5-read pattern in [io.py](src/ptychodus/api/io.py).
- **Classmethods vs free functions in `api/`.** Alt-constructors on domain types use `@classmethod` with the naming split `Cls.from_X(x, ...)` for data-taking factories (`NoiseFloor.from_values`, `Interval.from_bounds` — following the stdlib `dict.fromkeys` / `datetime.fromtimestamp` idiom) and `Cls.create_X()` for parameterless factories of well-known instances (`DiffractionMetadata.create_null`, `ReconstructionAmbiguities.create_identity`, `SimpleTreeNode.create_root`). Domain-verb names (`PluginRegistry.load_plugins`) are reserved for factories with a meaningful side effect that `from_*` doesn't convey. `@staticmethod` inside classes is avoided: helpers that touch neither `self` nor `cls` go to module scope (prefixed with `_` when private). Per-`Product` (or per-domain-object) analyzers that return a small dataclass live as free `compute_*` / `estimate_*` functions in the activity-verb module — [metrics.py](src/ptychodus/api/metrics.py), [preprocess/](src/ptychodus/api/preprocess/), [reconstruct.py](src/ptychodus/api/reconstruct.py) — even when the result dataclass itself is defined elsewhere; the reference pattern is `compute_fourier_ring_correlation`, `compute_object_comparison`, `estimate_reconstruction_ambiguities`, `compute_reconstruction_residuals`.
- The `ptychodus_store/ui/` frontend is TypeScript compiled to native ES modules with plain `tsc` — no bundler, no framework, no runtime npm deps. Wheel builds run `tsc` via a `build_py` cmdclass in `setup.py`; interactive dev needs `tsc` on PATH.
- **`cli/` versus `scripts/`.** `src/ptychodus/cli/` holds the modules behind `[project.scripts]` — it ships in the wheel, and `cli/__init__.py` carries the shared argparse helpers (`DirectoryType`, `verify_all_arguments_parsed`). The top-level `scripts/` is **not** packaged: it is checkout-run operator tooling (podman wrapper, per-facility HPC token/submit helpers, demos) invoked as `python scripts/…`. A new console command goes in `cli/` with an entry point; a new one-off or facility script goes in `scripts/` with none. Both trees are type-checked (`mypy src/ptychodus scripts`) and linted.
- **All documentation is Markdown.** `docs/source/*.md` is MyST, parsed by `myst_parser`; there is no reStructuredText left in the repo and no new `.rst` should be added. Sphinx constructs use MyST directive fences — ` ```{note} `, ` ```{toctree} `, ` ```{image} `, ` ```{literalinclude} ` — with `:option: value` lines directly under the opening fence and a blank line before directive content. Autodoc stanzas stay as raw reStructuredText inside ` ```{eval-rst} ` blocks. Roles use MyST syntax: `` {py:class}`…` ``, `` {py:func}`…` ``, `` {ref}`…` ``, `` {kbd}`…` ``. Enabled MyST extensions (`colon_fence`, `deflist`, `fieldlist`) are declared in [docs/source/conf.py](docs/source/conf.py).
- **Markdown style**, enforced by `pymarkdownlnt` (`uv run pymarkdown scan $(git ls-files '*.md')`, config in `pyproject.toml` `[tool.pymarkdown]`): ATX headings only, one H1 per file, `###` max outside `docs/source/` (MyST pages may use `####`); backtick fences never tildes, and the shell tag is `sh` not `bash`; `-` bullets, ordered lists with real incrementing numbers; inline links with repo-relative paths, bare URLs as `<https://…>` autolinks; pipe tables with leading and trailing pipes and an unpadded `| --- |` separator; **no hard wrapping** — one physical line per paragraph, however long; `**bold**` for identifiers, paths, and warnings, backticks for code tokens, em dash `—` as the aside separator; no YAML front matter except `.claude/skills/*/SKILL.md`; LF endings, no trailing whitespace, one terminating newline, a blank line before every heading and fence. Fences and sub-lists nested inside a list item indent to the parent's content column — 3 spaces under an ordered marker, 2 spaces under a dash marker — with a blank line before the fence.

## Repository

CI targets `main`; local dev branches (e.g. `amsc`, `webservice`) are active — confirm the intended target before opening PRs.
