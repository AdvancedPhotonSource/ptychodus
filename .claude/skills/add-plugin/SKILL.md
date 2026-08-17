---
name: add-plugin
description: Scaffold a new file-format plugin (diffraction / probe / probe-position / object / product / fluorescence reader or writer) under src/ptychodus/plugins/. Use when the user says "add a plugin for X format", "add a reader/writer for Y", or is integrating a new beamline data format.
---

# add-plugin

Plugins are auto-discovered by `PluginRegistry.load_plugins()` in [src/ptychodus/api/plugins.py](../../src/ptychodus/api/plugins.py). Every module under `src/ptychodus/plugins/` that defines a module-level `register_plugins(registry)` function is imported and registered at startup; modules whose imports fail are logged and skipped (this is how optional-dependency plugins silently disable themselves).

## Steps

1. **Identify the abstract interface.** Pick the right one from `ptychodus.api`:

   | You want to read/write... | Interface | Registry chooser |
   | --- | --- | --- |
   | Raw diffraction patterns | `DiffractionFileReader/Writer` (`api/diffraction.py`) | `diffraction_file_readers` / `_writers` |
   | Probe positions | `ProbePositionFileReader/Writer` (`api/probe_positions.py`) | `probe_position_file_readers` / `_writers` |
   | Probes | `ProbeFileReader/Writer` (`api/probe.py`) | `probe_file_readers` / `_writers` |
   | Objects | `ObjectFileReader/Writer` (`api/object.py`) | `object_file_readers` / `_writers` |
   | A full Product (probe + positions + object bundled) | `ProductFileReader/Writer` (`api/product.py`) | see step 4 |
   | Fluorescence / bad pixels | `Fluorescence*` / `BadPixelsFileReader` | corresponding choosers |

2. **Create the module** at `src/ptychodus/plugins/<beamline_or_format>_<kind>_file.py`. Follow the naming convention already in use — e.g. `csv_probe_file.py`, `aps33id_velociprobe/`, `aps19id_isn_diffraction_file.py`. Beamline plugins may be a subpackage if they need more than one file.

3. **Implement Reader and/or Writer** by subclassing the interface and providing the `read(path) -> ...` / `write(path, data) -> None` method. Keep the imports minimal — put any optional dependency import at module top so a `ModuleNotFoundError` cleanly disables the plugin.

4. **Register.** Add `register_plugins(registry: PluginRegistry)` at module level.

   - **For a product reader**, prefer `register_product_file_reader_with_adapters` so the same reader is also reachable as a probe/probe-position/object reader without extra boilerplate:

     ```python
     def register_plugins(registry: PluginRegistry) -> None:
         registry.register_product_file_reader_with_adapters(
             MyProductReader(),
             display_name='My Format (*.h5)',
             simple_name='MyFormat',
         )
     ```

   - **For everything else**, register on the appropriate chooser with both `display_name` (human-readable, includes the glob) and `simple_name` (short token used by the settings string):

     ```python
     def register_plugins(registry: PluginRegistry) -> None:
         registry.probe_file_readers.register_plugin(
             MyProbeReader(),
             simple_name='MyFormat',
             display_name='My Format Files (*.myext)',
         )
     ```

   `simple_name` defaults to a stripped alphanumeric form of `display_name`; provide it explicitly if you want a stable settings token.

## Reference examples

Read one of these before writing new code — pick the closest match:

- Simple single-file reader+writer pair: [src/ptychodus/plugins/csv_probe_file.py](../../src/ptychodus/plugins/csv_probe_file.py)
- HDF5-backed diffraction: [src/ptychodus/plugins/h5_diffraction_file.py](../../src/ptychodus/plugins/h5_diffraction_file.py)
- Product reader with adapters: [src/ptychodus/plugins/cxi_file.py](../../src/ptychodus/plugins/cxi_file.py), [src/ptychodus/plugins/h5_product_file.py](../../src/ptychodus/plugins/h5_product_file.py)
- Beamline subpackage: [src/ptychodus/plugins/aps33id_velociprobe/](../../src/ptychodus/plugins/aps33id_velociprobe/)
- Optional-dep plugin (fails cleanly if the dep is missing): [src/ptychodus/plugins/lcls_file_readers.py](../../src/ptychodus/plugins/lcls_file_readers.py)

## Index preservation

Diffraction patterns and probe positions are paired by **integer scan index**, not array order (see CLAUDE.md "Index-based pattern/position association"). A new diffraction reader must set `DiffractionArray.get_indexes()`; a new probe-position reader must set `ProbePosition.index`. A round-trip test that preserves indexes is worth writing.

## Verify

After creating the plugin, run the `check-plugin-load` skill to confirm it registers cleanly, then run `pre-push` before committing.
