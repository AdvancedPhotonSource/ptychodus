---
name: check-plugin-load
description: Diagnose why a ptychodus plugin isn't showing up in the file-format dropdowns (import errors, missing register_plugins, silent optional-dep skip). Use when the user says "my plugin isn't loading", "the reader/writer doesn't appear", or "plugin X is missing from the GUI".
---

# check-plugin-load

`PluginRegistry.load_plugins()` in [src/ptychodus/api/plugins.py](../../src/ptychodus/api/plugins.py) walks every module under `src/ptychodus/plugins/`, imports it, and calls `register_plugins(registry)`. Failures are logged at WARNING and skipped — so a broken plugin just silently vanishes from the GUI. This skill surfaces the warnings and then confirms registration.

## Steps

### 1. Ask which plugin

If the user hasn't named the plugin, ask: "Which plugin file or format is missing?" — the module name under `src/ptychodus/plugins/` is what you need.

### 2. Load plugins with debug logging

Run this — it triggers the same discovery walk the app does and prints every warning to stdout:

```sh
uv run python -c "
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s: %(message)s')
from ptychodus.api.plugins import PluginRegistry
r = PluginRegistry.load_plugins()
"
```

Scan the output for:

- **`ModuleNotFoundError`** on the plugin's own module name → the plugin's imports are broken. Look at the top of the plugin file for a missing optional dep, then either install it or wrap the import cleanly.
- **`ModuleNotFoundError`** on a *different* module the plugin imports → optional dependency missing; install it or move the import inside a `try/except`.
- **`Failed to register <plugin>`** with `AttributeError` → the plugin file was imported but has no top-level `register_plugins(registry)` function. Confirm the function is at module level (not nested), spelled exactly `register_plugins`, and takes one argument.
- **No mention of the plugin at all** → the file isn't under `src/ptychodus/plugins/` or isn't a valid Python module (missing `.py`, in a subdirectory without `__init__.py`, or the name isn't a valid identifier).

### 3. Confirm the module exists and defines the hook

```sh
ls src/ptychodus/plugins/ | grep -i <name>
grep -n "^def register_plugins" src/ptychodus/plugins/<file>.py
```

If `grep` finds nothing, the plugin needs a `register_plugins(registry: PluginRegistry) -> None` at module level.

### 4. Confirm it's registered

If discovery succeeded (a `DEBUG` line said `Registered ptychodus.plugins.<name>`), verify the plugin actually ended up in the right chooser:

```sh
uv run python -c "
from ptychodus.api.plugins import PluginRegistry
r = PluginRegistry.load_plugins()
# Pick the chooser matching the plugin category:
for p in r.probe_file_readers:
    print(p.simple_name, '—', p.display_name)
"
```

Categories on `PluginRegistry` (see [src/ptychodus/api/plugins.py](../../src/ptychodus/api/plugins.py)):
`bad_pixels_file_readers`, `diffraction_file_readers` / `_writers`, `probe_position_file_readers` / `_writers`, `fresnel_zone_plates`, `probe_file_readers` / `_writers`, `object_file_readers` / `_writers`, `product_file_readers` / `_writers`, `file_based_workflows`, `fluorescence_file_readers` / `_writers`, `upscaling_strategies`, `deconvolution_strategies`.

### 5. Report findings

Summarize: what the plugin should have registered, what actually happened, and the smallest-scope fix. Do not fix silently — hand the diagnosis back to the user with the suggested fix and let them confirm.

## Common gotchas

- Product readers not showing up as probe/position/object readers → the plugin registered on `product_file_readers` directly instead of using `registry.register_product_file_reader_with_adapters(...)`. The adapters are the reason product readers work universally.
- Plugin appears twice or with wrong casing → `simple_name` collision or missing `simple_name` (auto-derived from `display_name` may collide). Pass `simple_name` explicitly.
- Plugin loads but crashes at read time → not a load-failure; investigate the reader's `read()` method against a sample file, not with this skill.
