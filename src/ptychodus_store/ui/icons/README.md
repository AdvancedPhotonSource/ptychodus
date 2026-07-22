# UI icons

Single source of truth for icon SVGs used by both the ptychodus web UI (`src/ptychodus_store/ui/`) and the PyQt GUI (via `src/ptychodus/view/resources.qrc`, which uses relative paths that point back here).

## Files

- `ptychodus.svg`, `genesis.svg`, `globus.svg` — project-owned SVGs.
- Everything else — a subset of [Font-Awesome 7.1.0](https://fontawesome.com/) (Free), CC BY 4.0. See `Font-Awesome-LICENSE.txt`.

Filenames are the original Font-Awesome names, with no `solid/` vs `regular/` prefix (the subset in use has no name collisions). Serving is flat: `/ui/icons/<name>.svg`.

## Adding a new icon

1. Drop the SVG file into this directory (from Font-Awesome or elsewhere).
2. Update `src/ptychodus_store/ui/src/nav.ts` (or whichever component uses it) to reference `/ui/icons/<name>.svg`.
3. If the PyQt GUI also needs it, add a line to `src/ptychodus/view/resources.qrc`:
   ```xml
   <file alias="<alias>">../../ptychodus_store/ui/icons/<name>.svg</file>
   ```
   then run `src/ptychodus/view/make_qrc.sh` to regenerate `resources.py`.

## Updating Font-Awesome

Fetch a newer release from https://github.com/FortAwesome/Font-Awesome, replace the SVGs referenced above by name (Font-Awesome preserves filenames across minor releases), and refresh `Font-Awesome-LICENSE.txt`. Rerun `make_qrc.sh` if `resources.qrc` references changed files.
