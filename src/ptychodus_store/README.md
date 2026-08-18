# ptychodus-store

FastAPI service that indexes on-disk ptychodus artifacts (campaigns, diffraction datasets, reconstruction products, fluorescence maps), exposes them through a REST API, an MCP server, and a minimal browser UI, and keeps a SQLite metadata cache in sync with the storage root via a filesystem watcher.

- **REST**: `/api/v1/*` — list / get / render endpoints per resource kind
- **MCP**: `/mcp` — read-only tools mirroring the REST surface
- **Browser UI**: `/ui/` — six-page shell (Diffraction, Products, Positions, Probe, Object, Fluorescence) served from compiled TypeScript
- **OpenAPI**: `/openapi.json` and interactive docs at `/docs`

## Scope of the browser UI

The browser UI is intentionally **read-only** for this release. It browses artifacts already ingested into the storage root, previews them with the same colormap defaults as the PyQt desktop app, and offers `.h5` file downloads from each detail view. Reconstruction, settings editing, dataset ingestion, remote-compute (Globus, Genesis), fluorescence enhancement, and the automation / agent panels are only available in the desktop app (`uv run ptychodus`). Any writes to the storage root happen out of band — via the desktop app, batch runs (`uv run ptychodus -b reconstruct ...`), or the streaming processor.

## Install

```sh
uv sync --extra store              # required
uv sync --extra store --extra xraydb   # optional X-ray reference-data MCP sub-server
```

The shipped wheel already contains the compiled frontend. Rebuilding the UI in place is only needed when editing `ui/src/*.ts` — see [Rebuild the frontend](#rebuild-the-frontend).

## Storage layout

`PTYCHODUS_STORE_STORAGE_ROOT` should point at a directory laid out like this:

```text
<storage_root>/
  campaign/<uuid>/manifest.json
  diffraction/<uuid>/
    manifest.json
    diffraction.h5
  product/<uuid>/
    manifest.json
    product.h5
  fluorescence/<uuid>/
    manifest.json
    fluorescence.h5
```

Per-kind subdirectories are created on first start via `layout.ensure_kind_dirs()`. The service watches for changes to `manifest.json` files and reconciles them into the SQLite cache. Canonical definitions live in [storage/layout.py](storage/layout.py) and [storage/manifest.py](storage/manifest.py).

## Configuration

All settings come from `PTYCHODUS_STORE_*` environment variables (or a `.env` file in the working directory — pydantic-settings loads it automatically). See [config.py](config.py).

| Env var | Type | Default | Purpose |
| --- | --- | --- | --- |
| `PTYCHODUS_STORE_STORAGE_ROOT` | path | *(required)* | Root of the on-disk artifact tree |
| `PTYCHODUS_STORE_DATABASE_URL` | str | `sqlite+aiosqlite:///:memory:` | Async SQLAlchemy URL — use an on-disk path for durable state |
| `PTYCHODUS_STORE_HOST` | str | `127.0.0.1` | Bind address (`0.0.0.0` to expose beyond localhost) |
| `PTYCHODUS_STORE_PORT` | int | `8000` | Bind port |
| `PTYCHODUS_STORE_LOG_LEVEL` | str | `INFO` | Python + uvicorn log level |
| `PTYCHODUS_STORE_API_PREFIX` | str | `/api/v1` | REST route prefix |
| `PTYCHODUS_STORE_MCP_MOUNT_PATH` | str | `/mcp` | MCP HTTP mount path |
| `PTYCHODUS_STORE_POLLING_INTERVAL_S` | float | `2.0` | Watchdog observer poll interval |
| `PTYCHODUS_STORE_DEBOUNCE_WINDOW_S` | float | `1.0` | Manifest-change debounce window |
| `PTYCHODUS_STORE_AUTO_RECONCILE_ON_STARTUP` | bool | `true` | Run a full rescan at boot |

## Start

Foreground (dev / interactive):

```sh
PTYCHODUS_STORE_STORAGE_ROOT=/data/ptycho-store uv run ptychodus-store serve
```

The service prints `Uvicorn running on http://127.0.0.1:8000` when ready. Open `http://127.0.0.1:8000/` in a browser — the root redirects to `/ui/`.

Background (quick and dirty):

```sh
nohup uv run ptychodus-store serve > store.log 2>&1 &
echo $! > store.pid
```

For real deployments, use a systemd unit (see [Running in production](#running-in-production)).

## Stop

- **Foreground**: `Ctrl+C`.
- **Background (nohup)**: `kill $(cat store.pid)` or `pkill -f 'ptychodus-store serve'`.
- **systemd**: `systemctl stop ptychodus-store`.

## Health check

```sh
curl -s http://127.0.0.1:8000/api/v1/health/
# → {"status":"ok","db":"ok","watcher":"alive"}
```

Wire this to your uptime monitor. `status: degraded` means one of `db` or `watcher` is not `ok`/`alive`.

## Reindex

If manifests were added or moved out of band, or the watcher missed events:

```sh
PTYCHODUS_STORE_STORAGE_ROOT=/data/ptycho-store uv run ptychodus-store rebuild-index
```

This runs the same `full_rescan` the watcher invokes at startup. Safe to run while the service is up.

## Logs

Both application and uvicorn logs go to stdout, formatted as:

```text
2026-07-20 14:23:43,857 INFO ptychodus_store.ingest.watcher: manifest watcher started on /data/ptycho-store
```

Raise or lower volume with `PTYCHODUS_STORE_LOG_LEVEL=DEBUG` / `WARNING`. For file logging, redirect stdout (systemd captures stdout via journald automatically).

## Rebuild the frontend

Installing from PyPI needs no Node.js — the release sdist and wheel both ship the compiled `ui/dist/`. Building from a source checkout compiles it via `sdist`/`build_py` hooks in `setup.py` when `tsc` is on `PATH`, and warns and builds without the web UI when it is not. Release builds set `PTYCHODUS_STORE_REQUIRE_UI_BUILD=1` so a missing or stale UI fails the build instead. For interactive UI development, run `tsc` directly.

One-time setup on hosts without Node.js:

```sh
uv tool install nodeenv
nodeenv --node=lts --prebuilt ~/.local/node-lts
export PATH="$HOME/.local/node-lts/bin:$PATH"
npm install -g typescript
```

Then:

```sh
cd src/ptychodus_store/ui && tsc            # one-shot
cd src/ptychodus_store/ui && tsc --watch    # incremental during dev
```

The compiled output at `src/ptychodus_store/ui/dist/` is git-ignored. It reaches the sdist through `[tool.setuptools.package-data]`, which is why the `sdist` command compiles it too — `setuptools` finalizes `build_py` to collect package data but never runs it.

## Running in production

- **Use a durable database.** The default `sqlite+aiosqlite:///:memory:` loses all state on restart. Point at a file, e.g.:

  ```sh
  PTYCHODUS_STORE_DATABASE_URL=sqlite+aiosqlite:////var/lib/ptychodus-store/store.db
  ```

- **Sample systemd unit** (`/etc/systemd/system/ptychodus-store.service`):

  ```ini
  [Unit]
  Description=ptychodus-store HTTP+MCP service
  After=network.target

  [Service]
  Type=simple
  User=ptycho
  WorkingDirectory=/opt/ptychodus
  EnvironmentFile=/etc/ptychodus/store.env
  ExecStart=/opt/ptychodus/.venv/bin/ptychodus-store serve
  Restart=on-failure
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  ```

  Put the `PTYCHODUS_STORE_*` variables in `/etc/ptychodus/store.env`.
- **No built-in TLS.** Front with nginx or Caddy if you're exposing beyond `127.0.0.1`. To bind all interfaces set `PTYCHODUS_STORE_HOST=0.0.0.0`.
- **Storage-root permissions.** The service user must be able to create per-kind subdirectories under `PTYCHODUS_STORE_STORAGE_ROOT` on first start.
- **No auth today.** The current service is unauthenticated; keep it behind a reverse proxy that enforces auth if the data warrants it.
