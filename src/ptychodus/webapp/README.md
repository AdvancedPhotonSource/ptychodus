# ptychodus-webapp

FastAPI service that indexes the resources in a ptychodus storage root (campaigns, diffraction datasets, products, fluorescence datasets), exposes them through a REST API, an MCP server, and a minimal web UI, and keeps a SQLite metadata cache in sync with the storage root via a filesystem watcher.

- **REST**: `/api/v1/*` — list / get / render endpoints per resource kind, and a product's probe, object and probe-position metrics and their comparisons with a reference product (`/product/<uuid>/probe/metrics`, `/product/<uuid>/probe/comparison?reference_uuid=…`, and likewise `object` and `positions`)
- **MCP**: `/mcp` — read-only tools taking the REST routes' parameters and refusing what they refuse; a render tool returns the PNG with its scales as structured content. Each tool is annotated read-only and its description ends with its operation class — **instant** for metadata, **bounded** for renders and reports, which read resource data — and the server's instructions describe the storage root to an agent. Callers present a bearer token (see [MCP access](#mcp-access))
- **Web UI**: `/ui/` — Diffraction, Products (with Positions, Probe and Object nested beneath it) and Fluorescence, served from compiled TypeScript. Deep links mirror the REST paths: `/ui/#product/<uuid>` selects a product, `/ui/#product/<uuid>/probe` (or `/object`, `/positions`) opens that view of it, and `/ui/#diffraction/<uuid>` and `/ui/#fluorescence/<uuid>` theirs; the address bar follows the selection. Beneath the Probe, Object and Positions images, the selected product's metrics, and its comparison with another product chosen under **Compare with**
- **OpenAPI**: `/openapi.json` and interactive docs at `/docs`

## Scope of the web UI

The web UI is intentionally **read-only**. It browses resources the webapp has already ingested from the storage root and offers `.h5` downloads from each detail view.

Within that limit it mirrors the GUI: the navigation rail carries the same entries in the same nesting, labelled under each icon; the tables and trees carry the same columns, child rows and number formatting as the PyQt repository views, with the same totals beneath them; the product detail panel lists the same rows as the GUI property table; and the viewer offers the same renderers, transforms and colormaps in the same order and with the same defaults, along with a data range written back after each render and a color legend.

Some GUI details are absent because the files cannot supply them:

- **Builder**, which `product.h5` does not record.
- The diffraction **Counts**, which would need a pass over every pattern at ingest.
- The raw detector's extent and **Physical Pixel** pitch. `diffraction.h5` holds the assembled patterns, after cropping and binning, so Width, Height and **Processed Pixel** describe those instead.
- The diffraction pattern arrays, whose labels are not persisted. Patterns hang directly beneath their dataset.

The rail's **Agent** button opens a chat drawer beside the viewer, served by the separate agent service at `/agent/*` on the same origin (see [Running in production](#running-in-production) and the [agent service README](../agent/README.md)); without it, the drawer says the agent service is unavailable and every page works as before.

Others are absent by choice: the image tools (pan and zoom, save, ruler, rectangle histogram, line cut, Fourier) and the Measured/Enhanced fluorescence toggle, since enhancing is computation. The web viewer adds what the GUI lacks: a **clip** option, a caption with the physical pixel size, and an About menu on the rail.

The analysis dialogs' reports are served as numbers: the metrics panel beneath the Probe, Object and Positions views lists every scalar a report computes, as REST and MCP return them. The dialogs' plots — spectra, Fourier ring correlations, displacement fields — remain GUI-only.

Panels with no web counterpart are the ones that write or dispatch: Settings, Processing, Globus and Genesis. Those remain GUI-only (`uv run ptychodus`). Campaigns and lineage are the reverse — a webapp-only grouping with no GUI counterpart, served over REST and MCP for agents and scripts but not browsed. Any writes to the storage root happen out of band: the GUI, the reconstruct commands (`uv run ptychodus-reconstruct ...`), or the streaming processor.

## Install

```sh
uv sync --extra webapp
```

The shipped wheel already contains the compiled frontend. Rebuilding the UI in place is only needed when editing `ui/src/*.ts` — see [Rebuild the frontend](#rebuild-the-frontend).

## Storage layout

`PTYCHODUS_WEBAPP_STORAGE_ROOT` should point at a directory laid out like this:

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

Per-kind subdirectories are created on first start via `layout.ensure_kind_dirs()`. The service watches for changes to `manifest.json` files and reconciles them into the SQLite metadata cache, ignoring hidden paths such as the temporary `.manifest.json.tmp` every manifest is written through. A replaced manifest is ingested as an update. Canonical definitions live in [api/storage.py](../api/storage.py), shared with the writers.

Finished data is added through that module's chainable savers — `save_campaign_resource`, `save_diffraction_resource`, `save_product_resource` and `save_fluorescence_resource` — which write each resource into its folder in place, `fsync` it, and publish it by writing its manifest last and atomically, so the service sees it whole or not at all: a folder without a manifest is not a resource. Each returns the new uuid, which the next takes in `campaign` or `derived_from`:

```python
campaign = save_campaign_resource(layout, label='fly001 demo')
diffraction = save_diffraction_resource(layout, data, label='scan 42', campaign=campaign)
product = save_product_resource(layout, product, derived_from=[diffraction])
```

A processing run is written as a **run resource** instead, published at the start so its progress can be watched: `ptychodus-reconstruct --product <uuid>` and `ptychodus-enhance-xrf` given uuids each make one through `start_run`, `commit_checkpoint` and `finish_run`. Its folder is the run's standard output directory — the options document, the log, and one immutable checkpoint per committed `progress` value beside the final file:

```text
product/<uuid>/
  manifest.json
  reconstruct_options.yaml
  reconstruct.log
  product.000000.h5
  product.000100.h5
  product.h5
```

The manifest is replaced atomically after each checkpoint, its `files` entry naming the newest one, so every reader resolves a resource's data through its manifest rather than a fixed file name. Its `run` record gives the `state` — `pending`, `running`, `succeeded`, `failed` or `cancelled`, with a `reason` — the `progress` against a `progress_goal` (0 when unknown), the command, ptychodus version, user and options document, and `updated_at`; a `running` resource whose `updated_at` stops advancing belongs to a process that died. A finished or cancelled run names its final file; a failed one keeps its last checkpoint. Every resource a process produces carries a `run`, including `ptychodus-stream`'s; imported data carries none.

The service exposes the record read-only: the `run` object on product, diffraction and fluorescence reads, a repeatable `run_state=` filter on their listings (REST and MCP), and a badge in the web UI. A running resource with no checkpoint yet is ingested as `AWAITING_DATA`, and its renders answer 404 until one is committed.

## Configuration

All settings come from `PTYCHODUS_WEBAPP_*` environment variables (or a `.env` file in the working directory — pydantic-settings loads it automatically). See [config.py](config.py).

| Env var | Type | Default | Purpose |
| --- | --- | --- | --- |
| `PTYCHODUS_WEBAPP_STORAGE_ROOT` | path | *(required)* | The storage root, the on-disk tree of resources |
| `PTYCHODUS_WEBAPP_DATABASE_URL` | str | `sqlite+aiosqlite:///:memory:` | Async SQLAlchemy URL — use an on-disk path for durable state |
| `PTYCHODUS_WEBAPP_HOST` | str | `127.0.0.1` | Bind address (`0.0.0.0` to expose beyond localhost) |
| `PTYCHODUS_WEBAPP_PORT` | int | `8000` | Bind port |
| `PTYCHODUS_WEBAPP_LOG_LEVEL` | str | `INFO` | Python + uvicorn log level |
| `PTYCHODUS_WEBAPP_MCP_MOUNT_PATH` | str | `/mcp` | MCP HTTP mount path |
| `PTYCHODUS_WEBAPP_POLLING_INTERVAL_S` | float | `2.0` | Watchdog observer poll interval |
| `PTYCHODUS_WEBAPP_DEBOUNCE_WINDOW_S` | float | `1.0` | Manifest-change debounce window |
| `PTYCHODUS_WEBAPP_AUTO_RECONCILE_ON_STARTUP` | bool | `true` | Run a full rescan at boot |
| `PTYCHODUS_WEBAPP_RENDER_CONCURRENCY` | int | `4` | Bounded operations — renders and reports — run in worker threads, at most this many at once |
| `PTYCHODUS_WEBAPP_MCP_TOKENS_FILE` | path | *(unset)* | The MCP client token file (see [MCP access](#mcp-access)); MCP requires it unless the next setting is on |
| `PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED` | bool | `false` | Serve MCP to every caller when no token file is set — development only |

## MCP access

MCP authenticates every caller by a bearer token; REST and the web UI do not. Each client — an integrated agent service, Claude Code, Open WebUI — gets a token of its own from the operator, who records it in the token file:

```sh
export PTYCHODUS_WEBAPP_MCP_TOKENS_FILE=/etc/ptychodus/mcp_tokens.json
uv run ptychodus-webapp mint-token claude-code   # prints the token once
uv run ptychodus-webapp revoke-token claude-code
```

`mint-token` prints a new token on stdout and records only its SHA-256 digest, so the file grants nothing by itself; minting for a client already in the file replaces its token. Either subcommand also takes `--tokens-file PATH`. The file is JSON, written atomically and readable by its owner alone, and the webapp refuses to start if anyone else may read it. It is read at startup, so restart the service after a change.

A client sends the token on every request:

```text
Authorization: Bearer <token>
```

A missing or unknown token is answered with 401, and each tool call is logged with the client its token names. A client acting for someone may name them in an `X-Ptychodus-User` header, as the agent service does for the user of each turn; the log records it beside the client (`tool render_probe called by agent for jdoe`). It is the client's claim and authorizes nothing.

The agent service is one such client:

```sh
uv run ptychodus-webapp mint-token agent        # set it as PTYCHODUS_AGENT_MCP_TOKEN
``` With no token file, the webapp refuses to start unless `PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED=true`, which serves MCP to every caller and logs a warning. Set it only for local development: binding `127.0.0.1` does not keep a caller out once a reverse proxy on the same host forwards to it.

## Start

Foreground (dev / interactive), with MCP open to every caller (see [MCP access](#mcp-access)):

```sh
PTYCHODUS_WEBAPP_STORAGE_ROOT=/data/ptycho-storage PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED=true uv run ptychodus-webapp serve
```

The service prints `Uvicorn running on http://127.0.0.1:8000` when ready. Open `http://127.0.0.1:8000/` in a browser — the root redirects to `/ui/`.

Background (quick and dirty):

```sh
nohup uv run ptychodus-webapp serve > webapp.log 2>&1 &
echo $! > webapp.pid
```

For real deployments, use a systemd unit (see [Running in production](#running-in-production)).

## Stop

- **Foreground**: `Ctrl+C`.
- **Background (nohup)**: `kill $(cat webapp.pid)` or `pkill -f 'ptychodus-webapp serve'`.
- **systemd**: `systemctl stop ptychodus-webapp`.

## Health check

```sh
curl -s http://127.0.0.1:8000/api/v1/health/
# → {"status":"ok","db":"ok","watcher":"alive"}
```

Wire this to your uptime monitor. `status: degraded` means one of `db` or `watcher` is not `ok`/`alive`.

## Reindex

If manifests were added or moved out of band, or the watcher missed events:

```sh
PTYCHODUS_WEBAPP_STORAGE_ROOT=/data/ptycho-storage uv run ptychodus-webapp rebuild-index
```

This runs the same `full_rescan` the watcher invokes at startup. Safe to run while the service is up.

## Logs

Both application and uvicorn logs go to stdout, formatted as:

```text
2026-07-20 14:23:43,857 INFO ptychodus.webapp.ingest.watcher: manifest watcher started on /data/ptycho-storage
```

Raise or lower volume with `PTYCHODUS_WEBAPP_LOG_LEVEL=DEBUG` / `WARNING`. For file logging, redirect stdout (systemd captures stdout via journald automatically).

## Rebuild the frontend

Installing from PyPI needs no Node.js — the release sdist and wheel both ship the compiled `ui/dist/`. Building from a source checkout compiles it via `sdist`/`build_py` hooks in `setup.py` when `tsc` is on `PATH`, and warns and builds without the web UI when it is not. Release builds set `PTYCHODUS_WEBAPP_REQUIRE_UI_BUILD=1` so a missing or stale UI fails the build instead. For interactive UI development, run `tsc` directly.

One-time setup on hosts without Node.js:

```sh
uv tool install nodeenv
nodeenv --node=lts --prebuilt ~/.local/node-lts
export PATH="$HOME/.local/node-lts/bin:$PATH"
npm install -g typescript
```

Then:

```sh
cd src/ptychodus/webapp/ui && tsc            # one-shot
cd src/ptychodus/webapp/ui && tsc --watch    # incremental during dev
```

The compiled output at `src/ptychodus/webapp/ui/dist/` is git-ignored. It reaches the sdist through `[tool.setuptools.package-data]`, which is why the `sdist` command compiles it too — `setuptools` finalizes `build_py` to collect package data but never runs it.

To check the UI in a browser, run the opt-in browser tests after `tsc`. They use the system Chrome where there is one, and otherwise the Chromium that `uv run playwright install chromium` downloads:

```sh
uv run pytest -m browser tests/webapp/browser
```

## Running in production

- **Use a durable database.** The default `sqlite+aiosqlite:///:memory:` loses all state on restart. Point at a file, e.g.:

  ```sh
  PTYCHODUS_WEBAPP_DATABASE_URL=sqlite+aiosqlite:////var/lib/ptychodus-webapp/metadata.db
  ```

- **Sample systemd unit** (`/etc/systemd/system/ptychodus-webapp.service`):

  ```ini
  [Unit]
  Description=ptychodus-webapp HTTP+MCP service
  After=network.target

  [Service]
  Type=simple
  User=ptycho
  WorkingDirectory=/opt/ptychodus
  EnvironmentFile=/etc/ptychodus/webapp.env
  ExecStart=/opt/ptychodus/.venv/bin/ptychodus-webapp serve
  Restart=on-failure
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  ```

  Put the `PTYCHODUS_WEBAPP_*` variables in `/etc/ptychodus/webapp.env`.
- **No built-in TLS.** Front with nginx or Caddy if you're exposing beyond `127.0.0.1`. To bind all interfaces set `PTYCHODUS_WEBAPP_HOST=0.0.0.0`.
- **Storage-root permissions.** The service user must be able to create per-kind subdirectories under `PTYCHODUS_WEBAPP_STORAGE_ROOT` on first start.
- **Storage root on NFS.** Set `HDF5_USE_FILE_LOCKING=FALSE` for the service and for every process writing into the storage root. HDF5's `flock` can fail on NFS, and locking buys nothing here: a data file is immutable once a manifest names it. Expect a reader to see a new manifest only after the client's attribute-cache interval.
- **Authentication.** REST and the web UI authenticate no one; keep them behind a reverse proxy that enforces the facility's single sign-on. MCP checks its own bearer tokens (see [MCP access](#mcp-access)), so exempt `/mcp` from browser sign-on at the proxy, and do not let the proxy buffer its streamed responses. Never set `PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED` behind a proxy.
- **The agent service shares the origin.** The proxy routes `/agent/` to `ptychodus-agent`, unbuffered, behind the same sign-on, and sets the signed-in user in the header `PTYCHODUS_AGENT_USER_HEADER` names; setting it replaces any copy the client sent. A sketch for nginx with `auth_request`, whose SSO response carries the user in `X-Auth-Request-User`:

  ```nginx
  location /agent/ {
      auth_request /oauth2/auth;
      auth_request_set $user $upstream_http_x_auth_request_user;
      proxy_pass http://127.0.0.1:8001;
      proxy_http_version 1.1;
      proxy_set_header X-Forwarded-User $user;
      proxy_buffering off;
      proxy_read_timeout 1h;
  }

  location /mcp {
      proxy_pass http://127.0.0.1:8000;
      proxy_http_version 1.1;
      proxy_buffering off;
  }

  location / {
      auth_request /oauth2/auth;
      proxy_pass http://127.0.0.1:8000;
  }
  ```

  The agent service believes that header, so anyone who can reach its port directly can claim any user, read that user's conversations and spend the shared quota. Deploy it where only the proxy can connect — on a shared host, every local user can reach a loopback port.
