---
name: store-dev
description: Bootstrap and launch the ptychodus-store REST + MCP + browser-UI service locally against a storage root. Use when the user says "start the store", "run ptychodus-store", "serve the store", or wants to iterate on ptychodus_store code (routers, MCP tools, or the TypeScript UI in src/ptychodus_store/ui/).
---

# store-dev

Launches `ptychodus-store serve` with the standard local setup. Docs: `src/ptychodus_store/README.md`.

## Prerequisites

Ask the user for the storage root path if not already provided. Options, in priority order:

1. Path passed in the invocation.
2. `PTYCHODUS_STORE_STORAGE_ROOT` env var if already set in the shell.
3. Prompt: "Which storage root should the store serve from?" — do not guess a path.

## Steps

1. **Sync the store extra** (skip if already synced this session):

   ```sh
   uv sync --extra store
   ```

2. **Rebuild the SQLite index from disk** (optional — offer this only if the user says the on-disk artifacts changed outside the running service, e.g. after `git pull` or manual file moves):

   ```sh
   uv run ptychodus-store rebuild-index
   ```

3. **Launch the service** — this is a long-running process; run it in the background so you can continue helping the user:

   ```sh
   PTYCHODUS_STORE_STORAGE_ROOT=<path> uv run ptychodus-store serve
   ```

   Use `run_in_background: true` on the Bash call. Report the process id and log file path.

4. **Sanity-check** with a health probe once the server is up:

   ```sh
   curl -sf http://localhost:8000/api/v1/health
   ```

   (Port defaults per `src/ptychodus_store/config.py`; if the user configured `PTYCHODUS_STORE_*` env vars for host/port, use those.)

## Frontend iteration

The TypeScript UI lives in `src/ptychodus_store/ui/src/` and compiles to `src/ptychodus_store/ui/dist/` via plain `tsc` (no bundler). If the user is editing UI code:

```sh
cd src/ptychodus_store/ui && tsc --watch
```

`dist/` is a build artifact — never commit it.

## Stopping

The service is a foreground blocker unless backgrounded. Use `TaskStop` on the background task id when the user is done, or tell them to `Ctrl+C` if they launched it interactively.
