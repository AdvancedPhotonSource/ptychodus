---
name: webapp-dev
description: Bootstrap and launch the ptychodus-webapp REST + MCP + web UI service locally against a storage root. Use when the user says "start the webapp", "run ptychodus-webapp", "serve the webapp", or wants to iterate on ptychodus.webapp code (routers, MCP tools, or the TypeScript UI in src/ptychodus/webapp/ui/).
---

# webapp-dev

Launches `ptychodus-webapp serve` with the standard local setup. Docs: `src/ptychodus/webapp/README.md`.

## Prerequisites

Ask the user for the storage root path if not already provided. Options, in priority order:

1. Path passed in the invocation.
2. `PTYCHODUS_WEBAPP_STORAGE_ROOT` env var if already set in the shell.
3. Prompt: "Which storage root should the webapp serve from?" — do not guess a path.

## Steps

1. **Sync the webapp extra** (skip if already synced this session):

   ```sh
   uv sync --extra webapp
   ```

2. **Rebuild the SQLite index from disk** (optional — offer this only if the user says resources in the storage root changed outside the running service, e.g. after `git pull` or manual file moves):

   ```sh
   uv run ptychodus-webapp rebuild-index
   ```

3. **Launch the service** — this is a long-running process; run it in the background so you can continue helping the user:

   ```sh
   PTYCHODUS_WEBAPP_STORAGE_ROOT=<path> PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED=true uv run ptychodus-webapp serve
   ```

   The flag serves MCP without bearer tokens, which is right for local development only; without it, or a token file from `ptychodus-webapp mint-token`, the service refuses to start.

   Use `run_in_background: true` on the Bash call. Report the process id and log file path.

4. **Sanity-check** with a health probe once the server is up:

   ```sh
   curl -sf http://localhost:8000/api/v1/health
   ```

   (Port defaults per `src/ptychodus/webapp/config.py`; if the user configured `PTYCHODUS_WEBAPP_*` env vars for host/port, use those.)

## Frontend iteration

The TypeScript UI lives in `src/ptychodus/webapp/ui/src/` and compiles to `src/ptychodus/webapp/ui/dist/` via plain `tsc` (no bundler). If the user is editing UI code:

```sh
cd src/ptychodus/webapp/ui && tsc --watch
```

`dist/` is build output — never commit it.

## Stopping

The service is a foreground blocker unless backgrounded. Use `TaskStop` on the background task id when the user is done, or tell them to `Ctrl+C` if they launched it interactively.
