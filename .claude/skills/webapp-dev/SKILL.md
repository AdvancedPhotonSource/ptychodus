---
name: webapp-dev
description: Bootstrap and launch the ptychodus-webapp REST + MCP + web UI service locally against a storage root, and optionally the ptychodus-agent service behind a local nginx for the chat drawer. Use when the user says "start the webapp", "run ptychodus-webapp", "serve the webapp", "start the agent", or wants to iterate on ptychodus.webapp or ptychodus.agent code (routers, MCP tools, the agent service, or the TypeScript UI in src/ptychodus/webapp/ui/).
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

## The agent service and the chat drawer

The chat drawer calls `/agent/*` on the webapp's own origin, which only a reverse proxy provides. Locally that is nginx under podman, standing in for the facility's proxy with a fixed user. On a shared machine, pick free ports; these use 8100 for the webapp, 8101 for the agent and 8180 for nginx.

1. **Mint the agent's token** and start the webapp requiring tokens:

   ```sh
   uv sync --extra webapp --extra agent
   uv run ptychodus-webapp mint-token agent --tokens-file /tmp/ptychodus-dev/tokens.json > /tmp/ptychodus-dev/agent-token
   PTYCHODUS_WEBAPP_STORAGE_ROOT=<path> PTYCHODUS_WEBAPP_MCP_TOKENS_FILE=/tmp/ptychodus-dev/tokens.json PTYCHODUS_WEBAPP_PORT=8100 uv run ptychodus-webapp serve
   ```

2. **Start the agent** (background), with a language-model provider's credential in the environment:

   ```sh
   PTYCHODUS_AGENT_USER_HEADER=X-Forwarded-User PTYCHODUS_AGENT_PORT=8101 \
   PTYCHODUS_AGENT_MCP_URL=http://127.0.0.1:8100/mcp/ \
   PTYCHODUS_AGENT_MCP_TOKEN="$(cat /tmp/ptychodus-dev/agent-token)" \
   uv run ptychodus-agent serve
   ```

   With no language model to hand, `create_app(language_model=...)` takes any pydantic-ai model: `tests/agent_fixtures.py::ScriptedLanguageModel` calls tools and streams Markdown with math, launched from a one-off script calling `uvicorn.run(create_app(language_model=scripted.model()), port=8101)`.

3. **Start nginx** — development only, since it signs every request in as one fixed user. Save as `/tmp/ptychodus-dev/nginx.conf`:

   ```nginx
   server {
       listen 127.0.0.1:8180;

       location /agent/ {
           proxy_pass http://127.0.0.1:8101;
           proxy_http_version 1.1;
           proxy_set_header X-Forwarded-User "dev-user";
           proxy_buffering off;
           proxy_read_timeout 1h;
       }

       location / {
           proxy_pass http://127.0.0.1:8100;
           proxy_http_version 1.1;
           proxy_set_header Host $host;
       }
   }
   ```

   ```sh
   podman run -d --rm --name ptychodus-dev-nginx --network host \
       -v /tmp/ptychodus-dev/nginx.conf:/etc/nginx/conf.d/default.conf:ro,Z \
       docker.io/library/nginx:1.29-alpine
   ```

   Ask before the first `podman pull`. Open <http://127.0.0.1:8180/ui/> and press **Agent** on the rail. Stop with `podman stop ptychodus-dev-nginx`.

To drive the drawer in headless Chrome over the DevTools protocol instead of Playwright, start `google-chrome --headless=new --remote-debugging-port=9223`, percent-encode the page URL passed to `/json/new?` (a bare `#` ends the query there), and reload with `Page.reload` and `ignoreCache: true` after each `tsc`.

## Frontend iteration

The TypeScript UI lives in `src/ptychodus/webapp/ui/src/` and compiles to `src/ptychodus/webapp/ui/dist/` via plain `tsc` (no bundler). If the user is editing UI code:

```sh
cd src/ptychodus/webapp/ui && tsc --watch
```

`dist/` is build output — never commit it.

## Check the UI in a browser

`pytest-playwright` comes with the dev group, and the browser tests drive the system Chrome, so nothing is downloaded. After `tsc`:

```sh
uv run pytest -m browser tests/webapp/browser
```

They serve their own seeded storage root, so they need no running service. Without `-m browser`, `addopts` deselects every one; a stale `dist/` fails them with a reminder to run `tsc`.

To see a page of the running service — after a layout change, say — screenshot it, print what its console logs, and Read the PNG:

```sh
uv run python - <<'EOF'
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome')
    page = browser.new_page(viewport={'width': 1400, 'height': 900})
    page.on('console', lambda message: print(message.type, message.text))
    page.goto('http://localhost:8000/ui/#product')
    page.wait_for_load_state('networkidle')
    page.screenshot(path='/tmp/ptychodus-ui.png', full_page=True)
    browser.close()
EOF
```

Drop `channel='chrome'` where there is no Chrome, after `uv run playwright install chromium`.

## Stopping

The service is a foreground blocker unless backgrounded. Use `TaskStop` on the background task id when the user is done, or tell them to `Ctrl+C` if they launched it interactively.
