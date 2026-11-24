# ptychodus-agent

The integrated agent: a pydantic-ai agent whose tools are the [webapp](../webapp/README.md)'s MCP server, streaming to the web UI's chat drawer over [AG-UI](https://docs.ag-ui.com/). Each user's conversations are private to them, and every turn — one prompt and the agent's answer — is recorded in an audit log. It reads the storage root only through MCP, as any outside agent does, so it needs no access to the storage root itself.

## Install

```sh
uv sync --extra webapp --extra agent
```

The `agent` extra carries pydantic-ai with its AG-UI support, and the server-side Markdown and math renderer.

## Configuration

Environment variables, or a `.env` file in the working directory:

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `PTYCHODUS_AGENT_USER_HEADER` | str | *(required)* | The header the reverse proxy sets to the signed-in user, e.g. `X-Forwarded-User` or `Remote-User` |
| `PTYCHODUS_AGENT_DEVELOPMENT_USER` | str | *(unset)* | The user a request without that header is taken to be — development only, with no proxy |
| `PTYCHODUS_AGENT_HOST` | str | `127.0.0.1` | Bind address |
| `PTYCHODUS_AGENT_PORT` | int | `8001` | Bind port |
| `PTYCHODUS_AGENT_LOG_LEVEL` | str | `INFO` | Log level |
| `PTYCHODUS_AGENT_DATABASE_URL` | str | `sqlite+aiosqlite:///:memory:` | Conversations and the audit log; in memory, both vanish on a restart |
| `PTYCHODUS_AGENT_PROVIDER` | str | `Argo` | The language-model provider: `Anthropic`, `Argo`, `Google`, `OpenAI` or `OpenAICompatible` |
| `PTYCHODUS_AGENT_MODEL_ID` | str | *(provider's default)* | The language model |
| `PTYCHODUS_AGENT_PROVIDER_OPTIONS` | JSON | `{}` | The provider's options over its defaults, e.g. `{"base_url": "http://10.0.0.5:8000/v1"}` |
| `PTYCHODUS_AGENT_INSTRUCTIONS` | str | *(a short domain prompt)* | The agent's instructions; the webapp's MCP instructions are added to them |
| `PTYCHODUS_AGENT_THINKING` | str | `Default` | `Default`, `Off`, `Low`, `Medium` or `High` |
| `PTYCHODUS_AGENT_MAX_TOKENS` | int | `0` | The most tokens one response may hold; 0 leaves it to the language model |
| `PTYCHODUS_AGENT_MCP_URL` | str | `http://127.0.0.1:8000/mcp/` | The webapp's MCP server; https or loopback when a token is set |
| `PTYCHODUS_AGENT_MCP_TOKEN` | str | *(unset)* | The token `ptychodus-webapp mint-token agent` printed |
| `PTYCHODUS_AGENT_TURN_REQUEST_LIMIT` | int | `20` | The most language-model requests one turn may make |
| `PTYCHODUS_AGENT_TURN_TOOL_CALLS_LIMIT` | int | `50` | The most tool calls one turn may make |
| `PTYCHODUS_AGENT_TURN_TOTAL_TOKENS_LIMIT` | int | *(unset)* | The most tokens one turn may spend |
| `PTYCHODUS_AGENT_USER_DAILY_TOKENS_LIMIT` | int | *(unset)* | The most tokens one user may spend in any 24 hours |

Each provider reads its credential from its own environment variable — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and for Argo `ARGO_USER` — and sends it only to an https or loopback URL. The language model is resolved once at startup, without touching the network, so a missing credential stops the service from starting. With Argo, the service connects as one account, so Argo sees every request as that account's; the audit log records who each turn was for.

## Start

```sh
export PTYCHODUS_AGENT_USER_HEADER=X-Forwarded-User
export PTYCHODUS_AGENT_MCP_TOKEN=<the token mint-token printed>
export ARGO_USER=<the service account>
uv run ptychodus-agent serve
```

Run it as one process: a conversation's running turn is tracked in memory, which keeps two turns from running in one conversation at once.

## Behind the proxy

The web UI calls `/agent/v1/*` on its own origin, so the facility's reverse proxy routes that prefix to this service behind the same sign-on as the webapp, without buffering its event streams; the [webapp README](../webapp/README.md#running-in-production) shows an nginx sketch. The proxy sets the signed-in user in `PTYCHODUS_AGENT_USER_HEADER`, replacing any copy the client sent, and the service refuses a request without it (401).

**The service believes that header.** Anyone who can reach its port without passing through the proxy can claim any user, read that user's conversations and spend the shared quota. Deploy it where only the proxy can connect; on a shared host, every local user can reach a loopback port.

## Routes

| Route | Purpose |
| --- | --- |
| `GET /agent/v1/health` | Liveness, without a user |
| `GET /agent/v1/session` | The user and the language model |
| `GET /agent/v1/conversations` | The user's conversations, most recent first |
| `POST /agent/v1/conversations` | A new conversation (`application/json`) |
| `GET /agent/v1/conversations/{id}` | A conversation, its assistant text rendered as sanitized HTML |
| `DELETE /agent/v1/conversations/{id}` | Delete a conversation; its turns stay in the audit log |
| `POST /agent/v1/conversations/{id}/turns` | Run a turn: an AG-UI `RunAgentInput` whose `threadId` is the conversation, answered as AG-UI events over SSE |

A turn's stream carries the AG-UI events of the run, and a `ptychodus.rendered_text` CUSTOM event with each text message's HTML as each block of it completes. Tool results arrive without image data; the web UI draws each render tool's image from the webapp's REST route. A turn that fails or is stopped leaves its conversation as it was.

## Logs and audit

Every turn, however it ends — completed, failed, cancelled, or refused at the daily limit — writes a row to the `turns` table of the database (user, conversation, language model, outcome and error, requests, tokens and tool calls) and one INFO line to the log. The webapp logs each tool call with the user the agent named.
