---
name: add-store-endpoint
description: Add a matched pair of a FastAPI REST route and an MCP tool in the ptychodus_store service. Use when the user says "add an endpoint to the store", "expose X in the API", or "add an MCP tool for Y" — because REST and MCP surfaces must stay in sync.
---

# add-store-endpoint

`ptychodus_store` exposes the same read-only surface through two channels: FastAPI REST under `/api/v1/*` (routers in `src/ptychodus_store/routers/`) and MCP tools at `/mcp` ([src/ptychodus_store/mcp_server.py](../../src/ptychodus_store/mcp_server.py)). They must return the same data for the same query. Adding one without the other silently drifts.

## Steps

### 1. Choose or create a router module

- Existing resource? Extend an existing file in [src/ptychodus_store/routers/](../../src/ptychodus_store/routers/) (e.g. `diffraction.py`, `product.py`, `campaign.py`).
- New resource? Create `src/ptychodus_store/routers/<resource>.py`.

### 2. REST route

Follow the pattern in [src/ptychodus_store/routers/diffraction.py](../../src/ptychodus_store/routers/diffraction.py):

```python
from fastapi import APIRouter, HTTPException, Query
from ptychodus_store.db import repositories as repo
from ptychodus_store.routers._convert import <resource>_to_read
from ptychodus_store.routers.deps import SessionDep, LayoutDep
from ptychodus_store.routers.schemas import <Resource>Read, Page
from ptychodus_store.storage.manifest import ResourceKind

router = APIRouter(prefix='/<resource>', tags=['<resource>'])


@router.get('', response_model=Page[<Resource>Read])
async def list_<resource>(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    # ... filter params
) -> Page[<Resource>Read]:
    ...
```

Use `SessionDep` for DB access and `LayoutDep` when you need the on-disk storage layout. Filtering uses SQLAlchemy `where` clauses built up from optional query params — see the diffraction router for the canonical shape.

### 3. Register the router

In [src/ptychodus_store/app.py](../../src/ptychodus_store/app.py) inside `create_app()`, add:

```python
app.include_router(<resource>.router, prefix=api_prefix)
```

Match the position/style of the existing `app.include_router(...)` calls (health, campaign, diffraction, product, fluorescence, lineage, admin, visualization).

### 4. Schemas

Add pydantic response models to [src/ptychodus_store/routers/schemas.py](../../src/ptychodus_store/routers/schemas.py) if a new `*Read` shape is needed. Reuse `Page[T]` for paginated list responses.

### 5. Converters

If your ORM row → pydantic conversion is non-trivial (or needs a session query for related data), add a helper in [src/ptychodus_store/routers/_convert.py](../../src/ptychodus_store/routers/_convert.py) — this is what keeps REST and MCP in sync, because both channels call the same converter.

### 6. Matching MCP tool

Immediately add the corresponding tool in [src/ptychodus_store/mcp_server.py](../../src/ptychodus_store/mcp_server.py) inside `create_mcp_server()`. Every REST route must have an MCP counterpart with the same behavior:

```python
@mcp.tool()
async def list_<resource>(
    limit: int = 50,
    offset: int = 0,
    # ... same filter params as REST route
) -> Page[<Resource>Read]:
    """<one-line docstring — shown to MCP clients>"""
    async with _session() as session:
        # ... same repo calls, same converter, same response
```

Rules:

- MCP tools take `uuid: str` (they lack FastAPI's UUID coercion) — convert with `UUID(uuid_str)` inside the tool.
- Use the same `_convert` helper as the REST route. Do not duplicate business logic.
- Raise `ToolError` for MCP-side errors (not `HTTPException`).

### 7. DB model (only if adding a new resource type)

If this is a genuinely new resource (not a new query over an existing one), you also need:

- SQLAlchemy model in `src/ptychodus_store/db/models.py` (UUID PK, `ingest_state`, metadata fields).
- New `ResourceKind` enum variant in `src/ptychodus_store/storage/manifest.py`.
- Ingestion path in `src/ptychodus_store/ingest/` (reconciler + watcher will pick it up automatically once the kind is registered).

## Testing

Health check after launching (see `store-dev` skill):

```sh
curl -sf http://localhost:8000/api/v1/<resource>
curl -sf http://localhost:8000/api/v1/<resource>/<uuid>
```

For the MCP tool, `fastmcp inspect http://localhost:8000/mcp` (or invoke via an MCP client) should list the new tool and return matching data.

## Do not

- Do not import from `ptychodus.model` or `ptychodus.view` — `ptychodus_store` is read-only from `ptychodus.api` only.
- Do not add write endpoints without user confirmation — the store surface is intentionally read-only.
