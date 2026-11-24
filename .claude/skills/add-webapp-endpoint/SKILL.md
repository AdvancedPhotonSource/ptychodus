---
name: add-webapp-endpoint
description: Add a read operation to the ptychodus.webapp service — a ResourceService method with its operation class, exposed as a matched FastAPI REST route and MCP tool. Use when the user says "add an endpoint to the webapp", "add a route to the webapp", "expose X in the API", or "add an MCP tool for Y" — because REST and MCP surfaces must stay in sync.
---

# add-webapp-endpoint

`ptychodus.webapp` serves one read-only surface through two transports: FastAPI REST under `/api/v1/*` (routers in `src/ptychodus/webapp/routers/`) and MCP tools at `/mcp` ([src/ptychodus/webapp/mcp_server.py](../../../src/ptychodus/webapp/mcp_server.py)). Both are thin calls into one service, [service/resources.py](../../../src/ptychodus/webapp/service/resources.py)'s `ResourceService`, so an operation's logic is written once. [tests/webapp/test_rest_mcp_parity.py](../../../tests/webapp/test_rest_mcp_parity.py) fails on a route without a tool, or a tool without a route, unless it is listed as an intended difference.

## Steps

### 1. The service method

Add a method to `ResourceService` and decorate it with its operation class:

- `@_INSTANT` — reads only the metadata cache (`list_*`, `get_*`, counts, options).
- `@_BOUNDED` — also reads a resource's files and computes on them (renders, reports). Look the resources up first (`resource_file`, `_product_file_and_run`), then run the blocking work through `self._runner.run(view, *args)`, never on the event loop.

```python
@_BOUNDED
async def get_<thing>(self, uuid: UUID, *, <tunable>: int) -> <Thing>Read:
    """<One line an agent reads as the tool's description.>"""
    path, run = await self._product_file_and_run(uuid)
    scalars = await self._runner.run(<thing>_view, path, <tunable>)
    return <Thing>Read(uuid=uuid, run=run, scalars=scalars)
```

The docstring becomes the MCP tool's description, followed by `Operation class: <class>.`, so write it for an agent.

Refuse with the service's errors from [service/errors.py](../../../src/ptychodus/webapp/service/errors.py), never `HTTPException` or `ToolError`: `ResourceNotFoundError` (404; pass `available=` to list the choices there were) or `InvalidRequestError` (400). Each transport maps them to its own shape.

A blocking view takes the file it reads, not a uuid, and lives beside its kind: renders in [rendering/views.py](../../../src/ptychodus/webapp/rendering/views.py), reports in [service/analysis.py](../../../src/ptychodus/webapp/service/analysis.py).

### 2. Schemas and converters

Response models go in [service/schemas.py](../../../src/ptychodus/webapp/service/schemas.py); reuse `Page[T]` for listings. ORM row → model conversion that needs a session goes in [service/convert.py](../../../src/ptychodus/webapp/service/convert.py). A parameter description both transports show is a `*_DESCRIPTION` constant there too.

A model whose float fields can hold NaN or infinity serializes them as `null`; give its tool `output_schema=nullable_numbers_schema(<Model>)`, or an MCP client refuses the response.

### 3. REST route

In the resource's router (e.g. [routers/product.py](../../../src/ptychodus/webapp/routers/product.py)), a route is a signature and one call:

```python
@router.get(
    '/{uuid}/<thing>',
    response_model=<Thing>Read,
    openapi_extra=openapi_operation(ResourceService.get_<thing>),
)
async def get_<thing>(
    uuid: UUID, service: ServiceDep, <tunable>: int = Query(8, ge=1)
) -> <Thing>Read:
    return await service.get_<thing>(uuid, <tunable>=<tunable>)
```

`openapi_operation` states the operation class as `x-operation-class`. A new router module is registered in `create_app()` in [app.py](../../../src/ptychodus/webapp/app.py) and in the `app_client` fixture of [tests/webapp/conftest.py](../../../tests/webapp/conftest.py).

### 4. MCP tool

In `create_mcp_server()`, the tool takes the route's parameters with the same names, types, defaults and bounds — bounds as `Annotated[int, Field(ge=1)]` where the route has `Query(ge=1)`:

```python
@mcp.tool(**_tool_options(service.get_<thing>, 'Get <thing>'))
async def get_<thing>(uuid: UUID, <tunable>: Annotated[int, Field(ge=1)] = 8) -> <Thing>Read:
    return await _served(service.get_<thing>(uuid, <tunable>=<tunable>))
```

`_tool_options` gives the tool its title, the read-only annotations and the description from the service method. `_served` turns the service's errors into the `ToolError` text that matches REST's detail. A render tool returns `_rendered_result(...)` with `output_schema=RenderedImageInfo.model_json_schema()`.

### 5. TypeScript client

A field added to a mirrored `*Read` model must also be declared on the matching interface in [ui/src/api.ts](../../../src/ptychodus/webapp/ui/src/api.ts), or the browser cannot reach it; `tests/webapp/test_ts_client_parity.py` names the missing field. A mirror may inherit through `extends` (`RowBase`, `MetricsRead`, `ComparisonRead`). `CampaignRead` and `LineageRead` are deliberately unmirrored. A new render parameter also goes on `RenderParams` and into `renderQuery`.

Rebuild the frontend after touching `ui/src/`: `cd src/ptychodus/webapp/ui && tsc`.

### 6. Parity cases

Add the pair to `_ROUTES` in `test_rest_mcp_parity.py`, with a data case, the refusals the route makes (404, 400) and the arguments it rejects (422). The parameter-schema and operation-class checks then cover the pair automatically.

### 7. DB model (only if adding a new resource type)

A genuinely new resource also needs a SQLAlchemy model in `src/ptychodus/webapp/db/models.py`, a `ResourceKind` member in `src/ptychodus/api/storage.py`, and an ingestion path in `src/ptychodus/webapp/ingest/`.

## Testing

`uv run pytest tests/webapp/` runs both parity suites. Against a running webapp (see `webapp-dev`):

```sh
curl -sf http://localhost:8000/api/v1/<resource>/<uuid>/<thing>
```

With `PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED=true`, as `webapp-dev` launches it, MCP needs no token; otherwise send `Authorization: Bearer <token>` from `ptychodus-webapp mint-token`.

## Do not

- Do not import from `ptychodus.model`, `ptychodus.view` or `ptychodus.controller` — the webapp reads from `ptychodus.api` only.
- Do not put query or business logic in a router or a tool; it belongs in the service, once.
- Do not add write endpoints without user confirmation — the webapp surface is intentionally read-only.
