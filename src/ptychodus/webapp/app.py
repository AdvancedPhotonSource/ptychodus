"""FastAPI app factory: wires DB, watcher, routers, and MCP server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastmcp.server.http import StarletteWithLifespan

from ptychodus import __version__
from ptychodus.webapp.auth import ClientTokenVerifier, load_client_tokens
from ptychodus.webapp.config import Settings, get_settings
from ptychodus.webapp.db.session import SessionProvider, create_engine, create_schema
from ptychodus.webapp.ingest.reconciler import full_rescan
from ptychodus.webapp.ingest.watcher import ManifestWatcher
from ptychodus.webapp.mcp_server import create_mcp_server
from ptychodus.webapp.rendering.offload import RenderRunner
from ptychodus.webapp.routers import (
    admin,
    campaign,
    diffraction,
    fluorescence,
    health,
    lineage,
    visualization,
)
from ptychodus.webapp.routers import product as product_router
from ptychodus.webapp.routers.errors import add_service_error_handlers
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.api.storage import StorageLayout

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )


def _mcp_token_verifier(settings: Settings) -> ClientTokenVerifier | None:
    """The verifier of MCP bearer tokens, or None to serve MCP to every caller.

    Raises:
        RuntimeError: If neither a token file nor unauthenticated MCP is configured.
    """
    if settings.mcp_tokens_file is not None:
        verifier = ClientTokenVerifier(load_client_tokens(settings.mcp_tokens_file))
        logger.info('MCP accepts bearer tokens of clients: %s', ', '.join(verifier.client_ids))
        return verifier

    if settings.mcp_allow_unauthenticated:
        logger.warning(
            'PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED is set, so MCP authenticates no one; '
            'every caller that reaches the webapp may use it.'
        )
        return None

    raise RuntimeError(
        'MCP requires bearer tokens: set PTYCHODUS_WEBAPP_MCP_TOKENS_FILE to a token file, '
        'which `ptychodus-webapp mint-token <client_id>` creates, or, for development only, '
        'set PTYCHODUS_WEBAPP_MCP_ALLOW_UNAUTHENTICATED=true.'
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    layout = StorageLayout(settings.storage_root)
    engine = create_engine(settings.database_url)
    session_provider = SessionProvider(engine)
    render_runner = RenderRunner(settings.render_concurrency)
    service = ResourceService(session_provider, layout, render_runner)
    mcp_auth = _mcp_token_verifier(settings)

    # Built before the lifespan below so that lifespan can run the MCP app's own: a
    # mounted Starlette app's lifespan is never run by its parent, and without it the
    # streamable-HTTP session manager has no task group and every request fails.
    mcp_app: StarletteWithLifespan | None = None
    try:
        mcp_app = create_mcp_server(service, auth=mcp_auth).http_app(path='/')
    except Exception:  # noqa: BLE001
        logger.exception('failed to build MCP server; continuing without it')

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if mcp_app is not None:
                await stack.enter_async_context(mcp_app.lifespan(mcp_app))

            await create_schema(engine)
            layout.ensure_kind_dirs()

            if settings.auto_reconcile_on_startup:
                async with session_provider.session_factory() as session:
                    counts = await full_rescan(session, layout)
                logger.info('startup reconcile counts: %s', counts)

            watcher = ManifestWatcher(
                layout,
                session_provider.session_factory,
                asyncio.get_running_loop(),
                polling_interval_s=settings.polling_interval_s,
                debounce_window_s=settings.debounce_window_s,
            )
            watcher.start()
            app.state.watcher = watcher

            try:
                yield
            finally:
                watcher.stop()
                await session_provider.dispose()

    app = FastAPI(title='ptychodus-webapp', version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.layout = layout
    app.state.session_provider = session_provider
    app.state.service = service
    add_service_error_handlers(app)

    # Fixed, because the web UI's client spells it.
    api_prefix = '/api/v1'
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(campaign.router, prefix=api_prefix)
    app.include_router(diffraction.router, prefix=api_prefix)
    app.include_router(product_router.router, prefix=api_prefix)
    app.include_router(fluorescence.router, prefix=api_prefix)
    app.include_router(lineage.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(visualization.router, prefix=api_prefix)

    if mcp_app is not None:
        app.mount(settings.mcp_mount_path, mcp_app)

    ui_dir = Path(__file__).parent / 'ui'
    if ui_dir.is_dir():
        app.mount('/ui', StaticFiles(directory=ui_dir, html=True), name='ui')

        @app.get('/', include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url='/ui/')
    else:
        logger.warning('ui directory not found at %s; skipping /ui mount', ui_dir)

    return app
