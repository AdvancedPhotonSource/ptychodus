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
from ptychodus.webapp.config import Settings, get_settings
from ptychodus.webapp.db.session import SessionProvider, create_engine, create_schema
from ptychodus.webapp.ingest.reconciler import full_rescan
from ptychodus.webapp.ingest.watcher import ManifestWatcher
from ptychodus.webapp.mcp_server import bind_layout, bind_session_provider, create_mcp_server
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
from ptychodus.webapp.storage.layout import StoreLayout

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    layout = StoreLayout(settings.storage_root)
    engine = create_engine(settings.database_url)
    session_provider = SessionProvider(engine)

    # Built before the lifespan below so that lifespan can run the MCP app's own: a
    # mounted Starlette app's lifespan is never run by its parent, and without it the
    # streamable-HTTP session manager has no task group and every request fails.
    mcp_app: StarletteWithLifespan | None = None
    try:
        mcp_app = create_mcp_server().http_app(path='/')
    except Exception:  # noqa: BLE001
        logger.exception('failed to build MCP server; continuing without it')

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if mcp_app is not None:
                await stack.enter_async_context(mcp_app.lifespan(mcp_app))

            await create_schema(engine)
            bind_session_provider(session_provider)
            bind_layout(layout)
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

    api_prefix = settings.api_prefix
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
