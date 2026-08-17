"""FastAPI app factory: wires DB, watcher, routers, and MCP server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ptychodus_store.config import Settings, get_settings
from ptychodus_store.db.session import SessionProvider, create_engine, create_schema
from ptychodus_store.ingest.reconciler import full_rescan
from ptychodus_store.ingest.watcher import ManifestWatcher
from ptychodus_store.mcp_server import bind_layout, bind_session_provider, create_mcp_server
from ptychodus_store.routers import (
    admin,
    campaign,
    diffraction,
    fluorescence,
    health,
    lineage,
    visualization,
)
from ptychodus_store.routers import product as product_router
from ptychodus_store.storage.layout import StoreLayout

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
    mcp = create_mcp_server()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await create_schema(engine)
        bind_session_provider(session_provider)
        bind_layout(layout)
        layout.ensure_kind_dirs()

        watcher: ManifestWatcher | None = None
        if settings.auto_reconcile_on_startup:
            async with session_provider.session_factory() as session:
                counts = await full_rescan(session, layout)
            logger.info('startup reconcile counts: %s', counts)

        loop = asyncio.get_running_loop()
        watcher = ManifestWatcher(
            layout,
            session_provider.session_factory,
            loop,
            polling_interval_s=settings.polling_interval_s,
            debounce_window_s=settings.debounce_window_s,
        )
        watcher.start()
        app.state.watcher = watcher

        try:
            yield
        finally:
            if watcher is not None:
                watcher.stop()
            await session_provider.dispose()

    app = FastAPI(title='ptychodus-store', version='0.1.0', lifespan=lifespan)
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

    # Mount the fastmcp HTTP app at the configured path
    try:
        mcp_app = mcp.http_app(path='/')
        app.mount(settings.mcp_mount_path, mcp_app)
    except Exception:  # noqa: BLE001
        logger.exception('failed to mount MCP server; continuing without it')

    ui_dir = Path(__file__).parent / 'ui'
    if ui_dir.is_dir():
        app.mount('/ui', StaticFiles(directory=ui_dir, html=True), name='ui')

        @app.get('/', include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url='/ui/')
    else:
        logger.warning('ui directory not found at %s; skipping /ui mount', ui_dir)

    return app
