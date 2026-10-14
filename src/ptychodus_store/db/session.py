from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from sqlalchemy import Connection, event, inspect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ptychodus_store.db.base import Base

logger = logging.getLogger(__name__)


def _enable_sqlite_fk_pragma(engine: AsyncEngine) -> None:
    """Ensure FKs are enforced on every SQLite connection."""

    sync_engine = engine.sync_engine
    if not sync_engine.dialect.name.startswith('sqlite'):
        return

    @event.listens_for(sync_engine, 'connect')
    def _set_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute('PRAGMA foreign_keys=ON')
        finally:
            cursor.close()


def create_engine(database_url: str) -> AsyncEngine:
    # For in-memory SQLite we want a shared connection across the engine, not the
    # default `:memory:` which gives each connection its own scratch DB. Using
    # StaticPool with `check_same_thread=False` keeps the in-memory schema alive
    # across requests in the same process.
    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {'future': True}
    if database_url.startswith('sqlite+aiosqlite:///:memory:'):
        from sqlalchemy.pool import StaticPool

        engine_kwargs['poolclass'] = StaticPool
        connect_args['check_same_thread'] = False
    engine = create_async_engine(database_url, connect_args=connect_args, **engine_kwargs)
    _enable_sqlite_fk_pragma(engine)
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False, class_=AsyncSession)


def _has_drifted(connection: Connection) -> bool:
    """True when a table exists but is missing a column the models declare."""
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue

        actual = {column['name'] for column in inspector.get_columns(table.name)}
        missing = {column.name for column in table.columns} - actual

        if missing:
            logger.warning(
                'Table %r is missing column(s) %s; rebuilding the cache.',
                table.name,
                ', '.join(sorted(missing)),
            )
            return True

    return False


async def create_schema(engine: AsyncEngine) -> None:
    """Create all tables, rebuilding the cache when its schema has drifted.

    ``create_all`` adds missing tables but never alters one that already exists, so a
    durable database written before a model gained a column would keep failing every
    query that selects it. This database is a derived cache -- every column is either
    manifest-supplied or read from the companion HDF5, and ``full_rescan`` rebuilds all
    of it from disk -- so dropping and recreating costs only the bookkeeping timestamps
    recording when each row was first indexed, and is preferable to carrying migrations
    for a cache. Tables are dropped together rather than individually so foreign keys
    cannot block the rebuild.

    Safe to call multiple times: without drift it is exactly ``create_all``.
    """
    async with engine.begin() as conn:
        if await conn.run_sync(_has_drifted):
            await conn.run_sync(Base.metadata.drop_all)

        await conn.run_sync(Base.metadata.create_all)


class SessionProvider:
    """Holds the engine + session factory so FastAPI deps can grab a session per request."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.session_factory = create_session_factory(engine)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()
