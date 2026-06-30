from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ptychodus_store.db.base import Base


def _enable_sqlite_fk_pragma(engine: AsyncEngine) -> None:
    """Ensure FKs are enforced on every SQLite connection."""

    sync_engine = engine.sync_engine
    if not sync_engine.dialect.name.startswith('sqlite'):
        return

    @event.listens_for(sync_engine, 'connect')
    def _set_pragma(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
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


async def create_schema(engine: AsyncEngine) -> None:
    """Create all tables. Safe to call multiple times — idempotent."""
    async with engine.begin() as conn:
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
