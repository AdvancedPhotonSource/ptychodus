"""Tests for the cache rebuild in ``create_schema``.

``Base.metadata.create_all`` adds missing tables but never alters one that already
exists, so a durable database written before a model gained a column would keep
failing every query that selects it. These pin both halves of the contract: drift
rebuilds, and the absence of drift leaves the rows alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from uuid import uuid4

from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession

# Importing a model also registers every model on Base.metadata, which is what
# create_schema reflects against. The app gets this through its routers.
from ptychodus_store.db.models import Campaign
from ptychodus_store.db.session import create_engine, create_schema

_COLUMN = 'focus_object_distance_m'


async def _product_columns(engine) -> set[str]:  # type: ignore[no-untyped-def]
    async with engine.begin() as conn:
        return await conn.run_sync(
            lambda sync_conn: {c['name'] for c in inspect(sync_conn).get_columns('product')}
        )


def _database_url(tmp_path: Path) -> str:
    return f'sqlite+aiosqlite:///{tmp_path / "store.db"}'


@pytest.mark.asyncio
async def test_missing_column_rebuilds_the_table(tmp_path: Path) -> None:
    """Simulate a cache written before the column existed, then reopen it."""
    engine = create_engine(_database_url(tmp_path))
    await create_schema(engine)

    async with engine.begin() as conn:
        await conn.execute(text(f'ALTER TABLE product DROP COLUMN {_COLUMN}'))

    assert _COLUMN not in await _product_columns(engine)

    await create_schema(engine)

    assert _COLUMN in await _product_columns(engine)
    await engine.dispose()


@pytest.mark.asyncio
async def test_reopening_a_current_database_preserves_rows(tmp_path: Path) -> None:
    """The rebuild must trigger on drift only -- never on every start."""
    engine = create_engine(_database_url(tmp_path))
    await create_schema(engine)

    async with AsyncSession(engine) as session:
        session.add(Campaign(uuid=uuid4(), label='keep me', folder_path=str(tmp_path / 'campaign')))
        await session.commit()

    await create_schema(engine)

    async with AsyncSession(engine) as session:
        labels = (await session.execute(select(Campaign.label))).scalars().all()

    assert list(labels) == ['keep me']
    await engine.dispose()
