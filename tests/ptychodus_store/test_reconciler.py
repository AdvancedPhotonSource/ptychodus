from __future__ import annotations

from collections.abc import Callable
from ptychodus_store.db.models import Diffraction, Product
from ptychodus_store.storage.layout import StoreLayout
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

import shutil

import pytest

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.ingest.reconciler import full_rescan

pytestmark = pytest.mark.asyncio


async def test_full_rescan_counts(
    db_session: AsyncSession,
    layout: StoreLayout,
    seed_campaign: Callable[..., UUID],
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
    seed_fluorescence: Callable[..., UUID],
):
    c = seed_campaign()
    d = seed_diffraction(campaign_uuid=c)
    p = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    seed_fluorescence(derived_from=[{'kind': 'product', 'uuid': str(p)}])

    counts = await full_rescan(db_session, layout)
    assert counts == {
        'campaign': 1,
        'diffraction': 1,
        'product': 1,
        'fluorescence': 1,
    }


async def test_full_rescan_deletes_stale(
    db_session: AsyncSession, layout: StoreLayout, seed_diffraction: Callable[..., UUID]
):
    d1 = seed_diffraction()
    d2 = seed_diffraction()
    await full_rescan(db_session, layout)
    assert await repo.get_row(db_session, Diffraction, d1) is not None
    assert await repo.get_row(db_session, Diffraction, d2) is not None

    # Remove d2 from disk
    shutil.rmtree(layout.resource_folder('diffraction', d2))
    await full_rescan(db_session, layout)
    assert await repo.get_row(db_session, Diffraction, d1) is not None
    assert await repo.get_row(db_session, Diffraction, d2) is None


async def test_rescan_resolves_forward_refs(
    db_session: AsyncSession,
    layout: StoreLayout,
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
):
    d = seed_diffraction()
    p = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    await full_rescan(db_session, layout)

    row = await repo.get_row(db_session, Product, p)
    assert row is not None
    assert row.ingest_state == IngestState.VALID
