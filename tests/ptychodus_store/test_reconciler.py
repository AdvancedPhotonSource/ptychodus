from __future__ import annotations

import shutil

import pytest

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.ingest.reconciler import full_rescan

pytestmark = pytest.mark.asyncio


async def test_full_rescan_counts(  # type: ignore[no-untyped-def]
    db_session, layout, seed_campaign, seed_diffraction, seed_product, seed_fluorescence
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


async def test_full_rescan_deletes_stale(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction
):
    d1 = seed_diffraction()
    d2 = seed_diffraction()
    await full_rescan(db_session, layout)
    assert await repo.get_row(db_session, 'diffraction', d1) is not None
    assert await repo.get_row(db_session, 'diffraction', d2) is not None

    # Remove d2 from disk
    shutil.rmtree(layout.resource_folder('diffraction', d2))
    await full_rescan(db_session, layout)
    assert await repo.get_row(db_session, 'diffraction', d1) is not None
    assert await repo.get_row(db_session, 'diffraction', d2) is None


async def test_rescan_resolves_forward_refs(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction, seed_product
):
    d = seed_diffraction()
    p = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    await full_rescan(db_session, layout)

    row = await repo.get_row(db_session, 'product', p)
    assert row is not None
    assert row.ingest_state == IngestState.VALID
