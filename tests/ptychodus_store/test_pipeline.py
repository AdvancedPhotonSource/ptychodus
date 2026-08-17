from __future__ import annotations

import json

import pytest

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.ingest.pipeline import delete_manifest, ingest_manifest


pytestmark = pytest.mark.asyncio


async def test_ingest_diffraction_valid(db_session, layout, seed_diffraction):  # type: ignore[no-untyped-def]
    uuid = seed_diffraction()
    manifest_path = layout.manifest_path('diffraction', uuid)
    await ingest_manifest(db_session, layout, manifest_path)
    await db_session.commit()

    row = await repo.get_row(db_session, 'diffraction', uuid)
    assert row is not None
    assert row.ingest_state == IngestState.VALID
    assert row.probe_energy_eV == 8000.0
    assert row.num_patterns_total == 4
    assert row.pattern_height_px == 8
    assert row.pattern_width_px == 12
    assert row.detector_pixel_width_m is not None


async def test_ingest_diffraction_missing_h5(db_session, layout, seed_diffraction):  # type: ignore[no-untyped-def]
    uuid = seed_diffraction(write_h5=False)
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', uuid))
    await db_session.commit()
    row = await repo.get_row(db_session, 'diffraction', uuid)
    assert row is not None
    assert row.ingest_state == IngestState.MISSING_FILES


async def test_ingest_bad_json(db_session, layout, tmp_storage_root):  # type: ignore[no-untyped-def]
    from uuid import uuid4

    uuid = uuid4()
    folder = tmp_storage_root / 'product' / str(uuid)
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text('not json at all')
    await ingest_manifest(db_session, layout, folder / 'manifest.json')
    await db_session.commit()

    row = await repo.get_row(db_session, 'product', uuid)
    assert row is not None
    assert row.ingest_state == IngestState.INVALID
    assert row.error_message is not None


async def test_derived_from_creates_edge(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction, seed_product
):
    d_uuid = seed_diffraction()
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d_uuid))
    p_uuid = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d_uuid)}])
    await ingest_manifest(db_session, layout, layout.manifest_path('product', p_uuid))
    await db_session.commit()

    edges = await repo.outgoing_edges(db_session, p_uuid)
    assert len(edges) == 1
    assert edges[0].target_uuid == d_uuid
    assert edges[0].target_kind == 'diffraction'

    p_row = await repo.get_row(db_session, 'product', p_uuid)
    assert p_row is not None
    assert p_row.ingest_state == IngestState.VALID


async def test_orphan_then_resolves(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction, seed_product
):
    # Ingest a product whose parent doesn't yet exist
    from uuid import uuid4

    missing_parent = uuid4()
    p_uuid = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(missing_parent)}])
    await ingest_manifest(db_session, layout, layout.manifest_path('product', p_uuid))
    await db_session.commit()

    p = await repo.get_row(db_session, 'product', p_uuid)
    assert p is not None
    assert p.ingest_state == IngestState.ORPHANED

    # Now the parent shows up under the same UUID
    d_uuid = seed_diffraction(uuid=missing_parent)
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d_uuid))
    await db_session.commit()

    p_again = await repo.get_row(db_session, 'product', p_uuid)
    assert p_again is not None
    assert p_again.ingest_state == IngestState.VALID


async def test_delete_manifest_removes_row_and_propagates(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction, seed_product
):
    d_uuid = seed_diffraction()
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d_uuid))
    p_uuid = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d_uuid)}])
    await ingest_manifest(db_session, layout, layout.manifest_path('product', p_uuid))
    await db_session.commit()

    await delete_manifest(db_session, layout, layout.manifest_path('diffraction', d_uuid))
    await db_session.commit()

    assert await repo.get_row(db_session, 'diffraction', d_uuid) is None
    p = await repo.get_row(db_session, 'product', p_uuid)
    assert p is not None
    assert p.ingest_state == IngestState.ORPHANED


async def test_multi_parent_edges(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction, seed_product
):
    d1 = seed_diffraction()
    d2 = seed_diffraction()
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d1))
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d2))

    p_uuid = seed_product(
        derived_from=[
            {'kind': 'diffraction', 'uuid': str(d1)},
            {'kind': 'diffraction', 'uuid': str(d2)},
        ]
    )
    await ingest_manifest(db_session, layout, layout.manifest_path('product', p_uuid))
    await db_session.commit()

    edges = await repo.outgoing_edges(db_session, p_uuid)
    assert {e.target_uuid for e in edges} == {d1, d2}


async def test_rewrite_manifest_shrinks_edges(  # type: ignore[no-untyped-def]
    db_session, layout, seed_diffraction, seed_product
):
    d1 = seed_diffraction()
    d2 = seed_diffraction()
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d1))
    await ingest_manifest(db_session, layout, layout.manifest_path('diffraction', d2))

    p_uuid = seed_product(
        derived_from=[
            {'kind': 'diffraction', 'uuid': str(d1)},
            {'kind': 'diffraction', 'uuid': str(d2)},
        ]
    )
    p_manifest = layout.manifest_path('product', p_uuid)
    await ingest_manifest(db_session, layout, p_manifest)
    await db_session.commit()
    assert len(await repo.outgoing_edges(db_session, p_uuid)) == 2

    # Shrink derived_from to a single parent
    from datetime import datetime, timezone

    p_manifest.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'kind': 'product',
                'uuid': str(p_uuid),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'derived_from': [{'kind': 'diffraction', 'uuid': str(d1)}],
            }
        )
    )
    await ingest_manifest(db_session, layout, p_manifest)
    await db_session.commit()

    edges = await repo.outgoing_edges(db_session, p_uuid)
    assert len(edges) == 1
    assert edges[0].target_uuid == d1
