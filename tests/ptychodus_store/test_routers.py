from __future__ import annotations

import pytest

from ptychodus_store.ingest.pipeline import ingest_manifest

pytestmark = pytest.mark.asyncio


async def _ingest(client, db_engine, layout, manifest_path) -> None:  # type: ignore[no-untyped-def]
    # Use a fresh session from the same engine the app is wired to.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await ingest_manifest(session, layout, manifest_path)
        await session.commit()


async def test_health(app_client) -> None:  # type: ignore[no-untyped-def]
    resp = await app_client.get('/api/v1/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['db'] == 'ok'
    assert body['watcher'] == 'disabled'


async def test_campaign_list_and_get(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_campaign
):
    c = seed_campaign(sample_name='alpha', tags=['benchmark'])
    await _ingest(app_client, db_engine, layout, layout.manifest_path('campaign', c))

    resp = await app_client.get('/api/v1/campaign')
    assert resp.status_code == 200
    body = resp.json()
    assert body['total'] == 1
    assert body['items'][0]['uuid'] == str(c)
    assert body['items'][0]['sample_name'] == 'alpha'

    one = await app_client.get(f'/api/v1/campaign/{c}')
    assert one.status_code == 200
    assert one.json()['uuid'] == str(c)

    miss = await app_client.get('/api/v1/campaign/00000000-0000-0000-0000-000000000000')
    assert miss.status_code == 404


async def test_diffraction_filter_by_campaign(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_campaign, seed_diffraction
):
    c = seed_campaign()
    other = seed_campaign()
    d1 = seed_diffraction(campaign_uuid=c)
    d2 = seed_diffraction(campaign_uuid=other)
    for m in (
        layout.manifest_path('campaign', c),
        layout.manifest_path('campaign', other),
        layout.manifest_path('diffraction', d1),
        layout.manifest_path('diffraction', d2),
    ):
        await _ingest(app_client, db_engine, layout, m)

    resp = await app_client.get('/api/v1/diffraction', params={'campaign_uuid': str(c)})
    assert resp.status_code == 200
    body = resp.json()
    assert body['total'] == 1
    assert body['items'][0]['uuid'] == str(d1)


async def test_product_derived_from_filter(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction, seed_product
):
    d = seed_diffraction()
    p_match = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    p_other = seed_product()
    for m in (
        layout.manifest_path('diffraction', d),
        layout.manifest_path('product', p_match),
        layout.manifest_path('product', p_other),
    ):
        await _ingest(app_client, db_engine, layout, m)

    resp = await app_client.get('/api/v1/product', params={'derived_from_uuid': str(d)})
    assert resp.status_code == 200
    body = resp.json()
    assert body['total'] == 1
    assert body['items'][0]['uuid'] == str(p_match)


async def test_lineage_dag_walk(  # type: ignore[no-untyped-def]
    app_client,
    db_engine,
    layout,
    seed_campaign,
    seed_diffraction,
    seed_product,
    seed_fluorescence,
):
    c = seed_campaign()
    d = seed_diffraction(campaign_uuid=c)
    p = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    f = seed_fluorescence(derived_from=[{'kind': 'product', 'uuid': str(p)}])
    for m in (
        layout.manifest_path('campaign', c),
        layout.manifest_path('diffraction', d),
        layout.manifest_path('product', p),
        layout.manifest_path('fluorescence', f),
    ):
        await _ingest(app_client, db_engine, layout, m)

    # Walk from the fluorescence node — ancestors should reach the diffraction
    resp = await app_client.get(f'/api/v1/lineage/{f}')
    assert resp.status_code == 200
    body = resp.json()
    ancestor_uuids = {a['uuid'] for a in body['ancestors']}
    assert str(p) in ancestor_uuids
    assert str(d) in ancestor_uuids
    assert body['campaign'] is not None
    assert body['campaign']['uuid'] == str(c)

    # Walk from the diffraction node — descendants should include product and fluorescence
    resp2 = await app_client.get(f'/api/v1/lineage/{d}')
    body2 = resp2.json()
    desc_uuids = {x['uuid'] for x in body2['descendants']}
    assert str(p) in desc_uuids
    assert str(f) in desc_uuids


async def test_file_download(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
):
    d = seed_diffraction()
    await _ingest(app_client, db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(f'/api/v1/diffraction/{d}/files/diffraction')
    assert resp.status_code == 200
    assert resp.headers['content-type'] == 'application/x-hdf5'
    assert len(resp.content) > 0


async def test_admin_stats(  # type: ignore[no-untyped-def]
    app_client,
    db_engine,
    layout,
    seed_campaign,
    seed_diffraction,
):
    c = seed_campaign()
    d = seed_diffraction()
    await _ingest(app_client, db_engine, layout, layout.manifest_path('campaign', c))
    await _ingest(app_client, db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get('/api/v1/admin/stats')
    assert resp.status_code == 200
    body = resp.json()
    assert body['campaign_count'] == 1
    assert body['diffraction_count'] == 1
    assert body['product_count'] == 0
    assert body['fluorescence_count'] == 0
