from __future__ import annotations

from collections.abc import Callable
from httpx import AsyncClient
from pathlib import Path
from ptychodus_store.storage.layout import StoreLayout
from sqlalchemy.ext.asyncio import AsyncEngine
from uuid import UUID

import pytest

from ptychodus_store.ingest.pipeline import ingest_manifest

pytestmark = pytest.mark.asyncio


async def _ingest(client, db_engine: AsyncEngine, layout: StoreLayout, manifest_path: Path) -> None:
    # Use a fresh session from the same engine the app is wired to.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await ingest_manifest(session, layout, manifest_path)
        await session.commit()


async def test_health(app_client: AsyncClient) -> None:
    resp = await app_client.get('/api/v1/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['db'] == 'ok'
    assert body['watcher'] == 'disabled'


async def test_campaign_list_and_get(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_campaign: Callable[..., UUID],
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


async def test_diffraction_filter_by_campaign(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_campaign: Callable[..., UUID],
    seed_diffraction: Callable[..., UUID],
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


async def test_product_derived_from_filter(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
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


async def test_lineage_dag_walk(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_campaign: Callable[..., UUID],
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
    seed_fluorescence: Callable[..., UUID],
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


async def test_file_download(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_diffraction: Callable[..., UUID],
):
    d = seed_diffraction()
    await _ingest(app_client, db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(f'/api/v1/diffraction/{d}/files/diffraction')
    assert resp.status_code == 200
    assert resp.headers['content-type'] == 'application/x-hdf5'
    assert len(resp.content) > 0


async def test_admin_stats(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_campaign: Callable[..., UUID],
    seed_diffraction: Callable[..., UUID],
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


async def test_derived_quantities_need_only_metadata(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_product: Callable[..., UUID],
):
    """Five of the nine follow from the product's own columns."""
    p = seed_product()
    await _ingest(app_client, db_engine, layout, layout.manifest_path('product', p))

    body = (await app_client.get(f'/api/v1/product/{p}')).json()

    # 9 keV -> 1.3776e-10 m, and 1e6 photons over 0.1 s.
    assert body['probe_wavenumber_per_m'] == pytest.approx(1.0 / 1.37761e-10, rel=1e-4)
    assert body['probe_photon_flux_per_s'] == pytest.approx(1.0e7)
    assert body['object_plane_propagation_distance_m'] == pytest.approx(1.5)


async def test_detector_dependent_quantities_need_the_lineage_edge(
    app_client: AsyncClient,
    db_engine: AsyncEngine,
    layout: StoreLayout,
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
):
    """product.h5 records no detector pitch, so these three come through derived_from.

    Without the edge they are null rather than invented -- the store reports what it
    cannot determine as unknown, and the UI renders that as an em dash.
    """
    detector_dependent = ('fresnel_number', 'detector_numerical_aperture', 'depth_of_field_m')

    d = seed_diffraction()
    linked = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    orphan = seed_product()
    for m in (
        layout.manifest_path('diffraction', d),
        layout.manifest_path('product', linked),
        layout.manifest_path('product', orphan),
    ):
        await _ingest(app_client, db_engine, layout, m)

    with_edge = (await app_client.get(f'/api/v1/product/{linked}')).json()
    without_edge = (await app_client.get(f'/api/v1/product/{orphan}')).json()

    for field in detector_dependent:
        assert with_edge[field] is not None, field
        assert without_edge[field] is None, field

    # The five that need no detector are present either way.
    assert without_edge['probe_photon_flux_per_s'] == pytest.approx(1.0e7)
