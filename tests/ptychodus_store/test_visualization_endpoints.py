from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from ptychodus_store.ingest.pipeline import ingest_manifest

pytestmark = pytest.mark.asyncio


async def _ingest(db_engine, layout, manifest_path) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await ingest_manifest(session, layout, manifest_path)
        await session.commit()


def _decode_png(body: dict, expected_h: int, expected_w: int) -> Image.Image:
    assert body['mime_type'] == 'image/png'
    assert body['shape_h_px'] == expected_h
    assert body['shape_w_px'] == expected_w
    png_bytes = base64.b64decode(body['png_base64'])
    image = Image.open(BytesIO(png_bytes))
    image.load()
    assert image.size == (expected_w, expected_h)
    return image


async def test_visualization_options(app_client) -> None:  # type: ignore[no-untyped-def]
    resp = await app_client.get('/api/v1/visualization/options')
    assert resp.status_code == 200
    body = resp.json()
    assert body['transforms'] == ['identity', 'sqrt', 'log2', 'log', 'log10']
    assert 'amplitude' in body['components']
    assert 'phase_rad' in body['components']
    assert 'hsv_value' in body['color_models']
    assert len(body['colormaps_linear']) > 0
    assert 'gray' in body['colormaps_linear'] or 'gray' in body['colormaps_cyclic']


async def test_diffraction_pattern_image(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(
        f'/api/v1/diffraction/{d}/patterns/0/image',
        params={'colormap': 'gray', 'transform': 'identity'},
    )
    assert resp.status_code == 200
    body = resp.json()
    _decode_png(body, expected_h=8, expected_w=12)
    assert body['pixel_width_m'] == pytest.approx(55e-6)


async def test_diffraction_pattern_out_of_range(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(f'/api/v1/diffraction/{d}/patterns/999/image')
    assert resp.status_code == 404


async def test_diffraction_aggregate_image(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(f'/api/v1/diffraction/{d}/patterns/aggregate/image')
    assert resp.status_code == 200
    _decode_png(resp.json(), expected_h=8, expected_w=12)


async def test_probe_image(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(
        f'/api/v1/product/{p}/probe/image',
        params={'component': 'amplitude', 'colormap': 'gray'},
    )
    assert resp.status_code == 200
    _decode_png(resp.json(), expected_h=8, expected_w=8)


async def test_probe_image_cylindrical(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(
        f'/api/v1/product/{p}/probe/image', params={'color_model': 'hsv_value'}
    )
    assert resp.status_code == 200


async def test_probe_image_component_and_color_model_rejected(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(
        f'/api/v1/product/{p}/probe/image',
        params={'component': 'amplitude', 'color_model': 'hsv_value'},
    )
    assert resp.status_code == 400


async def test_probe_incoherent_out_of_range(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(f'/api/v1/product/{p}/probe/image', params={'incoherent': 42})
    assert resp.status_code == 404


async def test_probe_modes_image(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(f'/api/v1/product/{p}/probe/modes/image')
    assert resp.status_code == 200
    # Fixture writes a single incoherent mode of shape (8, 8); tiled width == 8.
    _decode_png(resp.json(), expected_h=8, expected_w=8)


async def test_object_layer_image(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(
        f'/api/v1/product/{p}/object/0/image', params={'component': 'phase_rad'}
    )
    assert resp.status_code == 200
    _decode_png(resp.json(), expected_h=16, expected_w=16)


async def test_object_layer_out_of_range(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(db_engine, layout, layout.manifest_path('product', p))

    resp = await app_client.get(f'/api/v1/product/{p}/object/99/image')
    assert resp.status_code == 404


async def test_fluorescence_element_image(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_fluorescence
) -> None:
    f = seed_fluorescence(elements=['Fe', 'Cu'])
    await _ingest(db_engine, layout, layout.manifest_path('fluorescence', f))

    resp = await app_client.get(f'/api/v1/fluorescence/{f}/elements/Fe/image')
    assert resp.status_code == 200
    _decode_png(resp.json(), expected_h=6, expected_w=10)


async def test_fluorescence_element_missing(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_fluorescence
) -> None:
    f = seed_fluorescence(elements=['Fe', 'Cu'])
    await _ingest(db_engine, layout, layout.manifest_path('fluorescence', f))

    resp = await app_client.get(f'/api/v1/fluorescence/{f}/elements/Au/image')
    assert resp.status_code == 404
    detail = resp.json()['detail']
    assert 'Fe' in detail['available']


async def test_fluorescence_with_product_pixel_geometry(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_product, seed_fluorescence
) -> None:
    p = seed_product()
    f = seed_fluorescence(elements=['Fe'])
    for m in (
        layout.manifest_path('product', p),
        layout.manifest_path('fluorescence', f),
    ):
        await _ingest(db_engine, layout, m)

    resp = await app_client.get(
        f'/api/v1/fluorescence/{f}/elements/Fe/image', params={'product_uuid': str(p)}
    )
    assert resp.status_code == 200
    body = resp.json()
    # Product fixture writes object pixel geometry = 1e-9 m
    assert body['pixel_width_m'] == pytest.approx(1e-9)


async def test_bad_colormap_returns_400(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(
        f'/api/v1/diffraction/{d}/patterns/0/image', params={'colormap': 'not-a-colormap'}
    )
    assert resp.status_code == 400
    detail = resp.json()['detail']
    assert 'valid_linear' in detail
    assert 'valid_cyclic' in detail


async def test_bad_transform_returns_400(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(
        f'/api/v1/diffraction/{d}/patterns/0/image', params={'transform': 'cbrt'}
    )
    assert resp.status_code == 400


async def test_component_on_real_endpoint_rejected(  # type: ignore[no-untyped-def]
    app_client, db_engine, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(db_engine, layout, layout.manifest_path('diffraction', d))

    resp = await app_client.get(
        f'/api/v1/diffraction/{d}/patterns/0/image', params={'component': 'amplitude'}
    )
    assert resp.status_code == 400
