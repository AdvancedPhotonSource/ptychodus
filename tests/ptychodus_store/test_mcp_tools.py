from __future__ import annotations

import base64
from io import BytesIO

import pytest
from fastmcp.exceptions import ToolError
from PIL import Image

from ptychodus_store.ingest.pipeline import ingest_manifest
from ptychodus_store.mcp_server import bind_layout, bind_session_provider, create_mcp_server

pytestmark = pytest.mark.asyncio


def _decode_image_content(result, expected_h: int, expected_w: int) -> Image.Image:  # type: ignore[no-untyped-def]
    assert len(result.content) == 1
    item = result.content[0]
    assert item.type == 'image'
    assert item.mimeType == 'image/png'
    png_bytes = base64.b64decode(item.data)
    image = Image.open(BytesIO(png_bytes))
    image.load()
    assert image.size == (expected_w, expected_h)
    return image


async def test_mcp_tools_registered_and_callable(  # type: ignore[no-untyped-def]
    session_provider,
    layout,
    seed_campaign,
    seed_diffraction,
    seed_product,
):
    # Seed a tiny graph
    c = seed_campaign(sample_name='alpha')
    d = seed_diffraction(campaign_uuid=c)
    p = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d)}])
    async with session_provider.session_factory() as session:
        await ingest_manifest(session, layout, layout.manifest_path('campaign', c))
        await ingest_manifest(session, layout, layout.manifest_path('diffraction', d))
        await ingest_manifest(session, layout, layout.manifest_path('product', p))
        await session.commit()

    bind_session_provider(session_provider)
    bind_layout(layout)
    mcp = create_mcp_server()
    tools_list = await mcp.list_tools()
    names = {t.name for t in tools_list}
    expected = {
        'list_campaign',
        'get_campaign',
        'list_diffraction',
        'get_diffraction',
        'list_product',
        'get_product',
        'list_fluorescence',
        'get_fluorescence',
        'get_lineage',
        'get_store_stats',
        'get_visualization_options',
        'render_diffraction_pattern',
        'render_diffraction_aggregate',
        'render_probe',
        'render_probe_modes',
        'render_object_layer',
        'render_fluorescence_element',
    }
    assert expected.issubset(names), f'missing MCP tools: {expected - names}'

    stats_result = await mcp.call_tool('get_store_stats', {})
    payload = stats_result.structured_content
    assert payload['campaign_count'] == 1
    assert payload['diffraction_count'] == 1
    assert payload['product_count'] == 1

    lineage_result = await mcp.call_tool('get_lineage', {'uuid': str(p)})
    payload = lineage_result.structured_content
    # Optional return types are wrapped as {'result': ...} by fastmcp
    lineage_payload = payload['result'] if 'result' in payload else payload
    assert lineage_payload is not None
    ancestor_uuids = {a['uuid'] for a in lineage_payload['ancestors']}
    assert str(d) in ancestor_uuids


@pytest.fixture
def bound_mcp(session_provider, layout):  # type: ignore[no-untyped-def]
    bind_session_provider(session_provider)
    bind_layout(layout)
    return create_mcp_server()


async def _ingest(session_provider, layout, manifest_path) -> None:  # type: ignore[no-untyped-def]
    async with session_provider.session_factory() as session:
        await ingest_manifest(session, layout, manifest_path)
        await session.commit()


async def test_mcp_get_visualization_options(bound_mcp) -> None:  # type: ignore[no-untyped-def]
    result = await bound_mcp.call_tool('get_visualization_options', {})
    payload = result.structured_content
    assert 'amplitude' in payload['components']
    assert 'hsv_value' in payload['color_models']
    assert payload['transforms'] == ['identity', 'sqrt', 'log2', 'log', 'log10']


async def test_mcp_render_diffraction_pattern(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(session_provider, layout, layout.manifest_path('diffraction', d))
    result = await bound_mcp.call_tool(
        'render_diffraction_pattern', {'uuid': str(d), 'index': 0, 'colormap': 'gray'}
    )
    _decode_image_content(result, expected_h=8, expected_w=12)


async def test_mcp_render_diffraction_aggregate(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(session_provider, layout, layout.manifest_path('diffraction', d))
    result = await bound_mcp.call_tool('render_diffraction_aggregate', {'uuid': str(d)})
    _decode_image_content(result, expected_h=8, expected_w=12)


async def test_mcp_render_probe(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(session_provider, layout, layout.manifest_path('product', p))
    result = await bound_mcp.call_tool('render_probe', {'uuid': str(p), 'component': 'amplitude'})
    _decode_image_content(result, expected_h=8, expected_w=8)


async def test_mcp_render_probe_cylindrical(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(session_provider, layout, layout.manifest_path('product', p))
    result = await bound_mcp.call_tool('render_probe', {'uuid': str(p), 'color_model': 'hsv_value'})
    _decode_image_content(result, expected_h=8, expected_w=8)


async def test_mcp_render_probe_modes(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(session_provider, layout, layout.manifest_path('product', p))
    result = await bound_mcp.call_tool('render_probe_modes', {'uuid': str(p)})
    _decode_image_content(result, expected_h=8, expected_w=8)


async def test_mcp_render_object_layer(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(session_provider, layout, layout.manifest_path('product', p))
    result = await bound_mcp.call_tool(
        'render_object_layer', {'uuid': str(p), 'layer': 0, 'component': 'phase_rad'}
    )
    _decode_image_content(result, expected_h=16, expected_w=16)


async def test_mcp_render_fluorescence_element(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_fluorescence
) -> None:
    f = seed_fluorescence(elements=['Fe', 'Cu'])
    await _ingest(session_provider, layout, layout.manifest_path('fluorescence', f))
    result = await bound_mcp.call_tool(
        'render_fluorescence_element', {'uuid': str(f), 'name': 'Fe'}
    )
    _decode_image_content(result, expected_h=6, expected_w=10)


async def test_mcp_render_object_layer_out_of_range(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(session_provider, layout, layout.manifest_path('product', p))
    with pytest.raises(ToolError, match='out of range'):
        await bound_mcp.call_tool('render_object_layer', {'uuid': str(p), 'layer': 99})


async def test_mcp_render_bad_colormap(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_diffraction
) -> None:
    d = seed_diffraction()
    await _ingest(session_provider, layout, layout.manifest_path('diffraction', d))
    with pytest.raises(ToolError, match='invalid colormap'):
        await bound_mcp.call_tool(
            'render_diffraction_pattern',
            {'uuid': str(d), 'index': 0, 'colormap': 'not-a-colormap'},
        )


async def test_mcp_render_probe_component_and_color_model_rejected(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_product
) -> None:
    p = seed_product()
    await _ingest(session_provider, layout, layout.manifest_path('product', p))
    with pytest.raises(ToolError, match='mutually exclusive'):
        await bound_mcp.call_tool(
            'render_probe',
            {'uuid': str(p), 'component': 'amplitude', 'color_model': 'hsv_value'},
        )


async def test_mcp_render_fluorescence_missing_element(  # type: ignore[no-untyped-def]
    bound_mcp, session_provider, layout, seed_fluorescence
) -> None:
    f = seed_fluorescence(elements=['Fe', 'Cu'])
    await _ingest(session_provider, layout, layout.manifest_path('fluorescence', f))
    with pytest.raises(ToolError, match='not found'):
        await bound_mcp.call_tool('render_fluorescence_element', {'uuid': str(f), 'name': 'Au'})
