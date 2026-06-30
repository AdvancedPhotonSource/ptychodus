from __future__ import annotations

import pytest

from ptychodus_store.ingest.pipeline import ingest_manifest
from ptychodus_store.mcp_server import bind_session_provider, create_mcp_server

pytestmark = pytest.mark.asyncio


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
