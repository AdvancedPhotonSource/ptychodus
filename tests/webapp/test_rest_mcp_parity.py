"""REST and MCP are one surface in two channels: the same resources answer alike on both.

Each tool is paired with its REST route. Over one shared set of resources, a tool and its
route take the same parameters (names, types, defaults and bounds), return the same data,
and refuse the same requests with the same reason. A route or tool that is not paired fails
here unless it is listed as an intended difference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError, ValidationError
from httpx import AsyncClient

from ptychodus.api.processing import RunState, create_run_record
from ptychodus.api.storage import StorageLayout
from ptychodus.webapp.db.session import SessionProvider
from ptychodus.webapp.ingest.pipeline import ingest_manifest
from ptychodus.webapp.rendering.schemas import PlotImageInfo, RenderedImageInfo


_API = '/api/v1'

# Each tool's REST route.
_ROUTES: dict[str, str] = {
    'list_campaign': '/campaign',
    'get_campaign': '/campaign/{uuid}',
    'list_diffraction': '/diffraction',
    'get_diffraction': '/diffraction/{uuid}',
    'list_product': '/product',
    'get_product': '/product/{uuid}',
    'list_fluorescence': '/fluorescence',
    'get_fluorescence': '/fluorescence/{uuid}',
    'get_lineage': '/lineage/{uuid}',
    'get_resource_stats': '/admin/stats',
    'get_visualization_options': '/visualization/options',
    'render_diffraction_pattern': '/diffraction/{uuid}/patterns/{index}/image',
    'render_diffraction_aggregate': '/diffraction/{uuid}/patterns/aggregate/image',
    'render_probe': '/product/{uuid}/probe/image',
    'render_probe_modes': '/product/{uuid}/probe/modes/image',
    'render_object': '/product/{uuid}/object/image',
    'render_object_layer': '/product/{uuid}/object/{layer}/image',
    'render_probe_positions': '/product/positions/image',
    'render_fluorescence_summary': '/fluorescence/{uuid}/image',
    'render_fluorescence_element': '/fluorescence/{uuid}/elements/{name}/image',
}

# Intended differences. Downloads and administration are not for agents; the xraydb tools
# answer from a physics table, not the storage root.
_REST_ONLY = {
    ('get', '/health'),
    ('get', '/diffraction/{uuid}/files/diffraction'),
    ('get', '/product/{uuid}/files/product'),
    ('get', '/fluorescence/{uuid}/files/fluorescence'),
    ('post', '/admin/reindex'),
    ('post', '/admin/rescan/{kind}/{uuid}'),
}
_MCP_ONLY_PREFIX = 'xraydb_'

# A real-valued image has no complex component or color model; REST refuses them with a
# 400, and the tool does not offer them at all.
_REAL_VALUED_ONLY_ON_REST = {'component', 'color_model'}
_REST_PARAMETERS_NOT_ON_MCP = {
    'render_diffraction_pattern': _REAL_VALUED_ONLY_ON_REST,
    'render_diffraction_aggregate': _REAL_VALUED_ONLY_ON_REST,
    'render_fluorescence_summary': _REAL_VALUED_ONLY_ON_REST,
    'render_fluorescence_element': _REAL_VALUED_ONLY_ON_REST,
}


@dataclass(frozen=True)
class _Resources:
    tagged_campaign: UUID
    other_campaign: UUID
    polarized_diffraction: UUID
    plain_diffraction: UUID
    succeeded_product: UUID
    running_product: UUID
    imported_product: UUID
    fluorescence: UUID


async def _ingest_all(
    session_provider: SessionProvider, layout: StorageLayout, kinds_and_uuids: list[tuple]
) -> None:
    async with session_provider.session_factory() as session:
        for kind, uuid in kinds_and_uuids:
            await ingest_manifest(session, layout, layout.manifest_path(kind, uuid))
        await session.commit()


@pytest_asyncio.fixture
async def resources(
    session_provider: SessionProvider,
    layout: StorageLayout,
    seed_campaign: Callable[..., UUID],
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
    seed_fluorescence: Callable[..., UUID],
) -> _Resources:
    c1 = seed_campaign(sample_name='alpha', tags=['xmcd', '2026'])
    c2 = seed_campaign(sample_name='beta', tags=['other'])
    d1 = seed_diffraction(campaign_uuid=c1, polarization='left_circular')
    d2 = seed_diffraction(campaign_uuid=c2, photon_energy_eV=9000.0)
    p1 = seed_product(
        derived_from=[{'kind': 'diffraction', 'uuid': str(d1)}],
        run=create_run_record(RunState.SUCCEEDED, command='ptychodus-reconstruct'),
    )
    p2 = seed_product(
        derived_from=[
            {'kind': 'diffraction', 'uuid': str(d1)},
            {'kind': 'product', 'uuid': str(p1)},
        ],
        data_file=None,
        run=create_run_record(RunState.RUNNING, command='ptychodus-reconstruct'),
    )
    p3 = seed_product(derived_from=[{'kind': 'diffraction', 'uuid': str(d2)}])
    f1 = seed_fluorescence(
        derived_from=[{'kind': 'product', 'uuid': str(p1)}], elements=['Fe', 'Cu']
    )
    await _ingest_all(
        session_provider,
        layout,
        [
            ('campaign', c1),
            ('campaign', c2),
            ('diffraction', d1),
            ('diffraction', d2),
            ('product', p1),
            ('product', p2),
            ('product', p3),
            ('fluorescence', f1),
        ],
    )
    return _Resources(c1, c2, d1, d2, p1, p2, p3, f1)


def _structured(result: Any) -> dict[str, Any]:
    payload = result.structured_content
    assert payload is not None
    return payload


def _rest_path(tool: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The tool's REST URL with its path parameters filled in, and the rest as a query."""
    template = _ROUTES[tool]
    query = dict(arguments)
    path = template
    for name in list(query):
        if '{' + name + '}' in path:
            path = path.replace('{' + name + '}', str(query.pop(name)))
    return _API + path, query


def _strings(value: Any) -> Any:
    """`value` with uuids as strings, as both transports carry them."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list):
        return [_strings(v) for v in value]
    return value


def _arguments(resources: _Resources, raw: dict[str, Any]) -> dict[str, Any]:
    """`raw` with each `_Resources` field name in a value replaced by its uuid."""
    resolved: dict[str, Any] = {}
    for name, value in raw.items():
        if isinstance(value, str) and hasattr(resources, value):
            value = getattr(resources, value)
        elif isinstance(value, list):
            value = [
                getattr(resources, v) if isinstance(v, str) and hasattr(resources, v) else v
                for v in value
            ]
        resolved[name] = _strings(value)
    return resolved


async def _both(
    app_client: AsyncClient, mcp: FastMCP, tool: str, arguments: dict[str, Any]
) -> tuple[Any, Any]:
    path, query = _rest_path(tool, arguments)
    response = await app_client.get(path, params=query)
    assert response.status_code == 200, response.text
    return response.json(), await mcp.call_tool(tool, arguments)


_DATA_CASES = [
    ('list_campaign', {}),
    ('list_campaign', {'tag': 'xmcd'}),
    ('list_campaign', {'sample_name': 'beta'}),
    ('list_campaign', {'ingest_state': 'VALID'}),
    ('list_campaign', {'limit': 1, 'offset': 1}),
    ('get_campaign', {'uuid': 'tagged_campaign'}),
    ('list_diffraction', {}),
    ('list_diffraction', {'campaign_uuid': 'tagged_campaign'}),
    ('list_diffraction', {'polarization': 'left_circular'}),
    ('list_diffraction', {'photon_energy_eV_min': 8500.0}),
    ('get_diffraction', {'uuid': 'polarized_diffraction'}),
    ('list_product', {}),
    ('list_product', {'derived_from_uuid': 'polarized_diffraction'}),
    ('list_product', {'run_state': ['running', 'succeeded']}),
    ('list_product', {'ingest_state': 'AWAITING_DATA'}),
    ('list_product', {'scattering_regime': 'transmission'}),
    ('list_product', {'limit': 2}),
    ('get_product', {'uuid': 'running_product'}),
    ('list_fluorescence', {}),
    ('list_fluorescence', {'element': 'Fe'}),
    ('list_fluorescence', {'derived_from_uuid': 'succeeded_product'}),
    ('get_fluorescence', {'uuid': 'fluorescence'}),
    ('get_lineage', {'uuid': 'succeeded_product'}),
    ('get_resource_stats', {}),
    ('get_visualization_options', {}),
]


@pytest.mark.parametrize(('tool', 'raw'), _DATA_CASES)
async def test_a_tool_returns_what_its_route_returns(
    app_client: AsyncClient,
    bound_mcp: FastMCP,
    resources: _Resources,
    tool: str,
    raw: dict[str, Any],
) -> None:
    rest, result = await _both(app_client, bound_mcp, tool, _arguments(resources, raw))
    assert _structured(result) == rest


_RENDER_CASES = [
    ('render_diffraction_pattern', {'uuid': 'polarized_diffraction', 'index': 0}),
    ('render_diffraction_aggregate', {'uuid': 'polarized_diffraction', 'transform': 'log10'}),
    ('render_probe', {'uuid': 'succeeded_product'}),
    ('render_probe_modes', {'uuid': 'succeeded_product'}),
    ('render_object', {'uuid': 'succeeded_product', 'component': 'phase_rad'}),
    ('render_object_layer', {'uuid': 'succeeded_product', 'layer': 0, 'colormap': 'fire'}),
    ('render_fluorescence_summary', {'uuid': 'fluorescence', 'product_uuid': 'succeeded_product'}),
    ('render_fluorescence_element', {'uuid': 'fluorescence', 'name': 'Fe'}),
]


@pytest.mark.parametrize(('tool', 'raw'), _RENDER_CASES)
async def test_a_render_tool_returns_the_route_s_image_and_scales(
    app_client: AsyncClient,
    bound_mcp: FastMCP,
    resources: _Resources,
    tool: str,
    raw: dict[str, Any],
) -> None:
    rest, result = await _both(app_client, bound_mcp, tool, _arguments(resources, raw))

    assert [item.type for item in result.content] == ['image']
    assert result.content[0].data == rest['png_base64']
    assert _structured(result) == {name: rest[name] for name in RenderedImageInfo.model_fields}


async def test_the_probe_positions_tool_returns_the_route_s_plot_and_size(
    app_client: AsyncClient, bound_mcp: FastMCP, resources: _Resources
) -> None:
    arguments = _arguments(resources, {'uuid': ['succeeded_product', 'imported_product']})
    rest, result = await _both(app_client, bound_mcp, 'render_probe_positions', arguments)

    assert result.content[0].data == rest['png_base64']
    assert _structured(result) == {name: rest[name] for name in PlotImageInfo.model_fields}


def _tool_error_text(detail: Any) -> str:
    """REST's error detail as the tool states it: the message, then any choices it lists."""
    if isinstance(detail, str):
        return detail

    if 'available' in detail:
        return f'{detail["error"]}; available: [{", ".join(detail["available"])}]'

    choices = [f'{key}: [{", ".join(values)}]' for key, values in detail.items() if key != 'error']
    return '; '.join([detail['error'], *choices])


_MISSING = str(uuid4())

_REFUSAL_CASES = [
    # Unknown resources
    ('get_campaign', {'uuid': _MISSING}, 404),
    ('get_diffraction', {'uuid': _MISSING}, 404),
    ('get_product', {'uuid': _MISSING}, 404),
    ('get_fluorescence', {'uuid': _MISSING}, 404),
    ('get_lineage', {'uuid': _MISSING}, 404),
    ('render_object', {'uuid': _MISSING}, 404),
    ('render_fluorescence_summary', {'uuid': 'fluorescence', 'product_uuid': _MISSING}, 404),
    # A run with no data yet, and lookups inside a file
    ('render_object', {'uuid': 'running_product'}, 404),
    ('render_object_layer', {'uuid': 'succeeded_product', 'layer': 5}, 404),
    ('render_fluorescence_element', {'uuid': 'fluorescence', 'name': 'Au'}, 404),
    # Bad render parameters, checked before the resource is looked up
    ('render_probe', {'uuid': 'succeeded_product', 'colormap': 'no-such-map'}, 400),
    ('render_probe', {'uuid': _MISSING, 'colormap': 'no-such-map'}, 400),
    ('render_object', {'uuid': 'succeeded_product', 'transform': 'cube'}, 400),
    (
        'render_object',
        {'uuid': 'succeeded_product', 'component': 'phase_rad', 'color_model': 'hsv_value'},
        400,
    ),
]


@pytest.mark.parametrize(('tool', 'raw', 'status'), _REFUSAL_CASES)
async def test_a_tool_refuses_what_its_route_refuses_and_says_why(
    app_client: AsyncClient,
    bound_mcp: FastMCP,
    resources: _Resources,
    tool: str,
    raw: dict[str, Any],
    status: int,
) -> None:
    arguments = _arguments(resources, raw)
    path, query = _rest_path(tool, arguments)
    response = await app_client.get(path, params=query)
    assert response.status_code == status, response.text

    with pytest.raises(ToolError) as refusal:
        await bound_mcp.call_tool(tool, arguments)

    assert str(refusal.value) == _tool_error_text(response.json()['detail'])


_INVALID_CASES = [
    ('list_campaign', {'limit': 0}),
    ('list_diffraction', {'limit': 201}),
    ('list_product', {'offset': -1}),
    ('list_campaign', {'ingest_state': 'bogus'}),
    ('list_diffraction', {'polarization': 'sideways'}),
    ('list_diffraction', {'campaign_uuid': 'not-a-uuid'}),
    ('list_product', {'run_state': ['paused']}),
    ('list_fluorescence', {'derived_from_uuid': 'not-a-uuid'}),
    ('get_product', {'uuid': 'not-a-uuid'}),
    ('render_probe', {'uuid': 'succeeded_product', 'incoherent': -1}),
    ('render_probe_positions', {'uuid': ['succeeded_product'], 'width_px': 64}),
    ('render_probe_positions', {'uuid': ['not-a-uuid']}),
]


@pytest.mark.parametrize(('tool', 'raw'), _INVALID_CASES)
async def test_a_tool_rejects_the_arguments_its_route_rejects(
    app_client: AsyncClient,
    bound_mcp: FastMCP,
    resources: _Resources,
    tool: str,
    raw: dict[str, Any],
) -> None:
    arguments = _arguments(resources, raw)
    path, query = _rest_path(tool, arguments)
    assert (await app_client.get(path, params=query)).status_code == 422

    with pytest.raises(ValidationError):
        await bound_mcp.call_tool(tool, arguments)


async def test_the_probe_positions_tool_needs_a_product(bound_mcp: FastMCP) -> None:
    with pytest.raises(ValidationError):
        await bound_mcp.call_tool('render_probe_positions', {'uuid': []})


def _comparable(schema: Any, definitions: dict[str, Any]) -> Any:
    """`schema` with references inlined and prose, titles and null defaults dropped."""
    if isinstance(schema, list):
        return [_comparable(item, definitions) for item in schema]

    if not isinstance(schema, dict):
        return schema

    if '$ref' in schema:
        return _comparable(definitions[schema['$ref'].rsplit('/', 1)[-1]], definitions)

    return {
        key: _comparable(value, definitions)
        for key, value in schema.items()
        if key not in {'title', 'description'} and not (key == 'default' and value is None)
    }


async def _openapi(app_client: AsyncClient) -> dict[str, Any]:
    response = await app_client.get('/openapi.json')
    assert response.status_code == 200
    return response.json()


async def test_a_tool_takes_its_route_s_parameters(
    app_client: AsyncClient, bound_mcp: FastMCP
) -> None:
    spec = await _openapi(app_client)
    tools = {tool.name: tool for tool in await bound_mcp.list_tools()}
    rest_definitions = spec.get('components', {}).get('schemas', {})

    for name, route in _ROUTES.items():
        operation = spec['paths'][_API + route]['get']
        skipped = _REST_PARAMETERS_NOT_ON_MCP.get(name, set())
        rest = {
            p['name']: (p.get('required', False), _comparable(p['schema'], rest_definitions))
            for p in operation.get('parameters', [])
            if p['name'] not in skipped
        }
        schema = tools[name].parameters
        required = set(schema.get('required', []))
        mcp_definitions = schema.get('$defs', {})
        mcp = {
            param: (param in required, _comparable(param_schema, mcp_definitions))
            for param, param_schema in schema.get('properties', {}).items()
        }
        assert mcp == rest, name


async def test_every_route_and_tool_is_paired_or_an_intended_difference(
    app_client: AsyncClient, bound_mcp: FastMCP
) -> None:
    spec = await _openapi(app_client)
    routes = {
        (method, path.removeprefix(_API)) for path, ops in spec['paths'].items() for method in ops
    }
    paired = {('get', route) for route in _ROUTES.values()}
    assert routes - _REST_ONLY == paired

    tools = {tool.name for tool in await bound_mcp.list_tools()}
    assert {t for t in tools if not t.startswith(_MCP_ONLY_PREFIX)} == set(_ROUTES)
